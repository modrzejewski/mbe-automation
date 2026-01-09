import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from mbe_automation.calculators.mace import MACE

class DeltaMACECalculator(MACE):
    """
    Calculator for Baseline + Delta MACE models with identical r_cut.
    """

    def __init__(
        self,
        model_path_baseline,
        model_path_delta,
        device=None,
        **kwargs
    ):
        self.model_path_baseline = model_path_baseline
        self.model_path_delta = model_path_delta

        # Initialize MACE (wrapper) which handles device, dtype (float64), and loading baseline
        super().__init__(
            model_path=model_path_baseline,
            device=device,
            head="Default"
        )

        # MACE init loads baseline into self.models[0]
        self.baseline_model = self.models[0]

        # Load Delta model manually
        self.delta_model = torch.load(f=model_path_delta, map_location=self.device)
        self.delta_model.to(self.device)

        # MACE wrapper defaults to float64, so ensure delta is also float64
        self.delta_model = self.delta_model.double()

        for param in self.delta_model.parameters():
            param.requires_grad = False

        r_cut_base = float(self.baseline_model.r_max)
        r_cut_delta = float(self.delta_model.r_max)
        if not np.isclose(r_cut_base, r_cut_delta):
            raise ValueError(
                f"Baseline and Delta models must have the same r_cut. "
                f"Got {r_cut_base} and {r_cut_delta}"
            )

        self.level_of_theory = f"delta_{self.architecture}"

    def serialize(self) -> tuple:
        """
        Returns the class and arguments required to reconstruct the calculator.
        Used for passing the calculator to Ray workers.
        """
        return DeltaMACECalculator, {
            "model_path_baseline": self.model_path_baseline,
            "model_path_delta": self.model_path_delta,
            "device": self.device,
        }

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        
        batch_base = self._atoms_to_batch(atoms)
        compute_stress = not self.use_compile

        out_base = self.baseline_model(
            self._clone_batch(batch_base).to_dict(),
            compute_stress=compute_stress,
            training=self.use_compile,
            compute_edge_forces=self.compute_atomic_stresses,
            compute_atomic_stresses=self.compute_atomic_stresses,
        )

        out_delta = self.delta_model(
            self._clone_batch(batch_base).to_dict(),
            compute_stress=compute_stress,
            training=self.use_compile,
            compute_edge_forces=self.compute_atomic_stresses,
            compute_atomic_stresses=self.compute_atomic_stresses,
        )

        self.results = {}

        def get_summed(key):
            v1 = out_base.get(key)
            v2 = out_delta.get(key)
            if v1 is not None and v2 is not None:
                return v1 + v2
            return v1 if v1 is not None else v2

        # Note: MACE wrapper forces default_dtype="float64", so energy_units_to_eV=1.0 and length_units_to_A=1.0
        # effectively. We assume standard units here.
        energy_units_to_eV = 1.0
        length_units_to_A = 1.0

        energy_tensor = get_summed("energy")
        if energy_tensor is not None:
            self.results["energy"] = energy_tensor.item() * energy_units_to_eV
            self.results["free_energy"] = self.results["energy"]

        forces_tensor = get_summed("forces")
        if forces_tensor is not None:
            f_conv = energy_units_to_eV / length_units_to_A
            self.results["forces"] = forces_tensor.detach().cpu().numpy() * f_conv

        stress_tensor = get_summed("stress")
        if stress_tensor is not None:
            s_conv = energy_units_to_eV / length_units_to_A**3
            stress_numpy = stress_tensor.detach().cpu().numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress_numpy) * s_conv

        node_e_tensor = get_summed("node_energy")
        if node_e_tensor is not None:
            self.results["energies"] = node_e_tensor.detach().cpu().numpy() * energy_units_to_eV
