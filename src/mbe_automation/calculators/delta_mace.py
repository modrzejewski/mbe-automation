import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from mace.calculators import MACECalculator

class DeltaMACECalculator(MACECalculator):
    """
    Calculator for Baseline + Delta MACE models with identical r_cut.
    """

    def __init__(
        self,
        model_path_baseline,
        model_path_delta,
        device="cpu",
        energy_units_to_eV=1.0,
        length_units_to_A=1.0,
        default_dtype="",
        charges_key="Qs",
        **kwargs
    ):
        super().__init__(
            model_paths=model_path_baseline,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key=charges_key,
            model_type="MACE",
            **kwargs
        )

        self.baseline_model = self.models[0]
        self.delta_model = torch.load(f=model_path_delta, map_location=self.device)
        self.delta_model.to(self.device)

        if default_dtype == "float64":
            self.delta_model = self.delta_model.double()
        elif default_dtype == "float32":
            self.delta_model = self.delta_model.float()

        for param in self.delta_model.parameters():
            param.requires_grad = False

        r_cut_base = float(self.baseline_model.r_max)
        r_cut_delta = float(self.delta_model.r_max)
        if not np.isclose(r_cut_base, r_cut_delta):
            raise ValueError(
                f"Baseline and Delta models must have the same r_cut. "
                f"Got {r_cut_base} and {r_cut_delta}"
            )

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

        energy_tensor = get_summed("energy")
        if energy_tensor is not None:
            self.results["energy"] = energy_tensor.item() * self.energy_units_to_eV
            self.results["free_energy"] = self.results["energy"]

        forces_tensor = get_summed("forces")
        if forces_tensor is not None:
            f_conv = self.energy_units_to_eV / self.length_units_to_A
            self.results["forces"] = forces_tensor.detach().cpu().numpy() * f_conv

        stress_tensor = get_summed("stress")
        if stress_tensor is not None:
            s_conv = self.energy_units_to_eV / self.length_units_to_A**3
            stress_numpy = stress_tensor.detach().cpu().numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress_numpy) * s_conv

        node_e_tensor = get_summed("node_energy")
        if node_e_tensor is not None:
            self.results["energies"] = node_e_tensor.detach().cpu().numpy() * self.energy_units_to_eV
