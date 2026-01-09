import torch
import numpy as np
from pathlib import Path
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from mbe_automation.calculators.mace import MACE

class DeltaMACE(MACE):
    """
    Calculator for Baseline + Delta MACE models with identical r_cut.
    """

    def __init__(
        self,
        model_path_baseline,
        model_path_delta,
        device: str | None = None,
        head: str = "Default",
        baseline_head: str = "Default",
    ):
        # Initialize the baseline model using the MACE class
        # This sets self.models[0] to the baseline model
        super().__init__(
            model_path=model_path_baseline,
            device=device,
            head=baseline_head,
        )

        # Overwrite attributes specific to DeltaMACE
        self.model_path_baseline = self.model_path # set by super from model_path_baseline
        self.model_path_delta = Path(model_path_delta).expanduser()
        self.baseline_head = baseline_head
        # MACE.__init__ sets self.head to baseline_head. We overwrite it.
        self.head = head

        self.baseline_model = self.models[0]
        self.delta_model = torch.load(f=self.model_path_delta, map_location=self.device)
        self.delta_model.to(self.device)

        # Ensure double precision (MACE default)
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

        # Update level of theory
        self.level_of_theory = f"delta_mace_{self.architecture}"
        if head != "Default":
             self.level_of_theory += f"_{head}_head"

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # We need to call Calculator.calculate to set atoms and handle changes
        Calculator.calculate(self, atoms)
        
        batch_base = self._atoms_to_batch(atoms)

        # Attributes from MACECalculator
        compute_stress = not getattr(self, "use_compile", False)
        compute_atomic_stresses = getattr(self, "compute_atomic_stresses", False)
        training = getattr(self, "use_compile", False)

        out_base = self.baseline_model(
            self._clone_batch(batch_base).to_dict(),
            compute_stress=compute_stress,
            training=training,
            compute_edge_forces=compute_atomic_stresses,
            compute_atomic_stresses=compute_atomic_stresses,
        )

        out_delta = self.delta_model(
            self._clone_batch(batch_base).to_dict(),
            compute_stress=compute_stress,
            training=training,
            compute_edge_forces=compute_atomic_stresses,
            compute_atomic_stresses=compute_atomic_stresses,
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

    def serialize(self) -> tuple:
        """
        Returns the class and arguments required to reconstruct the calculator.
        Used for passing the calculator to Ray workers.
        """
        return DeltaMACE, {
            "model_path_baseline": self.model_path_baseline,
            "model_path_delta": self.model_path_delta,
            "device": self.device,
            "head": self.head,
            "baseline_head": self.baseline_head,
        }
