import torch
import numpy as np
from pathlib import Path
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from mbe_automation.calculators.mace import MACE
from mace.tools import AtomicNumberTable
from mace.data import config_from_atoms, config_to_atomic_data
from mace.tools.torch_geometric import Batch

class DeltaMACE(MACE):
    """
    Calculator for Baseline + Delta MACE models with identical r_cut.
    """

    def __init__(
        self,
        model_path_baseline: str | Path,
        model_path_delta: str | Path,
        device: str | None = None,
        head: str = "Default"
    ):
        # Initialize the baseline model using the MACE class
        # This sets self.models[0] to the baseline model
        super().__init__(
            model_path=model_path_baseline,
            device=device,
            head=head,
        )

        # Overwrite attributes specific to DeltaMACE
        self.model_path_baseline = self.model_path # set by super from model_path_baseline
        self.model_path_delta = Path(model_path_delta).expanduser()

        self.baseline_model = self.models[0]
        self.delta_model = torch.load(self.model_path_delta, map_location=self.device)
        self.delta_model.to(self.device)

        # Ensure double precision (MACE default)
        self.delta_model = self.delta_model.double()

        for param in self.delta_model.parameters():
            param.requires_grad = False

        self.z_table_delta = AtomicNumberTable([int(z) for z in self.delta_model.atomic_numbers.cpu()])

        r_cut_base = float(self.baseline_model.r_max)
        r_cut_delta = float(self.delta_model.r_max)
        if not np.isclose(r_cut_base, r_cut_delta):
            raise ValueError(
                f"Baseline and Delta models must have the same r_cut. "
                f"Got {r_cut_base} and {r_cut_delta}"
            )

        # Update level of theory
        self.level_of_theory += "+Δ"

    def _atoms_to_batch_with_z_table(self, atoms, z_table, r_cut):
        config = config_from_atoms(atoms)
        data = config_to_atomic_data(config, z_table, cutoff=r_cut)
        batch = Batch.from_data_list([data])
        return batch.to(self.device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # We need to call Calculator.calculate to set atoms and handle changes
        Calculator.calculate(self, atoms, properties, system_changes)
        
        batch_base = self._atoms_to_batch(self.atoms)
        batch_delta = self._atoms_to_batch_with_z_table(
            self.atoms, self.z_table_delta, float(self.delta_model.r_max)
        )

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
            batch_delta.to_dict(),
            compute_stress=compute_stress,
            training=training,
            compute_edge_forces=compute_atomic_stresses,
            compute_atomic_stresses=compute_atomic_stresses,
        )

        self.results = {}

        def _baseline_plus_delta(key):
            v1 = out_base.get(key)
            v2 = out_delta.get(key)
            if v1 is not None and v2 is not None:
                return v1 + v2
            else:
                return None

        energy_tensor = _baseline_plus_delta("energy")
        if energy_tensor is not None:
            self.results["energy"] = energy_tensor.item() * self.energy_units_to_eV
            self.results["free_energy"] = self.results["energy"]

        forces_tensor = _baseline_plus_delta("forces")
        if forces_tensor is not None:
            f_conv = self.energy_units_to_eV / self.length_units_to_A
            self.results["forces"] = forces_tensor.detach().cpu().numpy() * f_conv

        stress_tensor = _baseline_plus_delta("stress")
        if stress_tensor is not None:
            s_conv = self.energy_units_to_eV / self.length_units_to_A**3
            stress_numpy = stress_tensor.detach().cpu().numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress_numpy) * s_conv

        node_e_tensor = _baseline_plus_delta("node_energy")
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
        }
