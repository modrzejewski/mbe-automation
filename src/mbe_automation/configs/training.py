from dataclasses import dataclass, field
from typing import Literal, List
import numpy as np
import numpy.typing as npt
from pathlib import Path
import ase
from ase.calculators.calculator import Calculator as ASECalculator

from mbe_automation.configs.md import ClassicalMD

@dataclass
class TrainingSet:
                                   #
                                   # Initial structure
                                   #
    crystal: ase.Atoms
                                   #
                                   # Energy and forces calculator
                                   #
    calculator: ASECalculator
                                   #
                                   # Technical details for a short MD sampling of
                                   # configurations for delta learning
                                   #
    md_crystal: ClassicalMD = field(
        default_factory=lambda: ClassicalMD(
            ensemble="NPT",
            time_total_fs=100000.0,
            time_step_fs=1.0,
            time_equilibration_fs=1000.0,
            sampling_interval_fs=1000.0,
            supercell_radius=15.0,
        )
    )
                                   # ------------------------------------------------------------------------
                                   # Filter used to select molecules          Size parameter, which controls
                                   # from the PBC structure to create         how many molecules to include
                                   # a finite cluster                 
                                   # ------------------------------------------------------------------------
                                   # closest_to_center_of_mass,               n_molecules      
                                   # closest_to_central_molecule
                                   #
                                   # max_min_distance_to_central_molecule     distance
                                   # max_max_distance_to_central_molecule
                                   #
    finite_subsystem_filter: Literal[
        "closest_to_center_of_mass",
        "closest_to_central_molecule",
        "max_min_distance_to_central_molecule",
        "max_max_distance_to_central_molecule"
    ] = "closest_to_central_molecule"
    finite_subsystem_n_molecules: int | npt.NDArray[np.integer] | None = field(
        default_factory=lambda: np.array([1, 2, 3, 4, 5, 6, 7, 8])
    )
    finite_subsystem_distances: float | npt.NDArray[np.floating] | None  = None
                                   #
                                   # Assert that all molecules in the PBC structure
                                   # have identical elemental composition.
                                   #
                                   # Used only for validation during the clustering
                                   # step. Setting this parameter to False disables
                                   # the sanity check.
                                   #
    assert_identical_composition: bool = True
                                   #
                                   # Target temperature (K)
                                   # and pressure (GPa)
                                   #
    temperature_K: float = 298.15
    pressure_GPa: float = 1.0E-4
                                   #
                                   #
                                   #
    
                                   #
                                   # Directory where files are stored
                                   # at runtime
                                   #
    work_dir: str = "./"
                                   #
                                   # The main result of the calculations:
                                   # a single dataset file with all data computed
                                   # for the physical system
                                   #
    dataset: str = "./properties.hdf5"
                                   #
                                   # Root of the dataset hierarchical structure
                                   #
    root_key: str = "training_set"
                                   #
                                   # Verbosity of the program's output.
                                   # 0 -> suppressed warnings
                                   #
    verbose: int = 0
    save_plots: bool = True
    save_csv: bool = True

    def __post_init__(self):
        """
        Post-initialization processing.
        """

        n_molecules = self.finite_subsystem_n_molecules
        distances = self.finite_subsystem_distances
        
        n_mol_is_set = n_molecules is not None
        dist_is_set = distances is not None

        if n_mol_is_set and dist_is_set:
            raise ValueError(
                "Specify either 'finite_subsystem_n_molecules' or "
                "'finite_subsystem_distances', but not both."
            )
        
        if n_mol_is_set:
            if isinstance(n_molecules, int):
                self.finite_subsystem_n_molecules = np.array([n_molecules])
            n_size = len(self.finite_subsystem_n_molecules)
            self.finite_subsystem_distances = [None] * n_size

        elif dist_is_set:
            if isinstance(distances, float):
                self.finite_subsystem_distances = np.array([distances])
            n_size = len(self.finite_subsystem_distances)
            self.finite_subsystem_n_molecules = [None] * n_size
