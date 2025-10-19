from dataclasses import dataclass, field
from typing import Literal, List
import numpy as np
import numpy.typing as npt
from pathlib import Path
import ase
from ase.calculators.calculator import Calculator as ASECalculator

from mbe_automation.configs.md import ClassicalMD
from mbe_automation.structure.clusters import FiniteSubsystemFilter
from mbe_automation.dynamics.harmonic.modes import PhononFilter

@dataclass(kw_only=True)
class PhononSampling:
    force_constants_dataset: str = "./properties.hdf5"
    force_constants_key: str = "training/quasi_harmonic/phonons/crystal[opt atoms,shape]/force_constants"
                                   #
                                   # Energy and forces calculator
                                   #
    calculator: ASECalculator
                                   #
                                   # Rules how to select a subset
                                   # from the full set of phonons
                                   #
    phonon_filter: PhononFilter = field(
        default_factory = lambda: PhononFilter()
    )
                                   #
                                   # Rules how to select a finite molecular
                                   # cluster from the PBC system
                                   #
    finite_subsystem_filter: FiniteSubsystemFilter = field(
        default_factory = lambda: FiniteSubsystemFilter()
    )
    
    temperature_K: float = 298.15
    
    time_step_fs: float = 100.0
    
    n_frames: int = 20
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
    root_key: str = "training/phonon_sampling"


@dataclass(kw_only=True)
class MDSampling:
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

    finite_subsystem_filter: FiniteSubsystemFilter | None = field(
        default_factory = lambda: FiniteSubsystemFilter()
    )

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
    root_key: str = "training/md_sampling"
                                   #
                                   # Verbosity of the program's output.
                                   # 0 -> suppressed warnings
                                   #
    verbose: int = 0
    save_plots: bool = True
    save_csv: bool = True

