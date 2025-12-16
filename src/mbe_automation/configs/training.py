from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List
import numpy as np
import numpy.typing as npt
from pathlib import Path
import ase
from ase.calculators.calculator import Calculator as ASECalculator
from mace.calculators import MACECalculator

from .md import ClassicalMD
from .clusters import FiniteSubsystemFilter
from mbe_automation.dynamics.harmonic.modes import PhononFilter, AMPLITUDE_SCAN_MODES
from mbe_automation.ml.core import FEATURE_VECTOR_TYPES

@dataclass(kw_only=True)
class PhononSampling:
    force_constants_dataset: str = "./properties.hdf5"
    force_constants_key: str = "training/quasi_harmonic/phonons/force_constants/crystal[opt:atoms,shape]"
                                   #
                                   # Calculators used for
                                   # (1) energies and forces
                                   # (2) feature vectors
                                   #
    calculator: ASECalculator
    features_calculator: MACECalculator | None = None
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
                                   #
                                   # Type of the feature vectors for each frame
                                   # of sampled periodic or finite system.
                                   # Feature vectors are required for subsampling
                                   # based on the distances in the feature space.
                                   #
                                   # This setting is ignored unless
                                   # features_calculator is present.
                                   #
    feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES] = "averaged_environments"
                                   #
                                   # Temperature controls the maximum
                                   # amplitude of normal-mode displacements.
                                   # The classical amplitude at a given
                                   # temperature, Ajk(T), is found by forcing
                                   # the classical oscillator to have the same
                                   # average energy as the corresponding quantum
                                   # harmonic oscillator.
                                   #
    temperature_K: float = 298.15
                                   #
                                   # Method for scanning (probing) the normal-mode
                                   # coordinates.
                                   #
                                   # equidistant
                                   # -----------
                                   # Dynamical matrix eigenvectors
                                   # are multiplied by
                                   #
                                   # ζ * Exp(i*k*r) * A_jk
                                   #
                                   # where ζ represents a series of equidistant points
                                   # on the interval (-1, 1). This scanning mode
                                   # should be used to get the projection of the
                                   # potential energy surface onto a selected normal
                                   # coordinate.
                                   #
                                   # random
                                   # ------
                                   # Dynamical matrix eigenvectors are
                                   # multiplied by
                                   #
                                   # ξ_jk * Exp(i*k*r) * A_jk
                                   #
                                   # time_propagation
                                   # ----------------
                                   # Dynamical matrix eigenvectors are
                                   # multiplied by
                                   #
                                   # Exp(-i*omega_jk*t) * Exp(i*k*r) * A_jk
                                   #
                                   # ---
                                   #
                                   # j, k: phonon branch and crystal momentum (k point)
                                   # Ajk: max amplitude
                                   # omega_jk: angular frequency
                                   # r: position in the crystal lattice
                                   # t: series of time steps. The time dependent phase factor
                                   #    of each eigenvector is evaluated at n_frames time
                                   #    points.
                                   # ξ_jk: random number from uniform distribution on (-1, 1)
                                   #    There are n_frames independent random points
                                   #    for each (j,k).
                                   #
    amplitude_scan: Literal[*AMPLITUDE_SCAN_MODES] = "random"
                                   #
                                   # Distance between time points
                                   # (referenced only if amplitude_scan=="time_propagation")
                                   #
    time_step_fs: float = 100.0
                                   #
                                   # Random number generator for randomized
                                   # amplitude sampling. rng=None means that the random
                                   # number sequence will be seeded from operating
                                   # system's entropy in every function call.
                                   #
                                   # (referenced only if amplitude_scan=="random")
                                   #
    rng: np.random.Generator | None = None
                                   #
                                   # Number of periodic system frames
                                   # to generate by phonon sampling
                                   #
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
                                   #
                                   # Verbosity of the program's output.
                                   # 0 -> suppressed warnings
                                   #
    verbose: int = 0

    def __post_init__(self):
        if self.feature_vectors_type != "none" and self.features_calculator is None:
            raise ValueError(
                "A features_calculator must be provided when "
                "feature_vectors_type is not 'none'."
            )
            

@dataclass(kw_only=True)
class MDSampling:
                                   #
                                   # Initial structure
                                   #
    crystal: ase.Atoms
                                   #
                                   # Calculators used for
                                   # (1) energies and forces
                                   # (2) feature vectors
                                   #
    calculator: ASECalculator
    features_calculator: MACECalculator | None = None
                                   #
                                   # Type of the feature vectors for each frame
                                   # of sampled periodic or finite system.
                                   # Feature vectors are required for subsampling
                                   # based on the distances in the feature space.
                                   #
                                   # This setting is ignored unless
                                   # features_calculator is present.
                                   #
    feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES] = "averaged_environments"
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

    temperatures_K: float | npt.NDArray[np.floating] = 298.15
    pressures_GPa: float | npt.NDArray[np.floating] = 1.0E-4
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

    def __post_init__(self):
        self.temperatures_K = np.atleast_1d(self.temperatures_K)
        self.pressures_GPa = np.atleast_1d(self.pressures_GPa)
        
        if self.feature_vectors_type != "none" and self.features_calculator is None:
            raise ValueError(
                "A features_calculator must be provided when "
                "feature_vectors_type is not 'none'."
            )
