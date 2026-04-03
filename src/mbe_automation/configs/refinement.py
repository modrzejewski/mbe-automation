from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np
import numpy.typing as npt

import mbe_automation.calculators

@dataclass(kw_only=True)
class NormalModeRefinement:
    """
    Configuration parameters for normal mode refinement workflows.
    """
    #
    # Path to the experimental CIF file.
    #
    cif_path: str
    #
    # ASE-compatible force calculator or MBE calculator configuration.
    #
    calculator: mbe_automation.calculators.CALCULATORS
    #
    # Number of lowest-frequency groups to optimize individually.
    # Frequencies above this threshold remain fixed.
    #
    n_refined: int | None = None
    #
    # Temperature at which the experimental data (e.g. ADPs) were collected.
    # If None, the temperature will be read from the CIF file.
    #
    reference_temperature_K: float | None = None
    #
    # Threshold for maximum residual force after geometry 
    # relaxation during refinement (eV/Angs).
    #
    max_force_on_atom_eV_A: float = 1.0E-4
    #
    # Range of temperatures (K) at which phonons
    # and thermodynamic properties are computed.
    #
    temperatures_K: float | npt.NDArray[np.float64] = field(default_factory=lambda: np.array([298.15]))
    #
    # Directory where files are stored at runtime.
    #
    work_dir: str | Path = "./"
    #
    # The main result of the calculations:
    # a single dataset file with all data computed
    # for the physical system.
    #
    dataset: str = "./properties.hdf5"
    #
    # Root location in the dataset hierarchical structure.
    #
    root_key: str = "quasi_harmonic"
    #
    # Whether to save the resulting thermodynamic properties
    # to a CSV file.
    #
    save_csv: bool = True
    #
    # Verbosity of the program's output.
    # 0 -> suppressed warnings
    #
    verbose: int = 0
    #
    # Criterion used to select the winning combination of refinement
    # strategy and restraint scheme from the grid search.
    # "mean_s12"  — minimise the mean s12 similarity index over non-H atoms
    # "rmsd"      — minimise the normalised ADP residual norm (RMSD)
    #
    best_strategy_criterion: Literal["mean_s12", "rmsd"] = "mean_s12"

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.temperatures_K = np.sort(np.atleast_1d(self.temperatures_K))
        if len(self.temperatures_K) > 1:
            diffs = np.diff(self.temperatures_K)
            if np.any(diffs < 1.0E-5):
                 raise ValueError("Numerically close temperatures detected in temperatures_K.")
