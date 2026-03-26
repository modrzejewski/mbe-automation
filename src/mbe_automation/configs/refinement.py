from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
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

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.temperatures_K = np.sort(np.atleast_1d(self.temperatures_K))
        if len(self.temperatures_K) > 1:
            diffs = np.diff(self.temperatures_K)
            if np.any(diffs < 1.0E-5):
                 raise ValueError("Numerically close temperatures detected in temperatures_K.")
