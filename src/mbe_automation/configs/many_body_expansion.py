from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import ase
from ase.calculators.calculator import Calculator as ASECalculator

import mbe_automation.storage
from mbe_automation.structure.filters import UniqueClustersFilter
import mbe_automation.calculators.electronic

@dataclass(kw_only=True)
class Clusters:
                                   #
                                   # Structure of the crystal from which clusters
                                   # are cleaved. The atomic coordinates remain
                                   # unmodified.
                                   #
    crystal: ase.Atoms | mbe_automation.storage.Structure
                                   #
                                   # Frame index used if crystal is a Structure
                                   # with multiple frames.
                                   #
    frame_index: int = 0
                                   #
                                   # Energy calculator used to distinguish
                                   # crystallographically inequivalent molecules.
                                   # Molecules are cleaved from the crystal lattice
                                   # and their individual potential energies are
                                   # computed. They are considered unique if their
                                   # energies differ by more than the specified
                                   # energy threshold.
                                   #
    calculator: ASECalculator | None = None
                                   #
                                   # Cluster filtering settings. The cutoffs correspond to the
                                   # max(X,Y) min(i∈X, j∈Y) r_ij characteristic distance of the
                                   # cluster, where X, Y are molecules and i, j are their
                                   # respective atoms. Cutoffs are given in Å.
                                   #
    filter: UniqueClustersFilter = field(
        default_factory=lambda: UniqueClustersFilter(
            cluster_types=["monomers", "dimers", "trimers"],
            cutoffs={"dimers": 30.0, "trimers": 15.0}
        )
    )
                                   #
                                   # Energy threshold (eV/atom) used to detect
                                   # nonequivalent molecules in the input unit
                                   # cell.
                                   #
    unique_molecules_energy_thresh: float = 1.0E-5
                                   #
                                   # Directory where files are stored
                                   # at runtime
                                   #
    work_dir: str | Path = "./"
                                   #
                                   # The main result of the calculations:
                                   # a single dataset file with all data computed
                                   # for the physical system
                                   #
    dataset: str | Path | None = None
    root_key: str = "many_body_expansion"
                                   #
                                   # Whether to save the symmetry-unique clusters to .xyz files.
                                   # The .xyz files for individual clusters are saved in a
                                   # dedicated subdirectory of work_dir.
                                   #
    save_xyz: bool = True
                                   #
                                   # Whether to save the symmetry-unique clusters metadata
                                   # (such as symmetry numbers and characteristic distances) to .csv files.
                                   #
    save_csv: bool = True
                                   #
                                   # Whether to save diagnostic plots (e.g. cumulative cluster counts vs distance).
                                   #
    save_plots: bool = True

    def __post_init__(self):
        if isinstance(self.crystal, ase.Atoms):
            self.crystal = mbe_automation.storage.from_ase_atoms(self.crystal)
            
        if not self.crystal.periodic:
            raise ValueError("The input crystal structure must be periodic.")
            
        if not isinstance(self.calculator, ASECalculator):
            raise TypeError(
                "An ASE calculator is required to determine the number of "
                "crystallographically unique molecules in the unit cell. "
                f"Got: {type(self.calculator).__name__}"
            )
            
        self.work_dir = Path(self.work_dir).expanduser()
        if self.dataset is None:
            self.dataset = self.work_dir / "dataset.hdf5"
        else:
            self.dataset = Path(self.dataset).expanduser()
