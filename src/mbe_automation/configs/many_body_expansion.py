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


@dataclass(kw_only=True)
class MultiLevel(Clusters):
                                   #
                                   # Electronic structure theories to schedule
                                   #
    theory: Sequence[str] = (
        "rpa+ph",
        "lno-ccsd(t)",
    )
                                   #
                                   # Cutoff distances (Å) below which higher-level
                                   # LNO-CCSD(T) calculations are scheduled for each
                                   # cluster type. Setting a cutoff to None disables
                                   # higher-level calculations for that cluster type.
                                   # Monomers are always computed at both lower and
                                   # higher levels and must not be included.
                                   #
    switchover_distances: dict[str, float | None] = field(
        default_factory=lambda: {
            "dimers": 7.0,
            "trimers": None,
        }
    )
                                   #
                                   # Basis sets to use in electronic structure calculations
                                   #
    basis_sets: Sequence[str] = (
        "avtz",
        "avqz",
    )
                                   #
                                   # Accuracy tiers for LNO-CCSD(T). Multiple
                                   # settings allow extrapolation to the
                                   # local-approximation-free limit.
                                   #
    lno_accuracy: Sequence[str] = (
        "tight",
        "vtight",
    )
                                   #
                                   # Whether to export quantum-chemical input files and SLURM
                                   # scripts to disk under work_dir/tasks
                                   #
    save_inputs: bool = True
                                   #
                                   # SLURM queue configuration name
                                   #
    queue: str | None = None

    @property
    def methods(self) -> list[str]:
        """
        Generate all configured electronic structure method names.

        Returns:
            List of electronic structure method strings.
        """
        methods = []
        for t in self.theory:
            if (
                t
                in mbe_automation.calculators.electronic.beyond_rpa.THEORY_LEVELS
            ):
                methods.extend([f"{t}_{b}" for b in self.basis_sets])
            elif (
                t in mbe_automation.calculators.electronic.mrcc.THEORY_LEVELS
            ):
                for acc in self.lno_accuracy:
                    methods.extend([f"{t}_{acc}_{b}" for b in self.basis_sets])
            else:
                methods.append(t)
        return methods

    @property
    def low_level_methods(self) -> list[str]:
        """
        Return all configured low-level electronic structure methods.
        """
        return [
            m
            for m in self.methods
            if m in mbe_automation.calculators.electronic.LOW_LEVEL_METHODS
        ]

    @property
    def high_level_methods(self) -> list[str]:
        """
        Return all configured high-level electronic structure methods.
        """
        return [
            m
            for m in self.methods
            if m in mbe_automation.calculators.electronic.HIGH_LEVEL_METHODS
        ]

    def __post_init__(self):
        super().__post_init__()

        for attr in ("theory", "basis_sets", "lno_accuracy"):
            val = getattr(self, attr)
            if isinstance(val, str) or not isinstance(val, Sequence):
                raise TypeError(f"{attr} must be a sequence of strings.")
            setattr(self, attr, tuple(val))

        for method in self.methods:
            if method not in mbe_automation.calculators.electronic.METHODS:
                raise ValueError(
                    f"Unsupported electronic structure method: \"{method}\". "
                    f"Supported methods are: {mbe_automation.calculators.electronic.METHODS}"
                )

        for cluster_type, cutoff in self.switchover_distances.items():
            if cluster_type.startswith("monomers"):
                raise ValueError(
                    "Switchover distance cannot be applied to monomers. "
                    "Monomers are always computed at both lower and higher levels."
                )
            if cutoff is None:
                continue
            if cutoff <= 0.0:
                raise ValueError(
                    f"Switchover distance for \"{cluster_type}\" must be positive, got {cutoff}."
                )
            if cluster_type in self.filter.cutoffs:
                max_cutoff = self.filter.cutoffs[cluster_type]
                if cutoff > max_cutoff:
                    raise ValueError(
                        f"Requested switchover distance {cutoff} Å exceeds generation "
                        f"cutoff ({max_cutoff} Å) for \"{cluster_type}\"."
                    )

