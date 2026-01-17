from __future__ import annotations
from dataclasses import dataclass, field
import typing
from typing import Tuple, Literal, Sequence, List
from pathlib import Path
import pandas as pd
import numpy as np
import numpy.typing as npt
import ase
from ase.calculators.calculator import Calculator as ASECalculator
from pymatgen.analysis.local_env import NearNeighbors, CutOffDictNN

import mbe_automation.storage
import mbe_automation.common
import mbe_automation.dynamics.md.display
from mbe_automation.configs.execution import Resources
from mbe_automation.configs.clusters import FiniteSubsystemFilter
from mbe_automation.configs.structure import Minimum
from mbe_automation.storage import ForceConstants as _ForceConstants
from mbe_automation.storage import Structure as _Structure
from mbe_automation.storage import Trajectory as _Trajectory
from mbe_automation.storage import MolecularCrystal as _MolecularCrystal
from mbe_automation.storage import FiniteSubsystem as _FiniteSubsystem
from mbe_automation.storage import AtomicReference as _AtomicReference
import mbe_automation.dynamics.harmonic.modes
from mbe_automation.dynamics.harmonic.modes import PhononFilter, ThermalDisplacements
import mbe_automation.ml.core
import mbe_automation.ml.mace
import mbe_automation.calculators
from mbe_automation.calculators import CALCULATORS
import mbe_automation.structure.clusters
from mbe_automation.ml.core import SUBSAMPLING_ALGOS, FEATURE_VECTOR_TYPES
from mbe_automation.storage.core import (
    DATA_FOR_TRAINING,
    CALCULATION_STATUS_UNDEFINED,
    CALCULATION_STATUS_COMPLETED,
    CALCULATION_STATUS_SCF_NOT_CONVERGED,
    CALCULATION_STATUS_FAILED,
)
from mbe_automation.configs.structure import SYMMETRY_TOLERANCE_STRICT, SYMMETRY_TOLERANCE_LOOSE

class _TrainingStructure:
    def to_mace_dataset(
            self,
            save_path: str,
            level_of_theory: str | dict[Literal["target", "baseline"], str],
            atomic_reference: AtomicReference | None = None,
    ) -> None:
        _to_mace_dataset(
            dataset=[self],
            save_path=save_path,
            level_of_theory=level_of_theory,
            atomic_reference=atomic_reference,
        )

@dataclass(kw_only=True)
class AtomicReference(_AtomicReference):
    @classmethod
    def read(cls, dataset: str | Path, key: str) -> AtomicReference:
        return cls(**vars(
            mbe_automation.storage.core.read_atomic_reference(
                dataset=dataset,
                key=key
        )))

    def save(
            self,
            dataset: str | Path,
            key: str,
            overwrite: bool = False,
    ) -> None:
        mbe_automation.storage.core.save_atomic_reference(
            dataset=dataset,
            key=key,
            atomic_reference=self,
            overwrite=overwrite,
        )

    @classmethod
    def from_atomic_numbers(
            cls,
            atomic_numbers: npt.NDArray[np.int64],
            calculator: CALCULATORS
    ) -> AtomicReference:
        """
        Create a new AtomicReference object for a given set of atomic numbers
        and a calculator associated with a given level of theory.
        """
        return cls(
            energies={
                calculator.level_of_theory: mbe_automation.calculators.atomic_energies(
                    calculator=calculator,
                    z_numbers=atomic_numbers,
                )})

class _AtomicEnergiesCalc:
    def atomic_reference(self, calculator: CALCULATORS) -> AtomicReference:
        return AtomicReference.from_atomic_numbers(
            atomic_numbers=self.unique_elements,
            calculator=calculator
        )
        
@dataclass(kw_only=True)
class ForceConstants(_ForceConstants):
    @classmethod
    def read(cls, dataset: str, key: str) -> ForceConstants:
        return cls(**vars(
            mbe_automation.storage.read_force_constants(dataset, key)
        ))
    
    def frequencies_and_eigenvectors(
            self,
            k_point: npt.NDArray[np.floating],
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.complex128]]:
        ph = mbe_automation.storage.to_phonopy(self)
        return mbe_automation.dynamics.harmonic.modes.at_k_point(
            dynamical_matrix=ph.dynamical_matrix,
            k_point=k_point,
        )

    def thermal_displacements(
            self,
            temperature_K: float,
            phonon_filter: PhononFilter | None = None
    ) -> ThermalDisplacements:
        """
        Compute thermal displacement properties of atoms in the primitive cell.
        
        Args:
            temperature_K: Temperature in Kelvin.
            phonon_filter: A PhononFilter object which defines the subset
                of phonons. If None, all phonons up to infinite frequency are included.
        """
        return _thermal_displacements(
            force_constants=self,
            temperature_K=temperature_K,
            phonon_filter=phonon_filter
        )
        
    def to_cif_file(
            self,
            save_path: str,
            save_adps: bool = False,
            temperature_K: float | None = None,
            phonon_filter: PhononFilter | None = None,
    ) -> None:
        """
        Save the primitive cell to a CIF file.
        
        Args:
            save_path: Path to the output CIF file.
            save_adps: Whether to calculate and include anisotropic displacement parameters.
            temperature_K: Temperature in Kelvin (required if save_adps is True).
            phonon_filter: Optional filter for phonons (used if save_adps is True).
        """
        if not save_path.lower().endswith(".cif"):
            raise ValueError("The save_path for to_cif_file must end with .cif")
        _to_cif_file(
            force_constants=self,
            save_path=save_path,
            save_adps=save_adps,
            temperature_K=temperature_K,
            phonon_filter=phonon_filter
        )


@dataclass(kw_only=True)
class Structure(_Structure, _AtomicEnergiesCalc, _TrainingStructure):
    @classmethod
    def read(
            cls,
            dataset: str,
            key: str,
    ) -> Structure:
        return cls(**vars(
            mbe_automation.storage.read_structure(dataset, key)
        ))

    @classmethod
    def from_xyz_file(
            cls,
            read_path: str,
            transform_to_symmetrized_primitive: bool = True,
            symprec: float = SYMMETRY_TOLERANCE_LOOSE,
    ):
        ase_atoms = mbe_automation.storage.from_xyz_file(
            read_path=read_path,
            transform_to_symmetrized_primitive=transform_to_symmetrized_primitive,
            symprec=symprec,
        )        
        return cls(**vars(mbe_automation.storage.from_ase_atoms(ase_atoms)))
        
    def subsample(
            self,
            n: int,
            algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling",
            rng: np.random.Generator | None = None,
    ) -> Structure:
        return Structure(**vars(
            _subsample_structure(self, n, algorithm, rng)
        ))

    def select(
            self,
            indices: npt.NDArray[np.integer] | None = None,
            level_of_theory: str | dict | None = None,
    ) -> Structure | None:
        
        valid_indices = _completed_frames(self, level_of_theory)
        
        if indices is not None:
            final_indices = np.intersect1d(indices, valid_indices)
        else:
            final_indices = valid_indices
        
        if len(final_indices) == 0:
            return None

        return Structure(**vars(
            _select_frames(self, final_indices)
        ))

    def random_split(
            self,
            fractions: Sequence[float],
            rng: np.random.Generator | None = None,
    ) -> Sequence[Structure]:

        return [
            Structure(**vars(x)) for x in _split_frames(self, fractions, rng)
        ]

    def run(
            self,
            calculator: CALCULATORS,
            energies: bool = True,
            forces: bool = True,
            feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
            exec_params: Resources | None = None,
            overwrite: bool = False,
            selected_frames: npt.NDArray[np.int64] | None = None,
            chunk: Tuple[int, int] | None = None,
    ) -> None:
        """
        Run a calculator on the structure.
        
        Args:
            calculator: The calculator class to use.
            energies: Whether to calculate energies.
            forces: Whether to calculate forces.
            feature_vectors_type: Type of feature vectors to compute ("none" to skip).
            exec_params: Execution resources configuration.
            overwrite: Whether to overwrite existing results.
            selected_frames: Specific indices of frames to process.
            chunk: Tuple of (index, total_chunks) for disjoint work distribution.
                   Index is 1-based (1..total_chunks). Using this argument enables
                   parallel execution (e.g. in SLURM job arrays).
        """
        
        final_selected_frames = _resolve_chunk_indices(
            n_frames=self.n_frames,
            chunk=chunk,
            selected_frames=selected_frames,
        )
        
        _run_model(
            structure=self,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            exec_params=exec_params,
            overwrite=overwrite,
            selected_frames=final_selected_frames,
        )

    run_model = run # synonym

    def to_molecular_crystal(
            self,
            reference_frame_index: int = 0,
            assert_identical_composition: bool = True,
            bonding_algo: NearNeighbors | None = None,
    ) -> MolecularCrystal:

        if not self.periodic:
            raise ValueError("Cannot convert a finite structure to a molecular crystal.")
        
        return MolecularCrystal(**vars(
            mbe_automation.structure.clusters.detect_molecules(
                system=self,
                reference_frame_index=reference_frame_index,
                assert_identical_composition=assert_identical_composition,
                bonding_algo=bonding_algo,
            )
        ))

    detect_molecules = to_molecular_crystal # synonym

    def extract_all_molecules(
            self,
            bonding_algo: NearNeighbors | None = None,
            reference_frame_index: int = 0,
            calculator: ASECalculator | None = None,
    ) -> List[Structure]:

        if bonding_algo is None:
            bonding_algo = CutOffDictNN.from_preset("vesta_2019")
        
        return [Structure(**vars(molecule)) for molecule in
                mbe_automation.structure.clusters.extract_all_molecules(
                    crystal=self,
                    bonding_algo=bonding_algo,
                    reference_frame_index=reference_frame_index,
                    calculator=calculator,
                )]

    def extract_unique_molecules(
            self,
            calculator: ASECalculator,
            energy_thresh: float = 1.0E-5, # eV/atom
            bonding_algo: NearNeighbors | None = None,
            reference_frame_index: int = 0,
    ) -> List[Structure]:

        if bonding_algo is None:
            bonding_algo = CutOffDictNN.from_preset("vesta_2019")

        return [Structure(**vars(molecule)) for molecule in
                mbe_automation.structure.clusters.extract_unique_molecules(
                    crystal=self,
                    calculator=calculator,
                    energy_thresh=energy_thresh,
                    bonding_algo=bonding_algo,
                    reference_frame_index=reference_frame_index,
                )]

    def extract_relaxed_unique_molecules(
            self,
            dataset: str,
            key: str,
            calculator: ASECalculator,
            config: Minimum,
            energy_thresh: float = 1.0E-5, # eV/atom
            bonding_algo: NearNeighbors | None = None,
            reference_frame_index: int = 0,
            work_dir: Path | str = Path("./")
    ) -> None:

        if bonding_algo is None:
            bonding_algo = CutOffDictNN.from_preset("vesta_2019")
        
        mbe_automation.structure.clusters.extract_relaxed_unique_molecules(
            dataset=dataset,
            key=key,
            crystal=self,
            calculator=calculator,
            config=config,
            energy_thresh=energy_thresh,
            bonding_algo=bonding_algo,
            reference_frame_index=reference_frame_index,
            work_dir=work_dir,
        )

@dataclass(kw_only=True)
class Trajectory(_Trajectory, _TrainingStructure, _AtomicEnergiesCalc):
    @classmethod
    def read(
            cls,
            dataset: str,
            key: str,
    ) -> Trajectory:
        return cls(**vars(
            mbe_automation.storage.read_trajectory(dataset, key)
        ))

    def subsample(
            self,
            n: int,
            algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling",
            rng: np.random.Generator | None = None,
    ) -> Trajectory:
        return Trajectory(**vars(
            _subsample_trajectory(self, n, algorithm, rng)
        ))

    def run(
            self,
            calculator: CALCULATORS,
            energies: bool = True,
            forces: bool = True,
            feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
            exec_params: Resources | None = None,
            overwrite: bool = False,
            selected_frames: npt.NDArray[np.int64] | None = None,
            chunk: Tuple[int, int] | None = None,
    ) -> None:
        """
        Run a calculator on the trajectory frames.
        
        Args:
            calculator: The calculator class to use.
            energies: Whether to calculate energies.
            forces: Whether to calculate forces.
            feature_vectors_type: Type of feature vectors to compute ("none" to skip).
            exec_params: Execution resources configuration.
            overwrite: Whether to overwrite existing results.
            selected_frames: Specific indices of frames to process.
            chunk: Tuple of (index, total_chunks) for disjoint work distribution.
                   Index is 1-based (1..total_chunks). Using this argument enables
                   parallel execution (e.g. in SLURM job arrays).
        """

        final_selected_frames = _resolve_chunk_indices(
            n_frames=self.n_frames,
            chunk=chunk,
            selected_frames=selected_frames,
        )

        _run_model(
            structure=self,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            exec_params=exec_params,
            overwrite=overwrite,
            selected_frames=final_selected_frames,
        )

    run_model = run

    def display(
            self,
            quantity: Literal["energy_fluctuations"] = "energy_fluctuations",
            save_path: str | None = None
    ):
        if quantity == "energy_fluctuations":
            return _energy_fluctuations(self, save_path)

@dataclass(kw_only=True)
class MolecularCrystal(_MolecularCrystal, _AtomicEnergiesCalc):
    def save(
            self,
            dataset: str,
            key: str,
    ) -> None:
        mbe_automation.storage.core.save_molecular_crystal(
            dataset=dataset,
            key=key,
            system=self,
        )

    @classmethod
    def read(
            cls,
            dataset: str,
            key: str,
    ) -> MolecularCrystal:
        return cls(**vars(
            mbe_automation.storage.core.read_molecular_crystal(dataset, key)
        ))
        
    def subsample(
            self,
            n: int,
            algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling",
            rng: np.random.Generator | None = None,
    ) -> MolecularCrystal:
        return MolecularCrystal(
            supercell=_subsample_structure(self.supercell, n, algorithm, rng),
            index_map=self.index_map,
            centers_of_mass=self.centers_of_mass,
            identical_composition=self.identical_composition,
            n_molecules=self.n_molecules,
            central_molecule_index=self.central_molecule_index,
            min_distances_to_central_molecule=self.min_distances_to_central_molecule,
            max_distances_to_central_molecule=self.max_distances_to_central_molecule
        )

    def extract_finite_subsystems(
            self,
            filter: FiniteSubsystemFilter | None = None,
    ) -> List[FiniteSubsystem]:

        if filter is None:
            filter = FiniteSubsystemFilter()

        clusters = mbe_automation.structure.clusters.extract_finite_subsystem(
            system=self,
            filter=filter
        )
        return [FiniteSubsystem(**vars(s)) for s in clusters]

    extract_finite_subsystem = extract_finite_subsystems # synonym

@dataclass(kw_only=True)
class FiniteSubsystem(_FiniteSubsystem, _AtomicEnergiesCalc, _TrainingStructure):
    @classmethod
    def read(
            cls,
            dataset: str,
            key: str,
    ) -> FiniteSubsystem:
        return cls(**vars(
            mbe_automation.storage.read_finite_subsystem(dataset, key)
        ))

    def save(
            self,
            dataset: str,
            key: str,
            only: List[Literal[*DATA_FOR_TRAINING]] | Literal[*DATA_FOR_TRAINING] | None = None,
    ) -> None:

        mbe_automation.storage.core.save_finite_subsystem(
            dataset=dataset,
            key=key,
            subsystem=self,
            only=only
        )

    def subsample(
            self,
            n: int,
            algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling",
            rng: np.random.Generator | None = None,
    ) -> FiniteSubsystem:
        return FiniteSubsystem(
            cluster_of_molecules=_subsample_structure(self.cluster_of_molecules, n, algorithm, rng),
            molecule_indices=self.molecule_indices,
            n_molecules=self.n_molecules
        )

    def select(
            self,
            indices: npt.NDArray[np.integer] | None = None,
            level_of_theory: str | dict | None = None,
    ) -> FiniteSubsystem | None:
        
        valid_indices = _completed_frames(self.cluster_of_molecules, level_of_theory)
        
        if indices is not None:
            final_indices = np.intersect1d(indices, valid_indices)
        else:
            final_indices = valid_indices
        
        if len(final_indices) == 0:
            return None

        return FiniteSubsystem(
            cluster_of_molecules=_select_frames(self.cluster_of_molecules, final_indices),
            molecule_indices=self.molecule_indices,
            n_molecules=self.n_molecules
        )

    def run(
            self,
            calculator: CALCULATORS,
            energies: bool = True,
            forces: bool = True,
            feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
            exec_params: Resources | None = None,
            overwrite: bool = False,
            selected_frames: npt.NDArray[np.int64] | None = None,
            chunk: Tuple[int, int] | None = None,
    ) -> None:
        """
        Run a calculator on the finite subsystem.
        
        Args:
            calculator: The calculator class to use.
            energies: Whether to calculate energies.
            forces: Whether to calculate forces.
            feature_vectors_type: Type of feature vectors to compute ("none" to skip).
            exec_params: Execution resources configuration.
            overwrite: Whether to overwrite existing results.
            selected_frames: Specific indices of frames to process.
            chunk: Tuple of (index, total_chunks) for disjoint work distribution.
                   Index is 1-based (1..total_chunks). Using this argument enables
                   parallel execution (e.g. in SLURM job arrays).
        """
        
        final_selected_frames = _resolve_chunk_indices(
            n_frames=self.cluster_of_molecules.n_frames,
            chunk=chunk,
            selected_frames=selected_frames,
        )
        
        _run_model(
            structure=self.cluster_of_molecules,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            exec_params=exec_params,
            overwrite=overwrite,
            selected_frames=final_selected_frames,
        )

    run_model = run

    def random_split(
            self,
            fractions: Sequence[float],
            rng: np.random.Generator | None = None,
    ) -> Sequence[FiniteSubsystem]:

        return [
            FiniteSubsystem(
                cluster_of_molecules=s,
                molecule_indices=self.molecule_indices,
                n_molecules=self.n_molecules
            )
            for s in _split_frames(self.cluster_of_molecules, fractions, rng)
        ]

@dataclass
class Dataset(_AtomicEnergiesCalc):
    """
    Collection of atomistic structures or finite subsystems
    for machine learning tasks.
    """
    structures: List[Structure|FiniteSubsystem] = field(default_factory=list)

    def append(self, structure: Structure | FiniteSubsystem):
        """
        Add a structure or finite subsystem to the dataset.
        """
        self.structures.append(structure)

    def statistics(self, level_of_theory: str) -> None:
        _statistics(
            systems=self.structures,
            level_of_theory=level_of_theory,
        )

    def to_mace_dataset(
            self,
            save_path: str,
            level_of_theory: str | dict[Literal["target", "baseline"], str],
            atomic_reference: AtomicReference | None = None,
    ) -> None:
        _to_mace_dataset(
            dataset=self.structures,
            save_path=save_path,
            level_of_theory=level_of_theory,
            atomic_reference=atomic_reference,
        )

    @property
    def unique_elements(self) -> npt.NDArray[np.int64]:
        """
        Return a sorted NumPy array of unique Z numbers for a list of structures.
        """
        unique_elements = [structure.unique_elements for structure in self.structures]
        return np.unique(np.concatenate(unique_elements))

def _completed_frames(structure: _Structure, level_of_theory: str | dict | None) -> npt.NDArray[np.int64]:
    """
    Get indices of frames where calculations for a given level of theory
    are completed.
    
    If level_of_theory is None, returns all indices.

    If level_of_theory matches structure.level_of_theory, it is implied
    that all calculations are COMPLETED (since the structure itself exists).
    
    If level_of_theory is a dictionary (delta learning), the function returns
    indices of frames where calculations for both target and baseline methods
    are completed (intersection of valid frames).

    If level_of_theory does not match structure.level_of_theory covering
    all required methods (target/baseline) and is not present in 
    structure.ground_truth, an empty array is returned.
    """
    if level_of_theory is None:
        return np.arange(structure.n_frames)

    valid_mask = np.ones(structure.n_frames, dtype=bool)

    targets = []
    if isinstance(level_of_theory, dict):
        if "target" in level_of_theory: targets.append(level_of_theory["target"])
        if "baseline" in level_of_theory: targets.append(level_of_theory["baseline"])
    else:
        targets.append(level_of_theory)

    for method in targets:
        #
        # If the requested method matches the level of theory used to generate
        # the structure, we assume that calculations are implicitly COMPLETED.
        # Otherwise, we wouldn't have the structure in the first place.
        #
        if (structure.level_of_theory is not None) and (method == structure.level_of_theory):
            continue

        #
        # If the method is not the structure's level of theory, we look for it
        # in the ground truth data.
        #
        found_in_ground_truth = False
        if structure.ground_truth is not None:
            if method in structure.ground_truth.calculation_status:
                valid_mask &= (
                    structure.ground_truth.calculation_status[method] == CALCULATION_STATUS_COMPLETED
                )
                found_in_ground_truth = True
        
        #
        # If the method corresponds to neither the structure.level_of_theory
        # nor any item in ground_truth, then we return an empty array.
        #
        if not found_in_ground_truth:
            return np.array([], dtype=np.int64)

    return np.nonzero(valid_mask)[0]

def _frames_to_compute(
    structure: _Structure,
    level_of_theory: str,
    overwrite: bool,
    selected_frames: npt.NDArray[np.int64] | None,
) -> npt.NDArray[np.int64]:
    """
    Determine which frames need calculation at a given level_of_theory.
    
    If overwrite is False, skips frames where the data are already
    computed (status COMPLETED at a given level_of_theory)
    If overwrite is True, does not take into account if the frame
    status is COMPLETED or not.
    """
    all_frames = (
        selected_frames if selected_frames is not None 
        else np.arange(structure.n_frames)
    )
    if overwrite:
        return all_frames
    else:
        #
        # (frames to compute) = (all frames) - (completed frames)
        #
        return np.setdiff1d(
            all_frames,
            _completed_frames(structure, level_of_theory),
            assume_unique=True
        )

def _select_frames(
        struct: _Structure,
        indices: npt.NDArray[np.integer]
    ) -> _Structure:
        """
        Return a new Structure containing only the specified frames.
        """
        if len(indices) == 0:
            raise ValueError("Cannot create a Structure with 0 frames.")

        selected_cell_vectors = struct.cell_vectors
        if struct.cell_vectors is not None:
            if struct.cell_vectors.ndim == 3:
                selected_cell_vectors = struct.cell_vectors[indices]

        return _Structure(
            positions=struct.positions[indices],
            atomic_numbers=(
                struct.atomic_numbers[indices] if struct.atomic_numbers.ndim == 2 else
                struct.atomic_numbers
            ),
            masses=(
                struct.masses[indices] if struct.masses.ndim == 2 else
                struct.masses
            ),
            cell_vectors=selected_cell_vectors,
            n_frames=len(indices),
            n_atoms=struct.n_atoms,
            E_pot=(
                struct.E_pot[indices]
                if struct.E_pot is not None else None
            ),
            forces=(
                struct.forces[indices]
                if struct.forces is not None
                else None
            ),
            feature_vectors=(
                struct.feature_vectors[indices]
                if struct.feature_vectors is not None else None
            ),
            feature_vectors_type=struct.feature_vectors_type,
            ground_truth=(
                struct.ground_truth.select_frames(indices)
                if struct.ground_truth is not None else None
            )
        )

def _subsample_structure(
        struct: _Structure,
        n: int,
        algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling",
        rng: np.random.Generator | None = None,
    ) -> _Structure:
        """
        Return new Structure containing a subset of frames selected using
        either either farthest point sampling or K-means sampling.

        The feature vectors used to compute distances in the feature
        space are averaged over atoms and normalized.
        """

        if struct.feature_vectors_type == "none":
            raise ValueError(
                "Subsampling requires precomputed feature vectors. "
                "Execute run_model on your Structure before subsampling."
            )
        if struct.masses.ndim == 2 or struct.atomic_numbers.ndim == 2:
            raise ValueError(
                "subsample cannot work on a structure where atoms "
                "are permuted between frames."
            )
        
        selected_indices = mbe_automation.ml.core.subsample(
            feature_vectors=struct.feature_vectors,
            feature_vectors_type=struct.feature_vectors_type,
            n_samples=n,
            algorithm=algorithm,
            rng=rng,
        )
        
        return _select_frames(struct, selected_indices)

def _split_frames(
        struct: _Structure,
        fractions: Sequence[float],
        rng: np.random.Generator | None = None
) -> Sequence[_Structure]:
    """Split the frames of a structure into multiple structures.

    The split is random. For reproducibility, a random number generator
    can be passed. If not, a new generator is created, seeded from OS entropy.

    Args:
        struct: The structure to split.
        fractions: A sequence of floats that sum to 1.0, representing the
            fraction of frames for each new structure.
        rng: An optional numpy random number generator.

    Returns:
        A sequence of new _Structure objects.
    """
    if any(f < 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Fractions must be non-negative and sum to 1.0")

    if rng is None:
        rng = np.random.default_rng()

    n_total = struct.n_frames
    indices = np.arange(n_total)
    rng.shuffle(indices)

    lengths = [int(f * n_total) for f in fractions]
    lengths[-1] = n_total - sum(lengths[:-1])

    if 0 in lengths:
        raise ValueError(
            f"Splitting {n_total} frames with fractions {fractions} would result "
            f"in at least one empty split, which is not allowed."
        )

    split_points = np.cumsum(lengths[:-1])
    indices_split = np.split(indices, split_points)
    return [
        _select_frames(struct, idx_group) for idx_group in indices_split
    ]

def _subsample_trajectory(
        traj: _Trajectory,
        n: int,
        algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling",
        rng: np.random.Generator | None = None,
) -> _Trajectory:
        """
        Return new Trajectory containing a subset of frames
        selected based on the distances in the feature
        vector space.
        """
        if traj.feature_vectors_type == "none":
            raise ValueError(
                "Subsampling requires precomputed feature vectors. "
                "Execute run_model on your Structure before subsampling."
            )
        
        if traj.masses.ndim == 2 or traj.atomic_numbers.ndim == 2:
            raise ValueError(
                "subsample cannot work on a trajectory where atoms"
                "are permuted between frames."
            )
                
        selected_indices = mbe_automation.ml.core.subsample(
            feature_vectors=traj.feature_vectors,
            feature_vectors_type=traj.feature_vectors_type,
            n_samples=n,
            algorithm=algorithm,
            rng=rng,
        )
        
        selected_cell_vectors = traj.cell_vectors
        if traj.cell_vectors is not None and traj.cell_vectors.ndim == 3:
            selected_cell_vectors = traj.cell_vectors[selected_indices]
                
        return _Trajectory(
            time_equilibration=traj.time_equilibration,
            target_temperature=traj.target_temperature,
            target_pressure=traj.target_pressure,
            ensemble=traj.ensemble,
            n_removed_trans_dof=traj.n_removed_trans_dof,
            n_removed_rot_dof=traj.n_removed_rot_dof,
            n_atoms=traj.n_atoms,
            periodic=traj.periodic,
            atomic_numbers=traj.atomic_numbers,
            masses=traj.masses,
            n_frames=len(selected_indices),
            positions=traj.positions[selected_indices],
            velocities=traj.velocities[selected_indices],
            time=traj.time[selected_indices],
            temperature=traj.temperature[selected_indices],
            E_kin=traj.E_kin[selected_indices],
            E_trans_drift=traj.E_trans_drift[selected_indices],
            cell_vectors=selected_cell_vectors,
            E_pot=(
                traj.E_pot[selected_indices] 
                if traj.E_pot is not None else None
            ),
            forces=(
                traj.forces[selected_indices] 
                if traj.forces is not None else None
            ),
            feature_vectors=traj.feature_vectors[selected_indices],
            feature_vectors_type=traj.feature_vectors_type,
            ground_truth=(
                traj.ground_truth.select_frames(selected_indices)
                if traj.ground_truth is not None else None
            ),
            pressure=(
                traj.pressure[selected_indices] 
                if traj.pressure is not None else None
            ),
            volume=(
                traj.volume[selected_indices] 
                if traj.volume is not None else None
            ),
            E_rot_drift=(
                traj.E_rot_drift[selected_indices] 
                if traj.E_rot_drift is not None else None
            ),
        )

def _resolve_chunk_indices(
    n_frames: int,
    chunk: Tuple[int, int] | None,
    selected_frames: npt.NDArray[np.int64] | None
) -> npt.NDArray[np.int64]:
    """
    Resolve the subset of frames to process based on chunk distribution.
    
    This function handles the scenario where concurrent processes work on the
    same structure to get energies and forces. It ensures disjoint work distribution
    and handles edge cases where there are more workers than frames.
    
    If chunk is None, returns selected_frames (or all frames if None).
    """
    # 1. Determine pool of frames
    if selected_frames is not None:
        pool = selected_frames
    else:
        pool = np.arange(n_frames)
        
    if chunk is None:
        return pool

    idx, n_chunks = chunk
    
    # Validate input
    if not (1 <= idx <= n_chunks):
        raise ValueError(f"Chunk index {idx} must be between 1 and {n_chunks}")

    # 2. Split into segments
    segments = np.array_split(pool, n_chunks)
    
    # 3. Select segment (0-based index)
    my_segment = segments[idx - 1]
    
    # 4. Logging
    print(f"Worker {idx}/{n_chunks}: processing {len(my_segment)} frames "
          f"(indices: {my_segment if len(my_segment) > 0 else '[]'})")
          
    return my_segment

def _run_model(
        structure: _Structure,
        calculator: CALCULATORS,
        energies: bool = True,
        forces: bool = True,
        feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
        exec_params: Resources | None = None,
        overwrite: bool = False,
        selected_frames: npt.NDArray[np.int64] | None = None,
) -> None:
    """
    Generate energies, forces, and feature vectors for a series of structure
    frames.
    
    Args:
        selected_frames: Optional list of frame indices to process.
            If provided, calculations will be performed only for these frames.
            Must not be used with feature_vectors.
    
    (1) Energies and forces computed with this function
        are stored as ground truth. Ground truth data consists of multiple
        series of energies and forces tagged by the level_of_theory attribute
        extracted from the calculator. Make sure that this attribute is descriptive
        enough to identify the theoretical model.
    (2) If feature vectors were not computed at the structure generation stage,
        they can be generated here to enable subsampling in the feature space.
        
    """
    if not isinstance(calculator, CALCULATORS):
        valid_names = [x.__name__ for x in typing.get_args(CALCULATORS)]
        raise TypeError(
            f"Invalid calculator. Expected one of {valid_names}, "
            f"got {type(calculator).__name__}. \n"
            f"Import a calculator class from mbe_automation.calculators."
        )
    
    if feature_vectors_type not in FEATURE_VECTOR_TYPES:
        raise ValueError("Invalid type of feature vectors")

    if selected_frames is not None:
        if (
                len(selected_frames) > 0 and
                (np.min(selected_frames) < 0 or np.max(selected_frames) > structure.n_frames - 1)
        ):
            raise ValueError("Invalid indices in selected_frames")

        unique_selected_frames = np.unique(selected_frames)
        if len(unique_selected_frames) != len(selected_frames):
            raise ValueError("Repeated elements in selected_frames")
        selected_frames = unique_selected_frames

    if (
            feature_vectors_type != "none" and
            structure.feature_vectors_type != "none" and
            not overwrite
    ):
        raise ValueError("Cannot overwrite existing feature vectors unless overwrite=True.")

    if (
            feature_vectors_type != "none" and
            selected_frames is not None
    ):
        raise ValueError("Feature vectors can be computed only for the full set of frames. "
                         "Do not specify selected_frames.")

    frames_to_compute = _frames_to_compute(
        structure=structure,
        level_of_theory=calculator.level_of_theory,
        overwrite=overwrite,
        selected_frames=selected_frames,
    )

    if (
            feature_vectors_type != "none" and
            len(frames_to_compute) < structure.n_frames
    ):
        raise ValueError(
            "Feature vectors can be computed only for the full set of frames. "
            "If computed alongside energies and forces, set overwrite=True. "
        )

    if len(frames_to_compute) == 0:
        print(f"Found zero frames to process with {calculator.level_of_theory}.")
        return
    
    level_of_theory = calculator.level_of_theory

    if (energies or forces) and structure.ground_truth is None:
        structure.ground_truth = mbe_automation.storage.GroundTruth()

    if exec_params is None:
        exec_params = Resources.auto_detect()

    exec_params.set()
    #
    # To avoid calculation on frames with COMPLETE status,
    # create a new view of the structure with the subset
    # of frames where the data are missing
    #
    calculation_structure = structure.select(frames_to_compute)
    
    E_pot, F, d, statuses = mbe_automation.calculators.run_model(
        structure=calculation_structure,
        calculator=calculator,
        compute_energies=energies,
        compute_forces=forces,
        compute_feature_vectors=(feature_vectors_type!="none"),
        average_over_atoms=(feature_vectors_type=="averaged_environments"),
        resources=exec_params,
    )
    #
    # Broadcast the computed data to arrays
    # of full dimension (structure.n_frames).
    #
    if energies:
        if level_of_theory not in structure.ground_truth.energies:
            structure.ground_truth.energies[level_of_theory] = np.full(
                structure.n_frames, np.nan
            )

        structure.ground_truth.energies[level_of_theory][frames_to_compute] = E_pot

    if forces:
        if level_of_theory not in structure.ground_truth.forces:
            structure.ground_truth.forces[level_of_theory] = np.full(
                (structure.n_frames, structure.n_atoms, 3), np.nan
            )
        structure.ground_truth.forces[level_of_theory][frames_to_compute] = F

    if energies or forces:
        #
        # We store the calculation status only for energies and forces
        # because that's where something can go wrong. For feature vectors,
        # which are computed using MLIPs, we assume that the computation is performed
        # for all frames in the structure and always succeeds.
        #
        if level_of_theory not in structure.ground_truth.calculation_status:
            structure.ground_truth.calculation_status[level_of_theory] = np.full(
                structure.n_frames, CALCULATION_STATUS_UNDEFINED, dtype=np.int64
            )

        structure.ground_truth.calculation_status[level_of_theory][frames_to_compute] = statuses

    if feature_vectors_type != "none" and d is not None:
        assert len(frames_to_compute) == structure.n_frames
        structure.feature_vectors = d
        structure.feature_vectors_type = feature_vectors_type

    return

def _to_mace_dataset(
        dataset: List[Structure|FiniteSubsystem],
        save_path: str,
        level_of_theory: str | dict[Literal["target", "baseline"], str],
        atomic_reference: AtomicReference | None = None,
) -> None:
    """
    Export dataset to training files readable by MACE.

    Args:
        save_path: Path to save the XYZ training file.
        level_of_theory: The level of theory to use for energies and forces.
            Can be a string for direct learning, or a dict with "target" and
            "baseline" keys for delta learning.
        atomic_reference: Ground-state energies for isolated
            atoms, used to generate baseline atomic shifts
            in MACE.
    """
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    mbe_automation.common.display.framed([
        "Exporting dataset to MACE training file"
    ])

    print(f"save_path:        {save_path}")
    print(f"atomic_reference: {'available' if atomic_reference is not None else 'not available'}")
    print(f"level_of_theory:  {level_of_theory}")
    print("")

    if isinstance(level_of_theory, str):
        _statistics(dataset, level_of_theory)
        _print_status_report(dataset, level_of_theory)
    elif isinstance(level_of_theory, dict):
        for key, method in level_of_theory.items():
            print(f"statistics for {key} method ({method}):")
            _statistics(dataset, method)
            _print_status_report(dataset, method)
            print("")

    structures = []
    for x in dataset:
        #
        # Select frames for which the energies and optionally forces
        # are available
        #
        if isinstance(x, FiniteSubsystem):
            y = x.select(level_of_theory=level_of_theory)
            if y is not None:
                structures.append(y.cluster_of_molecules)
        else:
            y = x.select(level_of_theory=level_of_theory)
            if y is not None:
                structures.append(y)

    if len(structures) == 0:
        raise ValueError("No structures with completed calculations found.") 

    mbe_automation.ml.mace.to_xyz_training_set(
        structures=structures,
        level_of_theory=level_of_theory,
        save_path=save_path,
        atomic_reference=atomic_reference,
    )
    
def _statistics(
    systems: List[Structure | FiniteSubsystem],
    level_of_theory: str
) -> None:
    """
    Print mean and standard deviation of energy per atom.
    """

    energies = []
    for i, x in enumerate(systems):
        if isinstance(x, FiniteSubsystem):
            y = x.select(level_of_theory=level_of_theory)
            if y is not None:
                y = y.cluster_of_molecules
        else:
            y = x.select(level_of_theory=level_of_theory)

        if y is not None:
            e_i = y.energies_at_level_of_theory(level_of_theory)
            if e_i is not None:
                 energies.append(e_i)

    if len(energies) == 0:
        print(f"Zero frames with data at level of theory '{level_of_theory}'.")
        return

    data = np.concatenate(energies)
    n_frames = len(data)

    print(f"Statistics computed on {n_frames} frames at level of theory '{level_of_theory}'.")
    print(f"Mean energy: {np.mean(data):.5f} eV/atom")
    print(f"Std energy:  {np.std(data):.5f} eV/atom")

def _print_status_report(
    dataset: List[Structure | FiniteSubsystem],
    level_of_theory: str
) -> None:
    """
    Print a report of calculation statuses for a given level of theory.
    """
    total_frames = 0
    completed = 0
    unconverged = 0
    failed = 0
    
    for x in dataset:
        if isinstance(x, FiniteSubsystem):
             curr_struct = x.cluster_of_molecules
        else:
             curr_struct = x
        
        total_frames += curr_struct.n_frames
        
        if curr_struct.ground_truth is not None and level_of_theory in curr_struct.ground_truth.calculation_status:
            statuses = curr_struct.ground_truth.calculation_status[level_of_theory]
             
            completed += np.sum(statuses == CALCULATION_STATUS_COMPLETED)
            unconverged += np.sum(statuses == CALCULATION_STATUS_SCF_NOT_CONVERGED)
            failed += np.sum(statuses == CALCULATION_STATUS_FAILED)
             
        else:
            #
            # If no ground truth or level of theory not present, all are undefined/missing
            # unless the structure ITSELF is generated at this level of theory and includes
            # energies or forces.
            #
            if (
                    curr_struct.level_of_theory == level_of_theory and
                    (curr_struct.E_pot is not None or curr_struct.forces is not None)
            ):
                completed += curr_struct.n_frames

    print(f"Status report for {level_of_theory}:")
    print(f"  Total frames                   {total_frames}")
    print(f"  Frames with complete data      {completed}")
    print(f"  Unconverged SCF                {unconverged}")
    print(f"  Failed                         {failed}")
    
    if completed > 0:
        print("Only frames with complete data will be exported")
    else:
        print("No frames with complete data found!")

def _energy_fluctuations(
        traj: Trajectory,
        save_path: str | None = None
):
    df = pd.DataFrame({
        "time (fs)": traj.time,
        "T (K)": traj.temperature,
        "E_kin (eV∕atom)": traj.E_kin,
        "E_pot (eV∕atom)": traj.E_pot,
        "p (GPa)": traj.pressure,
        "V (Å³∕atom)": traj.volume
    })
    df.attrs["time_equilibration (fs)"] = traj.time_equilibration
    df.attrs["ensemble"] = traj.ensemble
    df.attrs["target_temperature (K)"] = traj.target_temperature
    df.attrs["target_pressure (GPa)"] = traj.target_pressure
    
    return mbe_automation.dynamics.md.display.energy_fluctuations(df, save_path)

def _thermal_displacements(
        force_constants: ForceConstants,
        temperature_K: float,
        phonon_filter: PhononFilter | None = None,
) -> ThermalDisplacements:
    
    if phonon_filter is None:
        phonon_filter = PhononFilter(
            freq_max_THz=None,
            k_point_mesh=50.0,
        )
        
    return mbe_automation.dynamics.harmonic.modes.thermal_displacements(
        force_constants=force_constants,
        temperatures_K=np.array([temperature_K]),
        phonon_filter=phonon_filter,
        cell_type="primitive"
    )

def _to_cif_file(
    force_constants: ForceConstants,
    save_path: str,
    save_adps: bool = False,
    temperature_K: float | None = None,
    phonon_filter: PhononFilter | None = None,
) -> None:
    disp = None
    if save_adps:
        if temperature_K is None:
            raise ValueError("temperature_K must be provided to save thermal displacements (ADPs).")
        disp = force_constants.thermal_displacements(
            temperature_K=temperature_K,
            phonon_filter=phonon_filter
        )
    
    mbe_automation.storage.to_xyz_file(
        save_path=save_path,
        system=force_constants.primitive,
        thermal_displacements=disp,
        temperature_idx=0 
    )
