from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Literal, Sequence, List
import numpy as np
import numpy.typing as npt
from mace.calculators import MACECalculator
from ase.calculators.calculator import Calculator as ASECalculator
from pymatgen.analysis.local_env import NearNeighbors, CutOffDictNN

import mbe_automation.storage
from mbe_automation.configs.execution import ParallelCPU
from mbe_automation.configs.clusters import FiniteSubsystemFilter
from mbe_automation.configs.structure import Minimum
from mbe_automation.storage import ForceConstants as _ForceConstants
from mbe_automation.storage import Structure as _Structure
from mbe_automation.storage import Trajectory as _Trajectory
from mbe_automation.storage import MolecularCrystal as _MolecularCrystal
from mbe_automation.storage import FiniteSubsystem as _FiniteSubsystem
import mbe_automation.dynamics.harmonic.modes
import mbe_automation.ml.core
import mbe_automation.ml.mace
import mbe_automation.ml.delta
import mbe_automation.calculators
import mbe_automation.structure.clusters
from mbe_automation.ml.core import SUBSAMPLING_ALGOS, FEATURE_VECTOR_TYPES
from mbe_automation.ml.core import REFERENCE_ENERGY_TYPES
from mbe_automation.storage.core import DATA_FOR_TRAINING
from mbe_automation.configs.structure import SYMMETRY_TOLERANCE_STRICT, SYMMETRY_TOLERANCE_LOOSE

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

@dataclass(kw_only=True)
class Structure(_Structure):
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
            indices: npt.NDArray[np.integer]
    ) -> Structure:
        return Structure(**vars(
            _select_frames(self, indices)
        ))

    def random_split(
            self,
            fractions: Sequence[float],
            rng: np.random.Generator | None = None,
    ) -> Sequence[Structure]:

        return [
            Structure(**vars(x)) for x in _split_frames(self, fractions, rng)
        ]

    def to_training_set(
            self,
            save_path: str,
            quantities: List[Literal["energies", "forces"]],
            append: bool = False,
            data_format: Literal["mace_xyz"] = "mace_xyz",
    ) -> None:
        if data_format == "mace_xyz":
            mbe_automation.ml.mace.to_xyz_training_set(
                structure=self,
                save_path=save_path,
                append=append,
                E_pot=(self.E_pot if "energies" in quantities else None),
                forces=(self.forces if "forces" in quantities else None),
            )
        else:
            raise ValueError("Unsupported data format of the training set")

    def run_model(
            self,
            calculator: ASECalculator | MACECalculator,
            energies: bool = True,
            forces: bool = True,
            feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
            delta_learning: Literal["target", "baseline", "none"]="none",
            exec_params: ParallelCPU | None = None,
    ) -> None:
        _run_model(
            structure=self,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            delta_learning=delta_learning,
            exec_params=exec_params,
        )

    def detect_molecules(
            self,
    ) -> MolecularCrystal:

        return MolecularCrystal(**vars(
            mbe_automation.structure.clusters.detect_molecules(system=self)
        ))

    def extract_all_molecules(
            self,
            bonding_algo: NearNeighbors=CutOffDictNN.from_preset("vesta_2019"),
            reference_frame_index: int = 0,
            calculator: ASECalculator | None = None,
    ) -> List[Structure]:

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
            bonding_algo: NearNeighbors=CutOffDictNN.from_preset("vesta_2019"),
            reference_frame_index: int = 0,
    ) -> List[Structure]:

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
            bonding_algo: NearNeighbors=CutOffDictNN.from_preset("vesta_2019"),
            reference_frame_index: int = 0,
            work_dir: Path | str = Path("./")
    ) -> None:
        
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
class Trajectory(_Trajectory):
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

    def run_model(
            self,
            calculator: ASECalculator | MACECalculator,
            energies: bool = True,
            forces: bool = True,
            feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
            delta_learning: Literal["none", "target", "baseline"]="none",
            exec_params: ParallelCPU | None = None,
    ) -> None:
        _run_model(
            structure=self,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            delta_learning=delta_learning,
            exec_params=exec_params,
        )

@dataclass(kw_only=True)
class MolecularCrystal(_MolecularCrystal):
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

    def extract_finite_subsystem(
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

@dataclass(kw_only=True)
class FiniteSubsystem(_FiniteSubsystem):
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
            only: List[Literal[*DATA_FOR_TRAINING]] | None = None,            
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

    def run_model(
            self,
            calculator: ASECalculator | MACECalculator,
            energies: bool = True,
            forces: bool = True,
            feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
            delta_learning: Literal["none", "target", "baseline"]="none",
            exec_params: ParallelCPU | None = None,
    ) -> None:
        _run_model(
            structure=self.cluster_of_molecules,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            delta_learning=delta_learning,
            exec_params=exec_params,
        )

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

    def to_training_set(
            self,
            save_path: str,
            quantities: List[Literal["energies", "forces"]],
            append: bool = False,
            data_format: Literal["mace_xyz"] = "mace_xyz",
    ) -> None:
        if data_format == "mace_xyz":
            mbe_automation.ml.mace.to_xyz_training_set(
                structure=self.cluster_of_molecules,
                save_path=save_path,
                append=append,
                E_pot=(self.cluster_of_molecules.E_pot if "energies" in quantities else None),
                forces=(self.cluster_of_molecules.forces if "forces" in quantities else None),
            )
        else:
            raise ValueError("Unsupported data format of the training set")

@dataclass
class Dataset:
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

    def export_to_mace(
            self,
            save_path: str,
            learning_strategy: Literal["direct", "delta"],
            reference_energy_type: Literal[*REFERENCE_ENERGY_TYPES] = "none",
            reference_molecule: Structure | None = None,
            reference_frame_index: int = 0,
    ) -> None:
        """
        Export dataset to training files readable by MACE.
        """
        _export_to_mace(
            dataset=self.structures,
            save_path=save_path,
            learning_strategy=learning_strategy,
            reference_energy_type=reference_energy_type,
            reference_molecule=reference_molecule,
            reference_frame_index=reference_frame_index,
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
            delta=(
                struct.delta.select_frames(indices)
                if struct.delta is not None else None
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

def _run_model(
        structure: _Structure,
        calculator: ASECalculator | MACECalculator,
        energies: bool = True,
        forces: bool = True,
        feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
        delta_learning: Literal["target", "baseline", "none"]="none",
        exec_params: ParallelCPU | None = None,
) -> None:
    assert feature_vectors_type in FEATURE_VECTOR_TYPES
    
    if exec_params is None:
        exec_params = ParallelCPU.recommended()

    exec_params.set()

    E_pot, F, d = mbe_automation.calculators.run_model(
        structure=structure,
        calculator=calculator,
        compute_energies=energies,
        compute_forces=forces,
        compute_feature_vectors=(feature_vectors_type!="none"),
        average_over_atoms=(feature_vectors_type=="averaged_environments"),
        return_arrays=True,
    )

    if feature_vectors_type != "none" and d is not None:
        structure.feature_vectors = d
        structure.feature_vectors_type = feature_vectors_type

    if delta_learning == "none":
        if energies: structure.E_pot = E_pot
        if forces: structure.forces = F

    if delta_learning != "none" and structure.delta is None:
        structure.delta = mbe_automation.storage.core.DeltaTargetBaseline()

    if delta_learning == "baseline":
        if energies: structure.delta.E_pot_baseline = E_pot
        if forces: structure.delta.forces_baseline = F

    if delta_learning == "target":
        if energies: structure.delta.E_pot_target = E_pot
        if forces: structure.delta.forces_target = F

    if delta_learning == "baseline" and energies:
        unique_elements = structure.unique_elements
        E_atomic_baseline = mbe_automation.calculators.atomic_energies(
            calculator=calculator,
            z_numbers=unique_elements,
        )
        structure.delta.E_atomic_baseline = E_atomic_baseline

    return

def _export_to_mace(
        dataset: List[Structure|FiniteSubsystem],            
        save_path: str,
        learning_strategy: Literal["direct", "delta"] = "direct",
        reference_energy_type: Literal[*REFERENCE_ENERGY_TYPES]="none",
        reference_molecule: Structure | None = None,
        reference_frame_index: int = 0,
) -> None:

    structures = []
    for x in dataset:
        if isinstance(x, FiniteSubsystem):
            structures.append(x.cluster_of_molecules)
        else:
            structures.append(x)
            
    if learning_strategy == "direct":
        for i, x in enumerate(structures):
            mbe_automation.ml.mace.to_xyz_training_set(
                structure=x,
                save_path=save_path,
                append=(i>0),
                E_pot=(x.E_pot if x.E_pot is not None else None),
                forces=(x.forces if x.forces is not None else None),
            )

    elif learning_strategy == "delta":
        mbe_automation.ml.delta.export_to_mace(
            structures=structures,
            save_path=save_path,
            reference_energy_type=reference_energy_type,
            reference_molecule=reference_molecule,
            reference_frame_index=reference_frame_index,
        )

    return
