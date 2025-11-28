from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Literal, Sequence
import numpy as np
import numpy.typing as npt

import mbe_automation.storage
from mbe_automation.storage import ForceConstants as _ForceConstants
from mbe_automation.storage import Structure as _Structure
from mbe_automation.storage import Trajectory as _Trajectory
from mbe_automation.storage import MolecularCrystal as _MolecularCrystal
from mbe_automation.storage import FiniteSubsystem as _FiniteSubsystem
import mbe_automation.dynamics.harmonic.modes
import mbe_automation.ml.core
import mbe_automation.ml.mace
from mbe_automation.ml.core import SUBSAMPLING_ALGOS, FEATURE_VECTOR_TYPES

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
        
    def subsample(
            self,
            n: int,
            algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling"
    ) -> Structure:
        return Structure(**vars(
            _subsample_structure(self, n, algorithm)
        ))

    def select(
            self,
            indices: npt.NDArray[np.integer]
    ) -> Structure:
        return Structure(**vars(
            _select_frames(self, indices)
        ))

    def split(
            self,
            fractions: Sequence[float],
            rng: np.random.Generator = lambda: np.random.default_rng(seed=42)
    ) -> Sequence[Structure, ...]:

        return [
            Structure(**vars(x)) for x in _split_frames(self, fractions, rng)
        ]

    def to_training_set(
            self,
            save_path: str,
            quantities: List[Literal["energies", "forces"]],
            append: bool = False,
            data_format: Literal["mace_xyz"] = "mace_xyz",
    ):
        if data_format == "mace_xyz":
            mbe_automation.ml.mace.to_xyz_training_set(
                structure=self,
                save_path=save_path,
                append=append,
                quantities=quantities,
            )
        else:
            raise ValueError("Unsupported data format of the training set")

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
            algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling"
    ):
        return Trajectory(**vars(
            _subsample_trajectory(self, n, algorithm)
        ))

@dataclass(kw_only=True)
class MolecularCrystal(_MolecularCrystal):
    def subsample(
            self,
            n: int,
            algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling"
    ) -> MolecularCrystal:
        return MolecularCrystal(
            supercell=_subsample_structure(self.supercell, n, algorithm),
            index_map=self.index_map,
            centers_of_mass=self.centers_of_mass,
            identical_composition=self.identical_composition,
            n_molecules=self.n_molecules,
            central_molecule_index=self.central_molecule_index,
            min_distances_to_central_molecule=self.min_distances_to_central_molecule,
            max_distances_to_central_molecule=self.max_distances_to_central_molecule
        )

@dataclass(kw_only=True)
class FiniteSubsystem(_FiniteSubsystem):
    def subsample(
            self,
            n: int,
            algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling"
    ) -> FiniteSubsystem:
        return FiniteSubsystem(
            cluster_of_molecules=_subsample_structure(self.cluster_of_molecules, n, algorithm),
            molecule_indices=self.molecule_indices,
            n_molecules=self.n_molecules
        )

def _select_frames(
        struct: _Structure,
        indices: npt.NDArray[np.integer]
    ) -> _Structure:
        """
        Return a new Structure containing only the specified frames.
        """

        selected_cell_vectors = struct.cell_vectors
        if struct.cell_vectors is not None:
            if struct.cell_vectors.ndim == 3:
                selected_cell_vectors = struct.cell_vectors[indices]
            else:
                selected_cell_vectors = struct.cell_vectors

        return _Structure(
            positions=struct.positions[indices],
            atomic_numbers=struct.atomic_numbers,
            masses=struct.masses,
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
            feature_vectors_type=struct.feature_vectors_type
        )

def _subsample_structure(
        struct: _Structure,
        n: int,
        algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling"
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
        )
        
        return _select_frames(struct, selected_indices)

def _split_frames(
        struct: _Structure,
        fractions: Sequence[float],
        rng: np.random.Generator = lambda: np.random.default_rng(seed=42)
) -> Sequence[_Structure, ...]:

    if not np.isclose(sum(fractions), 1.0):
         raise ValueError("Fractions must sum to 1.0")

    n_total = struct.n_frames
    indices = np.arange(n_total)
    rng.shuffle(indices)

    lengths = [int(f * n_total) for f in fractions]
    lengths[-1] = n_total - sum(lengths[:-1])

    structures = []
    start = 0
    for length in lengths:
        end = start + length
        selected_indices = indices[start:end]
        structures.append(_select_frames(struct, selected_indices))
        start = end

    return structures
    
def _subsample_trajectory(
        traj: _Trajectory,
        n: int,
        algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling"
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
