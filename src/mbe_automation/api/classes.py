from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Literal, Sequence, List
from pathlib import Path
import numpy as np
import numpy.typing as npt
from mace.calculators import MACECalculator
import ase
from ase.calculators.calculator import Calculator as ASECalculator
from pymatgen.analysis.local_env import NearNeighbors, CutOffDictNN

import mbe_automation.storage
import mbe_automation.common
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
import mbe_automation.calculators
import mbe_automation.structure.clusters
from mbe_automation.ml.core import SUBSAMPLING_ALGOS, FEATURE_VECTOR_TYPES, LEVELS_OF_THEORY
from mbe_automation.ml.core import REFERENCE_ENERGY_TYPES
from mbe_automation.storage.core import DATA_FOR_TRAINING
from mbe_automation.configs.structure import SYMMETRY_TOLERANCE_STRICT, SYMMETRY_TOLERANCE_LOOSE

class _AtomicEnergiesCalc:
    def atomic_energies(self, calculator) -> dict[np.int64, np.float64]:
        """
        Calculate ground-state energies for all unique isolated atoms
        represented in the structure. Spin is selected automatically
        based on the ground-state configurations of isolated atoms
        defined in pyscf.data.elements.CONFIGURATION.

        This is the function that you need to generate isolated
        atomic baseline data for machine learning interatomic
        potentials.

        Remember to define the calculator with exactly the same
        settings (basis set, integral approximations, thresholds)
        as for the main dataset calculation.
        """
        return mbe_automation.calculators.atomic_energies(
            calculator=calculator,
            z_numbers=self.unique_elements,
        )

class _TrainingStructure:
    def to_mace_dataset(
            self,
            save_path: str,
            level_of_theory: str | dict[Literal["target", "baseline"], str],
            atomic_energies: dict[np.int64, np.float64] | dict[str, dict[np.int64, np.float64]] | None = None,
    ) -> None:
        _to_mace_dataset(
            dataset=[self],
            save_path=save_path,
            level_of_theory=level_of_theory,
            atomic_energies=atomic_energies,
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

    def to_ase_atoms(
            self,
            frame_index: int = 0,
    ) -> ase.Atoms:
        
        return mbe_automation.storage.to_ase(
            structure=self,
            frame_index=frame_index
        )

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

    def run_model(
            self,
            calculator: ASECalculator | MACECalculator,
            energies: bool = True,
            forces: bool = True,
            feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
            level_of_theory: str="default",
            exec_params: ParallelCPU | None = None,
    ) -> None:
        _run_model(
            structure=self,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            level_of_theory=level_of_theory,
            exec_params=exec_params,
        )

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
class Trajectory(_Trajectory, _AtomicEnergiesCalc, _TrainingStructure):
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
            level_of_theory: str="default",
            exec_params: ParallelCPU | None = None,
    ) -> None:
        _run_model(
            structure=self,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            level_of_theory=level_of_theory,
            exec_params=exec_params,
        )

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
            level_of_theory: str="default",
            exec_params: ParallelCPU | None = None,
    ) -> None:
        _run_model(
            structure=self.cluster_of_molecules,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            level_of_theory=level_of_theory,
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
            atomic_energies: dict[np.int64, np.float64] | None = None,
    ) -> None:
        _to_mace_dataset(
            dataset=self.structures,
            save_path=save_path,
            level_of_theory=level_of_theory,
            atomic_energies=atomic_energies,
        )

    @property
    def unique_elements(self) -> npt.NDArray[np.int64]:
        """
        Return a sorted NumPy array of unique Z numbers for a list of structures.
        """
        unique_elements = [structure.unique_elements for structure in self.structures]
        return np.unique(np.concatenate(unique_elements))

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
        level_of_theory: str = "default",
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

    if level_of_theory == "default":
        if energies: structure.E_pot = E_pot
        if forces: structure.forces = F

    if level_of_theory != "default" and structure.ground_truth is None:
        structure.ground_truth = mbe_automation.storage.GroundTruth()

    if level_of_theory != "default":
        if energies: structure.ground_truth.energies[level_of_theory] = E_pot
        if forces: structure.ground_truth.forces[level_of_theory] = F

    return

def _to_mace_dataset(
        dataset: List[Structure|FiniteSubsystem],
        save_path: str,
        level_of_theory: str | dict[Literal["target", "baseline"], str],
        atomic_energies: dict[np.int64, np.float64] | None = None,
) -> None:

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    structures = []
    for x in dataset:
        if isinstance(x, FiniteSubsystem):
            structures.append(x.cluster_of_molecules)
        else:
            structures.append(x)

    mbe_automation.ml.mace.to_xyz_training_set(
        structures=structures,
        level_of_theory=level_of_theory,
        save_path=save_path,
        atomic_energies=atomic_energies,
    )
    
def _statistics(
        systems: List[Structure | FiniteSubsystem],
        level_of_theory: str
) -> None:
        """
        Print mean and standard deviation of energy per atom.
        """

        mbe_automation.common.display.framed([
            "Dataset statistics"
        ])
        
        energies = []
        for i, x in enumerate(systems):
            struct = x.cluster_of_molecules if isinstance(x, FiniteSubsystem) else x

            if struct.ground_truth is None:
                raise ValueError(f"Ground truth data missing in structure {i}.")
            
            E_i = struct.ground_truth.energies.get(level_of_theory)

            if E_i is None:
                raise ValueError(f"Missing energies in structure {i}.")

            energies.append(E_i)

        data = np.concatenate(energies)
        
        print(f"Mean energy: {np.mean(data):.5f} eV/atom")
        print(f"Std energy:  {np.std(data):.5f} eV/atom")
