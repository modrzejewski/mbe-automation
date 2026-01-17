from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Literal, overload, Dict
import pymatgen
import pandas as pd
import phonopy
from phonopy import Phonopy
import h5py
import numpy as np
import numpy.typing as npt
import os
from .file_lock import dataset_file

DATA_FOR_TRAINING = [
    "feature_vectors",
    "ground_truth",
]
#
# Character used as a replacement for ordinary slash "/",
# which is a reserved character for HDF5 tree structure.
# By using the unicode character, we avoid an unintentional
# creation of HDF5 subgroups.
#
UNICODE_DIVISION_SLASH = "∕"

CALCULATION_STATUS_COMPLETED = 0
CALCULATION_STATUS_UNDEFINED = 1
CALCULATION_STATUS_SCF_NOT_CONVERGED = 2
CALCULATION_STATUS_FAILED = 3

@dataclass
class BrillouinZonePath:
    kpoints: List[npt.NDArray[np.floating]]
    frequencies: List[npt.NDArray[np.floating]]
    path_connections: npt.NDArray[np.bool_]
    labels: npt.NDArray[np.str_]
    distances: List[npt.NDArray[np.floating]]

@dataclass
class EOSCurves:
    temperatures: npt.NDArray[np.floating]
    V_sampled: npt.NDArray[np.floating]
    G_sampled: npt.NDArray[np.floating]
    V_interp: npt.NDArray[np.floating]
    G_interp: npt.NDArray[np.floating]
    V_min: npt.NDArray[np.floating]
    G_min: npt.NDArray[np.floating]

@dataclass(kw_only=True)
class AtomicReference:
    """
    Isolated atom energies required to generate reference energy
    for machine-learning interatomic potentials.


    In MLIPs like MACE, the energy predicted by the neural
    net is computed w.r.t. the trivial reference defined
    as the sum of isolated ground-state atom energies.
    Thanks to this, the training process focuses on learning
    interactions, not the absolute energies which include
    the inert core.
    
    """
    energies: dict[str, dict[np.int64, np.float64]] = field(default_factory=dict)
    
    def __getitem__(self, level_of_theory: str) -> dict[np.int64, np.float64]:
        return self.energies[level_of_theory]

    def __setitem__(self, level_of_theory: str, atom_energies: dict[np.int64, np.float64]) -> None:
        self.energies[level_of_theory] = atom_energies

    def __contains__(self, level_of_theory: str) -> bool:
        """Check availability of a specific level of theory."""
        return level_of_theory in self.energies

    def __add__(self, other: AtomicReference) -> AtomicReference:
        if not isinstance(other, AtomicReference):
            return NotImplemented

        merged_energies = {k: v.copy() for k, v in self.energies.items()}
        
        for level_of_theory, atom_energies in other.energies.items():
            if level_of_theory in merged_energies:
                merged_energies[level_of_theory].update(atom_energies)
            else:
                merged_energies[level_of_theory] = atom_energies.copy()

        return AtomicReference(energies=merged_energies)
    
    @property
    def levels_of_theory(self) -> list[str]:
        return list(self.energies.keys())
    
@dataclass
class GroundTruth:
    energies: Dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    forces: Dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    calculation_status: Dict[str, npt.NDArray[np.int64]] = field(default_factory=dict)

    def copy(self) -> GroundTruth:
        return GroundTruth(
            energies={k: v.copy() for k, v in self.energies.items()},
            forces={k: v.copy() for k, v in self.forces.items()},
            calculation_status={k: v.copy() for k, v in self.calculation_status.items()},
        )

    def select_frames(self, indices: npt.NDArray[np.integer]) -> GroundTruth:
        return GroundTruth(
            energies={k: v[indices] for k, v in self.energies.items()},
            forces={k: v[indices] for k, v in self.forces.items()},
            calculation_status={k: v[indices] for k, v in self.calculation_status.items()},
        )
        
@dataclass
class Structure:
    """
    Main data storage class for geometric data and corresponding
    energies and forces.

    Data generated with the theoretical model applied
    at the structure generation stage:
    
    Structure.positions
    Structure.E_pot
    Structure.forces

    Data generated for fixed, precomputed structures
    via a call to Structure.run:

    Structure.ground_truth

    It is expected that ground truth object is populated by
    data points from expensive models which cannot be used
    for structure generation.
    """
    positions: npt.NDArray[np.floating]
    atomic_numbers: npt.NDArray[np.integer]
    masses: npt.NDArray[np.floating]
    cell_vectors: npt.NDArray[np.floating] | None
    n_frames: int
    n_atoms: int
    periodic: bool = False
    E_pot: npt.NDArray[np.floating] | None = None
    forces: npt.NDArray[np.floating] | None = None
    ground_truth: GroundTruth | None = None
    feature_vectors: npt.NDArray[np.floating] | None = None
    feature_vectors_type: str = "none"
    level_of_theory: str | None = None
    
    def __post_init__(self):
        self.periodic = (self.cell_vectors is not None)
        
    def copy(self) -> Structure:
        return Structure(
            positions=self.positions.copy(),
            atomic_numbers=self.atomic_numbers.copy(),
            masses=self.masses.copy(),
            cell_vectors=(self.cell_vectors.copy() if self.cell_vectors is not None else None),
            n_frames=self.n_frames,
            n_atoms=self.n_atoms,
            periodic=self.periodic,
            E_pot=(self.E_pot.copy() if self.E_pot is not None else None),
            forces=(self.forces.copy() if self.forces is not None else None),
            ground_truth=(self.ground_truth.copy() if self.ground_truth is not None else None),
            feature_vectors=(
                self.feature_vectors.copy() 
                if self.feature_vectors_type != "none" else None
            ),
            feature_vectors_type=self.feature_vectors_type,
            level_of_theory=self.level_of_theory,
        )
    
    def save(
            self,
            dataset: str,
            key: str,
            only: List[Literal[*DATA_FOR_TRAINING]] | Literal[*DATA_FOR_TRAINING] | None = None,
            update_mode: Literal["update_properties", "replace"] = "update_properties",
    ) -> None:
        """
        Save the structure to a dataset.

        Parameters
        ----------
        dataset : str
            Path to the HDF5 dataset file.
        key : str
            Key within the HDF5 file.
        update_mode : Literal["update_ground_truth", "replace"]
            Mode for updating existing data. Defaults to "update_ground_truth".

        Notes
        -----
        When ``update_mode="update_ground_truth"`` (default), if the HDF5 key already exists,
        basic structural data (positions, atomic_numbers, cell_vectors) is NOT saved.
        Only energies, forces, and ground truth data are updated.

        .. warning::
           If you modify the geometry of a Structure in memory and call save() with
           ``update_mode="update_ground_truth"`` on an existing key, the file will contain
           the OLD geometry but NEW energies/forces. This corrupts the dataset integrity.
           Use ``update_mode="replace"`` if the geometry has changed.
        """

        if isinstance(only, str):
            only = [only]

        if only is None:
            save_structure(
                dataset=dataset,
                key=key,
                structure=self,
                update_mode=update_mode
            )
        else:
            _save_only(
                dataset=dataset,
                key=key,
                structure=self,
                quantities=only,
            )

    def energies_at_level_of_theory(
            self,
            level_of_theory: str
    ) -> npt.NDArray[np.float64] | None:
        return _energies_at_level_of_theory(self, level_of_theory)

    def forces_at_level_of_theory(
            self,
            level_of_theory: str
    ) -> npt.NDArray[np.float64] | None:
        return _forces_at_level_of_theory(self, level_of_theory)

    def available_energies(
            self,
            restrict_to: list[Literal["ground_truth", "structure_generation"]] | None = None
    ) -> list[str]:
        return _available_energies(self, restrict_to)

    def available_forces(
            self,
            restrict_to: list[Literal["ground_truth", "structure_generation"]] | None = None
    ) -> list[str]:
        return _available_forces(self, restrict_to)

    def to_ase_atoms(self, frame_index: int = 0) -> ase.Atoms:
        from .views import to_ase
        return to_ase(structure=self, frame_index=frame_index)

    def to_pymatgen(self, frame_index: int = 0) -> pymatgen.core.Structure | pymatgen.core.Molecule:
        from .views import to_pymatgen
        return to_pymatgen(structure=self, frame_index=frame_index)
    
    def lattice(self, frame_index: int = 0) -> pymatgen.core.Lattice:
        assert self.periodic, "Structure must be periodic."
        if self.variable_cell:
            cell = self.cell_vectors[frame_index]
        else:
            cell = self.cell_vectors
        return pymatgen.core.Lattice(cell)

    @property
    def unique_elements(self) -> npt.NDArray[np.int64]:
        return np.unique(np.atleast_2d(self.atomic_numbers)[0])

    @property
    def permuted_between_frames(self) -> bool:
        return (self.atomic_numbers.ndim == 2)

    @property
    def variable_cell(self) -> bool:
        return (self.periodic and self.cell_vectors.ndim == 3)

@dataclass
class ForceConstants:
    """Harmonic force constants model."""
    primitive: Structure
    supercell: Structure
    force_constants: npt.NDArray[np.floating]
    supercell_matrix: npt.NDArray[np.integer]

@dataclass(kw_only=True)
class Trajectory(Structure):
    time_equilibration: float
    target_temperature: float
    target_pressure: float | None
    ensemble: Literal["NPT", "NVT", "NVE"]
    time: npt.NDArray[np.floating]
    temperature: npt.NDArray[np.floating]
    pressure: npt.NDArray[np.floating] | None
    volume: npt.NDArray[np.floating] | None
    velocities: npt.NDArray[np.floating]
    E_kin: npt.NDArray[np.floating]
    E_trans_drift: npt.NDArray[np.floating]
    E_rot_drift: npt.NDArray[np.floating] | None
    n_removed_trans_dof: int
    n_removed_rot_dof: int

    @classmethod
    def empty(
            cls,
            n_atoms: int,
            n_frames: int,
            ensemble: Literal["NPT", "NVT", "NVE"],
            periodic: bool,
            time_equilibration: float,
            target_temperature: float,
            target_pressure: float | None = None,
            n_removed_trans_dof: int = 0,
            n_removed_rot_dof: int = 0,
            level_of_theory: str | None = None,
    ):
        if ensemble == "NPT" and target_pressure is None:
            raise ValueError("Target pressure must be specified for the NPT ensemble")
        
        return cls(
            time_equilibration=time_equilibration,
            ensemble=ensemble,
            n_atoms=n_atoms,
            n_frames=n_frames,
            periodic=periodic,
            positions=np.zeros((n_frames, n_atoms, 3)),
            forces=np.zeros((n_frames, n_atoms, 3)),
            velocities=np.zeros((n_frames, n_atoms, 3)),
            atomic_numbers=np.zeros(n_atoms, dtype=int),
            masses=np.zeros(n_atoms),
            cell_vectors=(np.zeros((n_frames, 3, 3)) if periodic else None),
            time=np.zeros(n_frames),
            temperature=np.zeros(n_frames),
            pressure=(np.zeros(n_frames) if ensemble=="NPT" else None),
            volume=(np.zeros(n_frames) if ensemble=="NPT" else None),
            E_kin=np.zeros(n_frames),
            E_pot=np.zeros(n_frames),
            E_trans_drift=np.zeros(n_frames),
            E_rot_drift=(np.zeros(n_frames) if not periodic else None),
            target_temperature=target_temperature,
            target_pressure=target_pressure,
            n_removed_trans_dof=n_removed_trans_dof,
            n_removed_rot_dof=n_removed_rot_dof,
            level_of_theory=level_of_theory,
        )

    def save(
            self,
            dataset: str,
            key: str,
            only: List[Literal[*DATA_FOR_TRAINING]] | Literal[*DATA_FOR_TRAINING] | None = None,
            update_mode: Literal["update_ground_truth", "replace"] = "update_ground_truth",
    ) -> None:
        """
        Save the trajectory to a dataset.

        Parameters
        ----------
        dataset : str
            Path to the HDF5 dataset file.
        key : str
            Key within the HDF5 file.
        update_mode : Literal["update_ground_truth", "replace"]
            Mode for updating existing data. Defaults to "update_ground_truth".

        Notes
        -----
        When ``update_mode="update_ground_truth"`` (default), if the HDF5 key already exists,
        basic structural data (positions, atomic_numbers, cell_vectors) is NOT saved.
        Only energies, forces, and ground truth data are updated.

        .. warning::
           If you modify the geometry of a Trajectory in memory and call save() with
           ``update_mode="update_ground_truth"`` on an existing key, the file will contain
           the OLD geometry but NEW energies/forces. This corrupts the dataset integrity.
           Use ``update_mode="replace"`` if the geometry has changed.
        """

        if isinstance(only, str):
            only = [only]

        if only is None:
            save_trajectory(
                dataset=dataset,
                key=key,
                traj=self,
                update_mode=update_mode,
            )
            
        else:
            _save_only(
                dataset=dataset,
                key=key,
                structure=self,
                quantities=only,
            )            

@dataclass
class MolecularCrystal:
    supercell: Structure
    index_map: List[npt.NDArray[np.integer]] | npt.NDArray[np.integer]
    #
    # COM locations for molecules *in the reference frame*
    # which was used in a call to structure.clusters.detect_molecules.
    # Note that the reference frame may no longer be present as
    # one of the frames in a MoleculeCrystal object returned
    # by MolecularCrystal.subsample.
    #
    centers_of_mass: npt.NDArray[np.floating] 
    identical_composition: bool
    n_molecules: int
    central_molecule_index: int
    min_distances_to_central_molecule: npt.NDArray[np.floating]
    max_distances_to_central_molecule: npt.NDArray[np.floating]

    def atomic_numbers(
            self,
            molecule_indices: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.integer]:
        
        atom_indices = np.concatenate([self.index_map[i] for i in molecule_indices])
        return self.supercell.atomic_numbers[atom_indices]

    @property
    def unique_elements(self) -> npt.NDArray[np.int64]:
        return self.supercell.unique_elements

    def positions(
            self,
            molecule_indices: npt.NDArray[np.integer],
            frame_index: int = 0
    ) -> npt.NDArray[np.floating]:
        
        atom_indices = np.concatenate([self.index_map[i] for i in molecule_indices])
        if self.supercell.positions.ndim == 3:
            selected_positions = self.supercell.positions[frame_index, atom_indices, :]
        else:
            selected_positions = self.supercell.positions[atom_indices, :]
            
        return selected_positions

@dataclass(kw_only=True)
class UniqueClusters:
    """
    Symmetry-unique molecular clusters within a MolecularCrystal.
    """
    n_clusters: int
    molecule_indices: npt.NDArray[np.integer] # Shape (n_unique_clusters, n_cluster_size)
    weights: npt.NDArray[np.integer]
    min_distances: npt.NDArray[np.floating]  # Shape: (n_unique_clusters, n_pairs)
    max_distances: npt.NDArray[np.floating]  # Shape: (n_unique_clusters, n_pairs)

@dataclass
class FiniteSubsystem:
    cluster_of_molecules: Structure
    molecule_indices: npt.NDArray[np.integer]
    n_molecules: int

    @property
    def unique_elements(self) -> npt.NDArray[np.int64]:
        return self.cluster_of_molecules.unique_elements
    
def save_data_frame(
        dataset: str,
        key: str,
        df: pd.DataFrame
):
    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with dataset_file(dataset, "a") as f:
        if key in f:
            del f[key]
        
        group = f.create_group(key)

        if df.attrs:
            for attr_key, attr_value in df.attrs.items():
                group.attrs[attr_key] = attr_value
        
        for column_label in df.columns:
            column = df[column_label]

            if pd.api.types.is_string_dtype(column.dtype):
                encoded_data = column.astype(str).str.encode("utf-8").to_numpy()
                group.create_dataset(
                    name=column_label,
                    data=encoded_data
                )
            else:
                group.create_dataset(
                    name=column_label,
                    data=column.to_numpy()
                )


def read_data_frame(
        dataset: str,
        key: str,
        columns: List[str] | Literal["all"] = "all"
) -> pd.DataFrame:
    
    with dataset_file(dataset, "r") as f:
        group = f[key]
        
        if columns == "all":
            candidate_labels = list(group.keys())
        else:
            candidate_labels = columns
            
        data = {
            label: group[label][:] 
            for label in candidate_labels 
            if (label in group and 
                isinstance(group[label], h5py.Dataset) and 
                group[label].ndim == 1)
        }
        metadata = dict(group.attrs)
        
    df = pd.DataFrame(data)
    df.attrs = metadata
    return df


def save_brillouin_zone_path(
        phonons: Phonopy,
        dataset: str,
        key: str,
):

    band_structure = phonons.band_structure    
    n_segments = len(band_structure.frequencies)
    _, n_bands = band_structure.frequencies[0].shape
    
    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with dataset_file(dataset, "a") as f:
        if key in f:
            del f[key]
            
        group = f.create_group(key)
        group.attrs["dataclass"] = "BrillouinZonePath"
        group.attrs["n_segments"] = n_segments
        group.attrs["n_bands"] = n_bands

        group.create_dataset(
            name="path_connections",
            data=np.array(band_structure.path_connections)
        )
        group.create_dataset(
            name="labels",
            data=np.array([s.encode("utf-8") for s in band_structure.labels]),
            dtype=h5py.string_dtype(encoding="utf-8")
        )
        for i in range(n_segments):
            group.create_dataset(
                name=f"frequencies_segment_{i} (THz)",
                data=band_structure.frequencies[i]                
            )
            group.create_dataset(
                name=f"distances_segment_{i}",
                data=band_structure.distances[i]
            )
            group.create_dataset(
                name=f"kpoints_segment_{i}",
                data=band_structure.qpoints[i]
            )

            
def read_brillouin_zone_path(
        dataset: str,
        key: str
) -> BrillouinZonePath:

    with dataset_file(dataset, "r") as f:
        group = f[key]
        n_segments = group.attrs["n_segments"]
        
        path_connections = group["path_connections"][...]
        labels = group["labels"][...].astype(str)

        frequencies, distances, kpoints, eigenvectors = [], [], [], []
        eigenvecs_available = True
        for i in range(n_segments):
            frequencies.append(group[f"frequencies_segment_{i} (THz)"][...])
            distances.append(group[f"distances_segment_{i}"][...])
            kpoints.append(group[f"kpoints_segment_{i}"][...])

    return BrillouinZonePath(
        kpoints=kpoints,
        frequencies=frequencies,
        path_connections=path_connections,
        labels=labels,
        distances=distances
    )


def save_eos_curves(
        G_tot_curves,
        temperatures,
        dataset,
        key
):
    """
    Save Gibbs free enthalpy vs. volume data for multiple temperatures.
    """

    n_temperatures = len(temperatures)
    n_volumes = len(G_tot_curves[0].V_sampled)
    n_interp = 200

    V_sampled = np.zeros((n_temperatures, n_volumes))
    G_sampled = np.zeros((n_temperatures, n_volumes))
    V_min = np.zeros(n_temperatures) 
    G_min = np.zeros(n_temperatures)
    for i, fit in enumerate(G_tot_curves):
        V_sampled[i, :] = fit.V_sampled[:]
        G_sampled[i, :] = fit.G_sampled[:]
        if fit.min_found:
            V_min[i] = fit.V_min
            G_min[i] = fit.G_min
        else:
            V_min[i] = np.nan
            G_min[i] = np.nan

    G_interp = np.full((n_temperatures, n_interp), np.nan)
    V_interp = np.linspace(np.min(V_sampled), np.max(V_sampled), n_interp)
    for i, fit in enumerate(G_tot_curves):
        if fit.G_interp is not None:
            G_interp[i, :] = fit.G_interp(V_interp)

    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with dataset_file(dataset, "a") as f:
        if key in f:
            del f[key]
            
        group = f.create_group(key)
        group.attrs["dataclass"] = "EOSCurves"
        group.attrs["n_temperatures"] = n_temperatures
        group.attrs["n_volumes"] = n_volumes
        group.attrs["n_interp"] = n_interp

        group.create_dataset(
            name="T (K)",
            data=temperatures
        )
        group.create_dataset(
            name="V_sampled (Å³∕unit cell)",
            data=V_sampled
        )
        group.create_dataset(
            name="G_sampled (kJ∕mol∕unit cell)",
            data=G_sampled
        )
        group.create_dataset(
            name="V_interp (Å³∕unit cell)",
            data=V_interp
        )
        group.create_dataset(
            name="G_interp (kJ∕mol∕unit cell)",
            data=G_interp
        )
        group.create_dataset(
            name="V_min (Å³∕unit cell)",
            data=V_min
        )
        group.create_dataset(
            name="G_min (kJ∕mol∕unit cell)",
            data=G_min
        )


def read_eos_curves(
        dataset,
        key
) -> EOSCurves:

    with dataset_file(dataset, "r") as f:
        group = f[key]
        eos_curves = EOSCurves(
            temperatures=group["T (K)"][...],
            V_sampled=group["V_sampled (Å³∕unit cell)"][...],
            G_sampled=group["G_sampled (kJ∕mol∕unit cell)"][...],
            V_interp=group["V_interp (Å³∕unit cell)"][...],
            G_interp=group["G_interp (kJ∕mol∕unit cell)"][...],
            V_min=group["V_min (Å³∕unit cell)"][...],
            G_min=group["G_min (kJ∕mol∕unit cell)"][...]
        )

    return eos_curves


def _save_structure(
        dataset: str,
        key: str,
        positions: npt.NDArray[np.floating],
        atomic_numbers: npt.NDArray[np.integer],
        masses: npt.NDArray[np.floating],
        cell_vectors: npt.NDArray[np.floating] | None=None,
        E_pot: npt.NDArray[np.floating] | None=None,
        forces: npt.NDArray[np.floating] | None=None,
        feature_vectors: npt.NDArray[np.floating] | None=None,
        feature_vectors_type: str="none",
        ground_truth: GroundTruth | None=None,
        level_of_theory: str | None = None,
        update_mode: Literal["update_properties", "replace"] = "update_properties",
):
    """
    Internal function to save structure components.

    Parameters
    ----------
    update_mode : Literal["update_properties", "replace"]
        Mode for updating existing data. Defaults to "update_properties".

    Notes
    -----
    When ``update_mode="update_properties"`` (default), if the HDF5 key already exists,
    basic structural data (positions, atomic_numbers, cell_vectors) are NOT saved.
    Only energies, forces, feature vectors (if missing), and ground truth data are updated.

    .. warning::
       If you modify the geometry of a Structure in memory and call save() with
       ``update_mode="update_properties"`` on an existing key, the file will contain
       the OLD geometry but NEW energies/forces. This corrupts the dataset integrity.
       Use ``update_mode="replace"`` if the geometry has changed.
    """
    if positions.ndim == 2:
        n_frames = 1
        n_atoms = positions.shape[0]
    elif positions.ndim == 3:
        n_frames = positions.shape[0]
        n_atoms = positions.shape[1]
    else:
        raise ValueError(
            f"positions array must have rank 2 or 3, but has rank {positions.ndim}"
        )

    Path(dataset).parent.mkdir(parents=True, exist_ok=True)    
    with dataset_file(dataset, "a") as f:
        if key in f and update_mode == "replace":
            del f[key]
        
        # Check if we need to initialize the structure data
        # If the key is present, we assume the structure data is already there 
        # (based on the assumption "either all of them are in hdf5, or none of them")
        save_basics = (key not in f)
        
        group = f.require_group(key)

        if save_basics:
            group.create_dataset(
                name="positions (Å)",
                data=positions
            )
            group.create_dataset(
                name="atomic_numbers",
                data=atomic_numbers
            )
            group.create_dataset(
                name="masses (u)",
                data=masses
            )
            is_periodic = (cell_vectors is not None)
            if is_periodic:
                group.create_dataset(
                    name="cell_vectors (Å)",
                    data=cell_vectors
                )
            if feature_vectors_type != "none":
                group.create_dataset(
                    name="feature_vectors",
                    data=feature_vectors
                )
            if E_pot is not None:
                group.create_dataset(
                    name="E_pot (eV∕atom)",
                    data=E_pot
                )
            if forces is not None:
                group.create_dataset(
                    name="forces (eV∕Å)",
                    data=forces
                )
                
            group.attrs["dataclass"] = "Structure"
            group.attrs["n_frames"] = n_frames
            group.attrs["n_atoms"] = n_atoms
            group.attrs["periodic"] = is_periodic
            group.attrs["feature_vectors_type"] = feature_vectors_type
            if level_of_theory is not None:
                group.attrs["level_of_theory"] = level_of_theory

        elif update_mode == "update_properties":
            #
            # If we are in update_properties mode and the structure already exists,
            # we check if we need to add feature vectors which might have been computed
            # after the structure was saved.
            #
            if feature_vectors is not None and feature_vectors_type != "none":
                if "feature_vectors" not in group:
                    group.create_dataset(
                        name="feature_vectors",
                        data=feature_vectors
                    )
                    group.attrs["feature_vectors_type"] = feature_vectors_type

        if ground_truth is not None:
            _save_ground_truth(f, f"{key}/ground_truth", ground_truth, update_mode=update_mode)

@overload
def save_structure(*, dataset: str, key: str, structure: Structure, update_mode: Literal["update_properties", "replace"] = "update_properties") -> None: ...

@overload
def save_structure(
    *,
    dataset: str,
    key: str,
    positions: npt.NDArray[np.floating],
    atomic_numbers: npt.NDArray[np.integer],
    masses: npt.NDArray[np.floating],
    cell_vectors: npt.NDArray[np.floating] | None = None,
    E_pot: npt.NDArray[np.floating] | None = None,
    forces: npt.NDArray[np.floating] | None = None,
    feature_vectors: npt.NDArray[np.floating] | None = None,
    feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
    ground_truth: GroundTruth | None = None,
    level_of_theory: str | None = None,
    update_mode: Literal["update_properties", "replace"] = "update_properties",
) -> None: ...

def save_structure(*, dataset: str, key: str, **kwargs):
    """
    Save a structure to a dataset.

    This function is overloaded and can be called in two ways:

    1. By providing a Structure object as a keyword argument:
       save_structure(dataset="...", key="...", structure=structure)

    2. By providing individual data arrays as keyword arguments:
       save_structure(dataset="...", key="...", positions=..., ...)
    """
    if "structure" in kwargs:
        # --- Signature 1: Called with a Structure object ---
        structure = kwargs["structure"]
        if not isinstance(structure, Structure):
             raise TypeError(
                "Argument 'structure' must be a Structure object, "
                f"but got {type(structure).__name__}"
            )
        _save_structure(
            dataset=dataset,
            key=key,
            positions=structure.positions,
            atomic_numbers=structure.atomic_numbers,
            masses=structure.masses,
            cell_vectors=structure.cell_vectors,
            E_pot=structure.E_pot,
            forces=structure.forces,
            feature_vectors=structure.feature_vectors,
            feature_vectors_type=structure.feature_vectors_type,
            ground_truth=structure.ground_truth,
            level_of_theory=structure.level_of_theory,
            update_mode=kwargs.get("update_mode", "update_properties"),
        )
    elif "positions" in kwargs:
        # --- Signature 2: Called with individual arrays ---
        _save_structure(
            dataset=dataset,
            key=key,
            positions=kwargs.get("positions"),
            atomic_numbers=kwargs.get("atomic_numbers"),
            masses=kwargs.get("masses"),
            cell_vectors=kwargs.get("cell_vectors"),
            E_pot=kwargs.get("E_pot"),
            forces=kwargs.get("forces"),
            feature_vectors=kwargs.get("feature_vectors"),
            ground_truth=kwargs.get("ground_truth"),
            feature_vectors_type=kwargs.get("feature_vectors_type", "none"),
            level_of_theory=kwargs.get("level_of_theory"),
            update_mode=kwargs.get("update_mode", "update_properties"),
        )
    else:
        raise ValueError(
            "Either a 'structure' object or 'positions' and other arrays "
            "must be provided as keyword arguments."
        )
        

def read_structure(dataset, key):
    with dataset_file(dataset, "r") as f:
        group = f[key]
        is_periodic = group.attrs["periodic"]
        feature_vectors_type = group.attrs["feature_vectors_type"]
        level_of_theory = group.attrs.get("level_of_theory")
        structure = Structure(
            positions=group["positions (Å)"][...],
            atomic_numbers=group["atomic_numbers"][...],
            masses=group["masses (u)"][...],
            cell_vectors=(
                group["cell_vectors (Å)"][...]
                if is_periodic else None
            ),
            n_frames=group.attrs["n_frames"],
            n_atoms=group.attrs["n_atoms"],
            periodic=is_periodic,
            feature_vectors_type=feature_vectors_type,
            feature_vectors=(
                group["feature_vectors"][...]
                if feature_vectors_type != "none" else None
            ),
            E_pot=(
                group["E_pot (eV∕atom)"][...]
                if "E_pot (eV∕atom)" in group else None
            ),
            forces=(
                group["forces (eV∕Å)"][...]
                if "forces (eV∕Å)" in group else None
            ),
            ground_truth=_read_ground_truth(f, key=f"{key}/ground_truth"),
            level_of_theory=level_of_theory,
        )
    return structure


def save_trajectory(
        dataset: str,
        key: str,
        traj: Trajectory,
        update_mode: Literal["update_properties", "replace"] = "update_properties",
):
    """
    Save the trajectory to a dataset.

    Parameters
    ----------
    dataset : str
        Path to the HDF5 dataset file.
    key : str
        Key within the HDF5 file.
    traj : Trajectory
        Trajectory object to save.
    update_mode : Literal["update_properties", "replace"]
        Mode for updating existing data. Defaults to "update_properties".

    Notes
    -----
    When ``update_mode="update_properties"`` (default), if the HDF5 key already exists,
    basic structural data (positions, atomic_numbers, cell_vectors) is NOT saved.
    Only energies, forces, feature vectors (if missing), and ground truth data are updated.

    .. warning::
       If you modify the geometry of a Trajectory in memory and call save() with
       ``update_mode="update_properties"`` on an existing key, the file will contain
       the OLD geometry but NEW energies/forces. This corrupts the dataset integrity.
       Use ``update_mode="replace"`` if the geometry has changed.
    """

    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with dataset_file(dataset, "a") as f:
        if key in f and update_mode == "replace":
            del f[key]

        save_basics = (key not in f)

        group = f.require_group(key)
        
        if save_basics:
            group.attrs["dataclass"] = "Trajectory"
            group.attrs["ensemble"] = traj.ensemble
            group.attrs["n_frames"] = traj.n_frames
            group.attrs["n_atoms"] = traj.n_atoms
            group.attrs["periodic"] = traj.periodic
            group.attrs["feature_vectors_type"] = traj.feature_vectors_type
            group.attrs["target_temperature (K)"] = traj.target_temperature
            group.attrs["time_equilibration (fs)"] = traj.time_equilibration
            group.attrs["n_removed_trans_dof"] = traj.n_removed_trans_dof
            group.attrs["n_removed_rot_dof"] = traj.n_removed_rot_dof
            if traj.ensemble == "NPT":
                group.attrs["target_pressure (GPa)"] = traj.target_pressure

            if traj.level_of_theory is not None:
                group.attrs["level_of_theory"] = traj.level_of_theory

            group.create_dataset(
                name="time (fs)",
                data=traj.time
            )
            group.create_dataset(
                name="T (K)",
                data=traj.temperature
            )
            if traj.ensemble == "NPT":
                group.create_dataset(
                    name="p (GPa)",
                    data=traj.pressure
                )
                group.create_dataset(
                    name="V (Å³∕atom)",
                    data=traj.volume
                )
            group.create_dataset(
                name="forces (eV∕Å)",
                data=traj.forces
            )
            group.create_dataset(
                name="velocities (Å∕fs)",
                data=traj.velocities
            )
            group.create_dataset(
                name="E_kin (eV∕atom)",
                data=traj.E_kin
            )
            group.create_dataset(
                name="E_pot (eV∕atom)",
                data=traj.E_pot
            )   
            group.create_dataset(
                name="E_trans_drift (eV∕atom)",
                data=traj.E_trans_drift
            )
            if not traj.periodic:
                group.create_dataset(
                    name="E_rot_drift (eV∕atom)",
                    data=traj.E_rot_drift
                )
            group.create_dataset(
                name="positions (Å)",
                data=traj.positions
            )
            group.create_dataset(
                name="atomic_numbers",
                data=traj.atomic_numbers
            )
            group.create_dataset(
                name="masses (u)",
                data=traj.masses
            )
            if traj.periodic:
                group.create_dataset(
                    name="cell_vectors (Å)",
                    data=traj.cell_vectors
                )
        if save_basics and traj.feature_vectors_type != "none":
            group.create_dataset(
                name="feature_vectors",
                data=traj.feature_vectors
            )
            
        elif update_mode == "update_properties":
            #
            # If we are in update_properties mode and the trajectory already exists,
            # we check if we need to add feature vectors which might have been computed
            # after the trajectory was saved.
            #
            if traj.feature_vectors is not None and traj.feature_vectors_type != "none":
                if "feature_vectors" not in group:
                    group.create_dataset(
                        name="feature_vectors",
                        data=traj.feature_vectors
                    )
                    group.attrs["feature_vectors_type"] = traj.feature_vectors_type

        if traj.ground_truth is not None:
            _save_ground_truth(f, f"{key}/ground_truth", traj.ground_truth, update_mode=update_mode)


def read_trajectory(dataset: str, key: str) -> Trajectory:
    
    with dataset_file(dataset, "r") as f:
        group = f[key]
        is_periodic = group.attrs["periodic"]
        feature_vectors_type = group.attrs["feature_vectors_type"]
        ensemble = group.attrs["ensemble"]
        level_of_theory = group.attrs.get("level_of_theory")
        traj = Trajectory(
            ensemble=ensemble,
            positions=group["positions (Å)"][...],
            atomic_numbers=group["atomic_numbers"][...],
            masses=group["masses (u)"][...],
            cell_vectors=(group["cell_vectors (Å)"][...] if is_periodic else None),
            n_frames=group.attrs["n_frames"],
            n_atoms=group.attrs["n_atoms"],
            periodic=is_periodic,
            time=group["time (fs)"][...],
            temperature=group["T (K)"][...],
            pressure=(group["p (GPa)"][...] if ensemble=="NPT" else None),
            volume=(group["V (Å³∕atom)"][...] if ensemble=="NPT" else None),
            forces=group["forces (eV∕Å)"][...],
            velocities=group["velocities (Å∕fs)"][...],
            E_kin=group["E_kin (eV∕atom)"][...],
            E_pot=group["E_pot (eV∕atom)"][...],
            E_trans_drift=group["E_trans_drift (eV∕atom)"][...],
            E_rot_drift=(group["E_rot_drift (eV∕atom)"][...] if not is_periodic else None),
            feature_vectors_type=feature_vectors_type,
            feature_vectors=(
                group["feature_vectors"][...]
                if feature_vectors_type != "none" else None
            ),
            target_temperature=group.attrs["target_temperature (K)"],
            target_pressure=(group.attrs["target_pressure (GPa)"] if ensemble=="NPT" else None),
            time_equilibration=group.attrs["time_equilibration (fs)"],
            n_removed_trans_dof=group.attrs["n_removed_trans_dof"],
            n_removed_rot_dof=group.attrs["n_removed_rot_dof"],
            ground_truth=_read_ground_truth(f, key=f"{key}/ground_truth"),
            level_of_theory=level_of_theory,
        )
        
    return traj


def save_force_constants(
    dataset: str,
    key: str,
    phonons: phonopy.Phonopy
):
    """Save force constants with their primitive and supercell structures."""

    assert isinstance(phonons.force_constants, np.ndarray)
    assert np.issubdtype(phonons.force_constants.dtype, np.floating)
    assert isinstance(phonons.supercell_matrix, np.ndarray)
    assert np.issubdtype(phonons.supercell_matrix.dtype, np.integer)
    assert isinstance(phonons.supercell, phonopy.structure.atoms.PhonopyAtoms)
    assert isinstance(phonons.primitive, phonopy.structure.atoms.PhonopyAtoms)

    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with dataset_file(dataset, "a") as f:
        if key in f:
            del f[key]
        group = f.create_group(key)
        group.attrs["dataclass"] = "ForceConstants"
        group.create_dataset(
            "force_constants (eV∕Å²)",
            data=phonons.force_constants
        )
        group.create_dataset(
            "supercell_matrix",
            data=phonons.supercell_matrix
        )

    primitive = phonons.primitive
    save_structure(
        dataset=dataset,
        key=f"{key}/primitive",
        positions=primitive.positions,
        atomic_numbers=primitive.numbers,
        masses=primitive.masses,
        cell_vectors=primitive.cell
    )
    
    supercell = phonons.supercell
    save_structure(
        dataset=dataset,
        key=f"{key}/supercell",
        positions=supercell.positions,
        atomic_numbers=supercell.numbers,
        masses=supercell.masses,
        cell_vectors=supercell.cell
    )

        
def read_force_constants(dataset: str, key: str) -> ForceConstants:
    """Read force constants and their associated structures."""

    with dataset_file(dataset, "r") as data:
        group = data[key]
        fc = group["force_constants (eV∕Å²)"][...]
        supercell_matrix = group["supercell_matrix"][...]
    primitive = read_structure(dataset, f"{key}/primitive")
    supercell = read_structure(dataset, f"{key}/supercell")
    return ForceConstants(
        force_constants=fc,
        supercell_matrix=supercell_matrix,
        primitive=primitive,
        supercell=supercell
    )


def save_molecular_crystal(
        dataset: str,
        key: str,
        system: MolecularCrystal
):
    """
    Save a MolecularCrystal object to a dataset.

    If molecules have different compositions, the index_map (a list of arrays)
    is converted to a single 2D array padded with -1.
    """

    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with dataset_file(dataset, "a") as f:
        if key in f:
            del f[key]
        
        group = f.create_group(key)
        group.attrs["dataclass"] = "MolecularCrystal"
        group.attrs["n_molecules"] = system.n_molecules
        group.attrs["identical_composition"] = system.identical_composition
        group.attrs["central_molecule_index"] = system.central_molecule_index
        group.create_dataset(
            name="centers_of_mass (Å)",
            data=system.centers_of_mass
        )
        index_map_to_save = system.index_map
        if not system.identical_composition:
            max_len = max(len(indices) for indices in system.index_map)
            padded_array = np.full(
                (system.n_molecules, max_len), 
                fill_value=-1, 
                dtype=system.index_map[0].dtype
            )
            for i, indices in enumerate(system.index_map):
                padded_array[i, :len(indices)] = indices
            index_map_to_save = padded_array

        group.create_dataset(
            name="index_map",
            data=index_map_to_save
        )
        group.create_dataset(
            name="min_distances_to_central_molecule (Å)",
            data=system.min_distances_to_central_molecule
        )
        group.create_dataset(
            name="max_distances_to_central_molecule (Å)",
            data=system.max_distances_to_central_molecule
        )
        
    save_structure(
        dataset=dataset,
        key=f"{key}/supercell",
        structure=system.supercell
    )


def read_molecular_crystal(dataset: str, key: str) -> MolecularCrystal:
    """
    Read a MolecularCrystal object from a dataset.
    
    """
    with dataset_file(dataset, "r") as f:
        group = f[key]
        
        n_molecules = group.attrs["n_molecules"]
        identical_composition = group.attrs["identical_composition"]
        central_molecule_index = group.attrs["central_molecule_index"]
        centers_of_mass = group["centers_of_mass (Å)"][...]
        if identical_composition:
            index_map = group["index_map"][...]
        else:
            padded_index_map = group["index_map"][...]
            index_map = [row[row != -1] for row in padded_index_map]
        min_distances_to_central_molecule = group["min_distances_to_central_molecule (Å)"][...]
        max_distances_to_central_molecule = group["max_distances_to_central_molecule (Å)"][...]

    supercell = read_structure(dataset, key=f"{key}/supercell")

    return MolecularCrystal(
        supercell=supercell,
        index_map=index_map,
        centers_of_mass=centers_of_mass,
        identical_composition=identical_composition,
        n_molecules=n_molecules,
        central_molecule_index=central_molecule_index,
        min_distances_to_central_molecule=min_distances_to_central_molecule,
        max_distances_to_central_molecule=max_distances_to_central_molecule
    )


def save_finite_subsystem(
        dataset: str,
        key: str,
        subsystem: FiniteSubsystem,
        only: List[Literal[*DATA_FOR_TRAINING]] | Literal[*DATA_FOR_TRAINING] | None = None,
) -> None:
    """Save a FiniteSubsystem object to a dataset."""

    if isinstance(only, str):
        only = [only]

    if only is None:
        Path(dataset).parent.mkdir(parents=True, exist_ok=True)
        with dataset_file(dataset, "a") as f:
            if key in f:
                del f[key]  
            group = f.create_group(key)
            group.attrs["dataclass"] = "FiniteSubsystem"
            group.attrs["n_molecules"] = subsystem.n_molecules
            group.create_dataset("molecule_indices", data=subsystem.molecule_indices)
            
        save_structure(
            structure=subsystem.cluster_of_molecules,
            dataset=dataset,
            key=f"{key}/cluster_of_molecules"
        )

    else:
        _save_only(
            dataset=dataset,
            key=f"{key}/cluster_of_molecules",
            structure=subsystem.cluster_of_molecules,
            quantities=only,
        )
        
    return


def read_finite_subsystem(dataset: str, key: str) -> FiniteSubsystem:

    structure = read_structure(
        dataset,
        key=f"{key}/cluster_of_molecules"
    )
    with dataset_file(dataset, "r") as f:
        group = f[key]
        n_molecules = group.attrs["n_molecules"]
        molecule_indices = group["molecule_indices"][...]

    return FiniteSubsystem(
        cluster_of_molecules=structure,
        molecule_indices=molecule_indices,
        n_molecules=n_molecules
    )


def save_attribute(
        dataset: str,
        key: str,
        attribute_name: str,
        attribute_value
):
    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with dataset_file(dataset, "a") as f:
        group = f.require_group(key)
        group.attrs[attribute_name] = attribute_value

    return


def read_attribute(
        dataset: str,
        key: str,
        attribute_name: str,
):
    with dataset_file(dataset, "r") as f:
        group = f[key]
        attribute_value = group.attrs[attribute_name]

    return attribute_value


def save_unique_clusters(
        dataset: str,
        key: str,
        clusters: UniqueClusters
) -> None:
    """Save a UniqueClusters object to a dataset."""

    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with dataset_file(dataset, "a") as f:
        if key in f:
            del f[key]
        group = f.create_group(key)
        group.attrs["dataclass"] = "UniqueClusters"
        group.attrs["n_clusters"] = clusters.n_clusters
        group.create_dataset("molecule_indices", data=clusters.molecule_indices)
        group.create_dataset("weights", data=clusters.weights)
        group.create_dataset("min_distances (Å)", data=clusters.min_distances)
        group.create_dataset("max_distances (Å)", data=clusters.max_distances)


def read_unique_clusters(dataset: str, key: str) -> UniqueClusters:
    """Read a UniqueClusters object from a dataset."""
    with dataset_file(dataset, "r") as f:
        group = f[key]
        return UniqueClusters(
            n_clusters=group.attrs["n_clusters"],
            molecule_indices=group["molecule_indices"][...],
            weights=group["weights"][...],
            min_distances=group["min_distances (Å)"][...],
            max_distances=group["max_distances (Å)"][...],
        )


def _save_only(
        dataset: str,
        key: str,
        structure: Structure,
        quantities: List[Literal[*DATA_FOR_TRAINING]],
) -> None:
    """
    Save selected physical quantities from
    a Structure object into a permanent storage
    dataset.
    
    This function is designed to update the dataset
    with data needed for training machine learning
    interatomic potentials.
    
    Keeps the rest of the saved Structure object unaltered.
    """

    with dataset_file(dataset, "r+") as f:
        group = f[key]

        if group.attrs["n_frames"] != structure.n_frames:
             raise ValueError(
                f"Cannot save partial data: frame count mismatch. "
                f"Existing dataset has {group.attrs['n_frames']} frames, "
                f"but the object being saved has {structure.n_frames} frames."
            )
        
        if "feature_vectors" in quantities:
            
            if structure.feature_vectors is None:
                raise RuntimeError(
                    "Feature vectors not present, cannot save requested data."
                )

            if "feature_vectors" in group:
                    del group["feature_vectors"]

            if structure.feature_vectors_type != "none":
                group.create_dataset(
                    name="feature_vectors",
                    data=structure.feature_vectors
                )
                
            group.attrs["feature_vectors_type"] = structure.feature_vectors_type

        if "ground_truth" in quantities:

            if structure.ground_truth is None:
                raise RuntimeError(
                    "Ground truth not present, cannot save requested data."
                )

            _save_ground_truth(f, f"{key}/ground_truth", structure.ground_truth, update_mode="update_properties")
            
    return

def _update_dataset(
        group: h5py.Group,
        dataset_name: str,
        new_data: npt.NDArray,
        method_name: str,
        ground_truth: GroundTruth,
        update_mode: str,
        energies_and_forces_data: bool,
) -> None:
    sanitized_method_name = method_name.replace("/", UNICODE_DIVISION_SLASH)
    
    data_to_write = new_data

    if update_mode == "update_properties" and dataset_name in group:
        existing_data = group[dataset_name][...]

        new_status = ground_truth.calculation_status.get(method_name)

        ds_status_name = f"status_{sanitized_method_name}"
        if ds_status_name in group:
            old_status = group[ds_status_name][...]
        else:
            old_status = np.full(existing_data.shape[0], CALCULATION_STATUS_COMPLETED)

        if new_status is not None:
            if energies_and_forces_data:
                mask = (new_status == CALCULATION_STATUS_COMPLETED)
                existing_data[mask] = new_data[mask]
                data_to_write = existing_data
            else:
                #
                # Writing calculation status. Here, it's importand to store 
                # the information not only which calculations are completed,
                # but also which calculations failed. Note that if the status
                # of a calculation is UNDEFINED, then it wasn't even started and
                # we have no information on it. It might happen that some other 
                # process has managed to complete the calculation because, e.g.,
                # it had access to more resources. Therefore, we're not updating
                # the status of COMPLETED calculations to avoid tagging valid
                # calculations as failed.
                #
                mask = (
                    new_status != CALCULATION_STATUS_UNDEFINED and 
                    old_status != CALCULATION_STATUS_COMPLETED
                )
                existing_data[mask] = new_data[mask]
                data_to_write = existing_data
    
    if dataset_name in group: del group[dataset_name]
    group.create_dataset(dataset_name, data=data_to_write)


def _save_ground_truth(
        f: h5py.File,
        key: str,
        ground_truth: GroundTruth,
        update_mode: Literal["update_properties", "replace"] = "update_properties",
) -> None:

    group = f.require_group(key)
    existing_levels_of_theory = set(group.attrs.get("levels_of_theory", []))
    levels_of_theory = existing_levels_of_theory.copy()

    for name, energy in ground_truth.energies.items():
        sanitized_method_name = name.replace("/", UNICODE_DIVISION_SLASH)
        sanitized_quantity_name = f"E_{sanitized_method_name} (eV∕atom)"
        
        _update_dataset(
            group=group,
            dataset_name=sanitized_quantity_name,
            new_data=energy,
            method_name=name,
            ground_truth=ground_truth,
            update_mode=update_mode,
            energies_and_forces_data=True,
        )
        levels_of_theory.add(name)

    for name, forces in ground_truth.forces.items():
        sanitized_method_name = name.replace("/", UNICODE_DIVISION_SLASH)
        ds_name = f"forces_{sanitized_method_name} (eV∕Å)"
        
        _update_dataset(
            group=group,
            dataset_name=ds_name,
            new_data=forces,
            method_name=name,
            ground_truth=ground_truth,
            update_mode=update_mode,
            energies_and_forces_data=True,
        )
        levels_of_theory.add(name)

    for name, status in ground_truth.calculation_status.items():
        sanitized_method_name = name.replace("/", UNICODE_DIVISION_SLASH)
        ds_name = f"status_{sanitized_method_name}"
        
        _update_dataset(
            group=group,
            dataset_name=ds_name,
            new_data=status,
            method_name=name,
            ground_truth=ground_truth,
            update_mode=update_mode,
            energies_and_forces_data=False,
        )
        #
        # We don't save the level_of_theory because there can't be a calculation status
        # without either energies or forces, so this information is redundant at this point.
        #

    group.attrs["levels_of_theory"] = sorted(list(levels_of_theory))

    return

def _read_ground_truth(f: h5py.File, key: str) -> GroundTruth | None:

    if key in f:
        group = f[key]
        levels_of_theory = group.attrs.get("levels_of_theory", [])

        energies = {}
        forces = {}
        calculation_status = {}

        for name in levels_of_theory:
            sanitized_name = name.replace("/", UNICODE_DIVISION_SLASH)
            if f"E_{sanitized_name} (eV∕atom)" in group:
                energies[name] = group[f"E_{sanitized_name} (eV∕atom)"][...]
            if f"forces_{sanitized_name} (eV∕Å)" in group:
                forces[name] = group[f"forces_{sanitized_name} (eV∕Å)"][...]
            
            if f"status_{sanitized_name}" in group:
                calculation_status[name] = group[f"status_{sanitized_name}"][...]
            else:
                # Fallback for backward compatibility
                # If we have energies, we can infer the number of frames from there
                # If not, we try forces.
                n_frames = 0
                if name in energies:
                    n_frames = len(energies[name])
                elif name in forces:
                    n_frames = len(forces[name])
                
                if n_frames > 0:
                    calculation_status[name] = np.full(
                        n_frames, 
                        CALCULATION_STATUS_COMPLETED, 
                        dtype=np.int64
                    )

        ground_truth = GroundTruth(
            energies=energies,
            forces=forces,
            calculation_status=calculation_status,
        )
    else:
        ground_truth = None

    return ground_truth

def _available_energies(
        structure: Structure,
        restrict_to: list[Literal["ground_truth", "structure_generation"]] | None = None
) -> list[str]:
    """
    Assemble a list of methods (levels of theory) for which
    energies are available in a given structure.
    """
    if restrict_to is None:
        restrict_to = ["ground_truth", "structure_generation"]
    
    methods = []
    if (
            "structure_generation" in restrict_to and
            structure.level_of_theory is not None and
            structure.E_pot is not None
    ):
        methods.append(structure.level_of_theory)

    if (
            "ground_truth" in restrict_to and
            structure.ground_truth is not None
    ):
        methods.extend([x for x in structure.ground_truth.energies])

    return methods

def _available_forces(
        structure: Structure,
        restrict_to: list[Literal["ground_truth", "structure_generation"]] | None = None
) -> list[str]:
    """
    Assemble a list of methods (levels of theory) for which
    forces are available in a given structure.
    """
    if restrict_to is None:
        restrict_to = ["ground_truth", "structure_generation"]
    
    methods = []    
    if (
            "structure_generation" in restrict_to and
            structure.level_of_theory is not None and
            structure.forces is not None
    ):
        #
        # Level of theory used to generate the geometry
        # via relaxation or molecular dynamics
        #
        methods.append(structure.level_of_theory) 

    if (
            "ground_truth" in restrict_to and
            structure.ground_truth is not None
    ):
        #
        # Levels of theory used to generate energies and
        # forces on a precomputed geometry
        #
        methods.extend([x for x in structure.ground_truth.forces])

    return methods
        
def _energies_at_level_of_theory(
        structure: Structure,
        level_of_theory: str,
) -> npt.NDArray[np.float64] | None:
    """
    Return energies at a given level of theory or None
    if energies are not present. level_of_theory can
    be either a ground truth method or the method used
    to obtain the geometry.
    """
    from_ground_truth = (
        structure.ground_truth is not None and
        level_of_theory in structure.ground_truth.energies
    )
    
    from_structure = (
        structure.level_of_theory is not None and
        structure.level_of_theory == level_of_theory and
        structure.E_pot is not None
    )
    
    if from_ground_truth:
        energies = structure.ground_truth.energies[level_of_theory]
    elif from_structure:
        energies = structure.E_pot
    else:
        energies = None
    
    return energies

def _forces_at_level_of_theory(
        structure: Structure,
        level_of_theory: str,
) -> npt.NDArray[np.float64] | None:
    """
    Return forces at a given level of theory or None
    if forces are not present. level_of_theory can
    be either a ground truth method or the method used
    to obtain the geometry.
    """
    
    from_ground_truth = (
        structure.ground_truth is not None and
        level_of_theory in structure.ground_truth.forces
    )
    
    from_structure = (
        structure.level_of_theory is not None and
        structure.level_of_theory == level_of_theory and
        structure.forces is not None
    )

    if from_ground_truth:
        forces = structure.ground_truth.forces[level_of_theory]
    elif from_structure:
        forces = structure.forces
    else:
        forces = None
    
    return forces

def save_atomic_reference(
        dataset: str | Path,
        key: str,
        atomic_reference: AtomicReference,
        overwrite: bool = False,
) -> None:

    dataset = Path(dataset)
    dataset.parent.mkdir(parents=True, exist_ok=True)

    key_E = "E (eV∕atom)"
    key_Z = "atomic_numbers"
    
    with dataset_file(dataset, "a") as f:
        if key in f:
            if overwrite:
                del f[key]
            else:
                raise RuntimeError(
                    f"Data with key '{key}' already exists in {dataset}. "
                    f"Set overwrite=True to replace it."
                )
        group = f.create_group(key)
        
        for method_name in atomic_reference.levels_of_theory:
            d = atomic_reference.energies[method_name]
            atomic_numbers = np.sort(np.fromiter(d.keys(), dtype=np.int64))
            energies = np.array([d[z] for z in atomic_numbers], dtype=np.float64)
            sanitized_method_name = method_name.replace("/", UNICODE_DIVISION_SLASH)

            subgroup = group.require_group(sanitized_method_name)
            subgroup.create_dataset(key_Z, data=atomic_numbers)
            subgroup.create_dataset(key_E, data=energies)
        
        group.attrs["levels_of_theory"] = sorted(atomic_reference.levels_of_theory)
        group.attrs["dataclass"] = "AtomicReference"

def read_atomic_reference(dataset: str | Path, key: str) -> AtomicReference:

    key_E = "E (eV∕atom)"
    key_Z = "atomic_numbers"
    energies = {}
    
    with dataset_file(dataset, "r") as f:
        atomic_reference_group = f[key]
        levels_of_theory = atomic_reference_group.attrs.get("levels_of_theory", [])
        
        for method_name in levels_of_theory:
            sanitized_method_name = method_name.replace("/", UNICODE_DIVISION_SLASH)            
            subgroup = atomic_reference_group[sanitized_method_name]
            energies[method_name] = dict(zip(
                subgroup[key_Z][...],  # atomic numbers
                subgroup[key_E][...]   # corresponding isolated atom energies for each Z (eV/atom)
            ))
            
    return AtomicReference(
        energies=energies
    )
