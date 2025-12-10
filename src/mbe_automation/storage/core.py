from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Literal, overload
import pandas as pd
import phonopy
from phonopy import Phonopy
import h5py
import numpy as np
import numpy.typing as npt
import os

DATA_FOR_TRAINING = [
    "feature_vectors",
    "potential_energies",
    "forces",
    "delta",
]

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
class DeltaTargetBaseline:
    E_pot_baseline: npt.NDArray[np.floating] | None = None # eV/atom
    forces_baseline: npt.NDArray[np.floating] | None = None # eV/Angs
    E_pot_target: npt.NDArray[np.floating] | None = None # eV/atom
    forces_target: npt.NDArray[np.floating] | None = None # eV/Angs
    E_atomic_baseline: npt.NDArray[np.float64] | None = None # eV/atom

    def copy(self) -> DeltaTargetBaseline:
        return DeltaTargetBaseline(
            E_pot_baseline=(self.E_pot_baseline.copy() if self.E_pot_baseline is not None else None),
            forces_baseline=(self.forces_baseline.copy() if self.forces_baseline is not None else None),
            E_pot_target=(self.E_pot_target.copy() if self.E_pot_target is not None else None),
            forces_target=(self.forces_target.copy() if self.forces_target is not None else None),
            E_atomic_baseline=(self.E_atomic_baseline.copy() if self.E_atomic_baseline is not None else None),
        )

    def select_frames(self, indices: npt.NDArray[np.integer]) -> DeltaTargetBaseline:
        return DeltaTargetBaseline(
            E_pot_baseline=(self.E_pot_baseline[indices] if self.E_pot_baseline is not None else None),
            forces_baseline=(self.forces_baseline[indices] if self.forces_baseline is not None else None),
            E_pot_target=(self.E_pot_target[indices] if self.E_pot_target is not None else None),
            forces_target=(self.forces_target[indices] if self.forces_target is not None else None),
            E_atomic_baseline=self.E_atomic_baseline, # composition is the same in all frames
        )
        
@dataclass
class Structure:
    positions: npt.NDArray[np.floating]
    atomic_numbers: npt.NDArray[np.integer]
    masses: npt.NDArray[np.floating]
    cell_vectors: npt.NDArray[np.floating] | None
    n_frames: int
    n_atoms: int
    periodic: bool = False
    E_pot: npt.NDArray[np.floating] | None = None
    forces: npt.NDArray[np.floating] | None = None
    delta: DeltaTargetBaseline | None = None
    feature_vectors: npt.NDArray[np.floating] | None = None
    feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES] = "none"
    
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
            delta=(self.delta.copy() if self.delta is not None else None),
            feature_vectors=(
                self.feature_vectors.copy() 
                if self.feature_vectors_type != "none" else None
            ),
            feature_vectors_type=self.feature_vectors_type,
        )
    
    def save(
            self,
            dataset: str,
            key: str,
            only: List[Literal[*DATA_FOR_TRAINING]] | None = None,
    ) -> None:
        """Save the structure to a dataset."""

        if only is None:
            save_structure(
                dataset=dataset,
                key=key,
                structure=self,
            )
        else:
            _save_only(
                dataset=dataset,
                key=key,
                structure=self,
                quantities=only,
            )

    @property
    def unique_elements(self) -> npt.NDArray[np.int64]:
        return np.unique(np.atleast_2d(self.atomic_numbers)[0])

    @property
    def z_map(self) -> npt.NDArray[np.int64]:
        unique_elements = self.unique_elements
        n_elements = len(unique_elements)
        x = np.full(np.max(unique_elements) + 1, -1, dtype=np.int64)
        x[unique_elements] = np.arange(n_elements)
        return x

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
            n_removed_rot_dof: int = 0
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
            n_removed_rot_dof=n_removed_rot_dof
        )

    def save(
            self,
            dataset: str,
            key: str,
            only: List[Literal[*DATA_FOR_TRAINING]] | None = None,
    ) -> None:
        """Save the trajectory to a dataset."""

        if only is None:
            save_trajectory(
                dataset=dataset,
                key=key,
                traj=self,
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
    
def save_data_frame(
        dataset: str,
        key: str,
        df: pd.DataFrame
):
    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset, "a") as f:
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
    
    with h5py.File(dataset, "r") as f:
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
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
            
        group = f.create_group(key)
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

    with h5py.File(dataset, "r") as f:
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
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
            
        group = f.create_group(key)
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
):

    with h5py.File(dataset, "r") as f:
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
        feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES]="none",
        delta: DeltaTargetBaseline | None=None,
):
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
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
        group = f.require_group(key)

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
        if delta is not None:
            _save_delta_target_baseline(f, f"{key}/delta", delta)
            
        group.attrs["n_frames"] = n_frames
        group.attrs["n_atoms"] = n_atoms
        group.attrs["periodic"] = is_periodic
        group.attrs["feature_vectors_type"] = feature_vectors_type

@overload
def save_structure(*, dataset: str, key: str, structure: Structure) -> None: ...

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
    delta: DeltaTargetBaseline | None = None,
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
            delta=structure.delta,
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
            delta=kwargs.get("delta"),
            feature_vectors_type=kwargs.get("feature_vectors_type", "none"),
        )
    else:
        raise ValueError(
            "Either a 'structure' object or 'positions' and other arrays "
            "must be provided as keyword arguments."
        )
        

def read_structure(dataset, key):
    with h5py.File(dataset, "r") as f:
        group = f[key]
        is_periodic = group.attrs["periodic"]
        feature_vectors_type = group.attrs["feature_vectors_type"]
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
            delta=_read_delta_target_baseline(f, key=f"{key}/delta"),
        )
    return structure


def save_trajectory(
        dataset: str,
        key: str,
        traj: Trajectory
):

    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]

        group = f.require_group(key)
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
        if traj.feature_vectors_type != "none":
            group.create_dataset(
                name="feature_vectors",
                data=traj.feature_vectors
            )
      

def read_trajectory(dataset: str, key: str) -> Trajectory:
    
    with h5py.File(dataset, "r") as f:
        group = f[key]
        is_periodic = group.attrs["periodic"]
        feature_vectors_type = group.attrs["feature_vectors_type"]
        ensemble = group.attrs["ensemble"]
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
            n_removed_rot_dof=group.attrs["n_removed_rot_dof"]
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
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
        group = f.create_group(key)
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

    with h5py.File(dataset, "r") as data:
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
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
        
        group = f.create_group(key)

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
    with h5py.File(dataset, "r") as f:
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
        only: List[Literal[*DATA_FOR_TRAINING]] | None = None,
) -> None:
    """Save a FiniteSubsystem object to a dataset."""

    if only is None:
        Path(dataset).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dataset, "a") as f:
            if key in f:
                del f[key]  
            group = f.create_group(key)
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
    with h5py.File(dataset, "r") as f:
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
    with h5py.File(dataset, "a") as f:
        group = f.require_group(key)
        group.attrs[attribute_name] = attribute_value

    return


def read_attribute(
        dataset: str,
        key: str,
        attribute_name: str,
):
    with h5py.File(dataset, "r") as f:
        group = f.create_group(key)
        attribute_value = group.attrs[attribute_name]

    return attribute_value


def save_unique_clusters(
        dataset: str,
        key: str,
        clusters: UniqueClusters
) -> None:
    """Save a UniqueClusters object to a dataset."""

    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
        group = f.create_group(key)
        group.attrs["n_clusters"] = clusters.n_clusters
        group.create_dataset("molecule_indices", data=clusters.molecule_indices)
        group.create_dataset("weights", data=clusters.weights)
        group.create_dataset("min_distances (Å)", data=clusters.min_distances)
        group.create_dataset("max_distances (Å)", data=clusters.max_distances)


def read_unique_clusters(dataset: str, key: str) -> UniqueClusters:
    """Read a UniqueClusters object from a dataset."""
    with h5py.File(dataset, "r") as f:
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

    with h5py.File(dataset, "r+") as f:
        group = f[key]
        
        if "feature_vectors" in quantities:
            
            if structure.feature_vectors is None:
                raise RuntimeError(
                    "Feature vectors are not present in the Structure object. "
                    "Execute run_model to compute the required data."
                )

            if "feature_vectors" in group:
                    del group["feature_vectors"]

            if structure.feature_vectors_type != "none":
                group.create_dataset(
                    name="feature_vectors",
                    data=structure.feature_vectors
                )
                
            group.attrs["feature_vectors_type"] = structure.feature_vectors_type

        if "potential_energies" in quantities:

            if structure.E_pot is None:
                raise RuntimeError(
                    "Potential energies are not present in the Structure object."
                )
            
            if "E_pot (eV∕atom)" in group:
                del group["E_pot (eV∕atom)"]

            group.create_dataset(
                name="E_pot (eV∕atom)",
                data=structure.E_pot
            )

        if "forces" in quantities:

            if structure.forces is None:
                raise RuntimeError(
                    "Forces are not present in the Structure object."
                )

            if "forces (eV∕Å)" in group:
                del group["forces (eV∕Å)"]

            group.create_dataset(
                name="forces (eV∕Å)",
                data=structure.forces
            )

        if "delta" in quantities:

            if structure.delta is None:
                raise RuntimeError(
                    "Delta learning data are not present in the Structure object."
                )

            _save_delta_target_baseline(f, f"{key}/delta", structure.delta)
            
    return


def _save_delta_target_baseline(
        f: h5py.File,
        key: str,
        delta: DeltaTargetBaseline,
) -> None:

    group = f.require_group(key)

    if delta.forces_baseline is not None:
        if "forces_baseline (eV∕Å)" in group: del group["forces_baseline (eV∕Å)"]
        group.create_dataset("forces_baseline (eV∕Å)", data=delta.forces_baseline)

    if delta.forces_target is not None:
        if "forces_target (eV∕Å)" in group: del group["forces_target (eV∕Å)"]
        group.create_dataset("forces_target (eV∕Å)", data=delta.forces_target)

    if delta.E_pot_baseline is not None:
        if "E_pot_baseline (eV∕atom)" in group: del group["E_pot_baseline (eV∕atom)"]
        group.create_dataset(name="E_pot_baseline (eV∕atom)", data=delta.E_pot_baseline)

    if delta.E_pot_target is not None:
        if "E_pot_target (eV∕atom)" in group: del group["E_pot_target (eV∕atom)"]
        group.create_dataset(name="E_pot_target (eV∕atom)", data=delta.E_pot_target)

    if delta.E_atomic_baseline is not None:
        if "E_atomic_baseline (eV∕atom)" in group: del group["E_atomic_baseline (eV∕atom)"]
        group.create_dataset(name="E_atomic_baseline (eV∕atom)", data=delta.E_atomic_baseline)

    return


def _read_delta_target_baseline(f: h5py.File, key: str) -> DeltaTargetBaseline | None:

    if key in f:
        group = f[key]        
        delta = DeltaTargetBaseline(
            forces_baseline=(
                group["forces_baseline (eV∕Å)"][...] if "forces_baseline (eV∕Å)" in group else None),
            forces_target=(
                group["forces_target (eV∕Å)"][...] if "forces_target (eV∕Å)" in group else None),
            E_pot_baseline=(
                group["E_pot_baseline (eV∕atom)"][...] if "E_pot_baseline (eV∕atom)" in group else None
            ),
            E_pot_target=(
                group["E_pot_target (eV∕atom)"][...] if "E_pot_target (eV∕atom)" in group else None
            ),
            E_atomic_baseline=(
                group["E_atomic_baseline (eV∕atom)"][...] if "E_atomic_baseline (eV∕atom)" in group else None
            ),
        )
    else:
        delta = None

    return delta

