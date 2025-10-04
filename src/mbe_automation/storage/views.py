from __future__ import annotations
import io
from typing import overload
from functools import singledispatch
import h5py
import ase
import ase.io.trajectory
import dynasor
import phonopy
import pymatgen
from phonopy.structure.atoms import PhonopyAtoms

from . import core

class SlicedTrajectory:
    """
    Wrapper to return a slice from a trajectory without loading
    from disk.
    """
    def __init__(self, trajectory, sliced_obj):
        self.trajectory = trajectory
        self.index_map = range(len(trajectory))[sliced_obj]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        if isinstance(i, slice):
            # Handle nested slicing
            new_slice = SlicedTrajectory(self.trajectory, slice(None))
            new_slice.index_map = self.index_map[i]
            return new_slice
        return self.trajectory[self.index_map[i]]


class ASETrajectory(ase.io.trajectory.TrajectoryReader):
    """
    Present a trajectory as an ASE-compatible object.

    The object can be initialized from a stored trajectory on disk or from an
    in-memory Structure object. It supports indexing to retrieve ase.Atoms
    objects for each frame on-demand.
    """
    @overload
    def __init__(self, structure: core.Structure): ...

    @overload
    def __init__(self, *, dataset: str, key: str): ...

    def __init__(
        self,
        structure: core.Structure | None = None,
        *,
        dataset: str | None = None,
        key: str | None = None
    ):
        self._is_file_based = False
        if structure is not None:
            # In-memory mode
            self.positions = structure.positions
            self.atomic_numbers = structure.atomic_numbers
            self.masses = structure.masses
            self.periodic = structure.periodic
            self.n_frames = structure.n_frames
            self.cell_vectors = structure.cell_vectors
            self.storage = None
        elif dataset is not None and key is not None:
            # File-based mode
            self._is_file_based = True
            self.storage = h5py.File(dataset, "r")
            group = self.storage[key]
            self.positions = group["positions (Å)"]
            self.atomic_numbers = group["atomic_numbers"][:]
            self.masses = group["masses (u)"][:]
            self.periodic = group.attrs["periodic"]
            self.n_frames = group.attrs["n_frames"]
            self.cell_vectors = None
            if self.periodic:
                self.cell_vectors = group["cell_vectors (Å)"]
        else:
            raise ValueError(
                "Provide either a 'structure' object or both 'dataset' and 'key'."
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def __len__(self):
        """Return the total number of frames."""
        return self.n_frames

    def __getitem__(self, i):
        """Get atoms object or a slice of the trajectory."""
        if isinstance(i, slice):
            return SlicedTrajectory(self, i)
        return self._get_atoms(i)

    def _get_atoms(self, index):
        """Return a single frame as an ase.Atoms object."""
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError("Trajectory index out of range")

        positions = self.positions[index]
        
        cell = None
        if self.periodic:
            if self.cell_vectors.ndim == 3:
                cell = self.cell_vectors[index]
            else:
                cell = self.cell_vectors[:]

        return ase.Atoms(
            numbers=self.atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=self.periodic,
            masses=self.masses
        )

    def close(self):
        """Close the dataset file if it was opened."""
        if self._is_file_based and self.storage:
            self.storage.close()

            
@singledispatch
def to_ase_converter(structure: object, frame_index: int = 0) -> ase.Atoms:
    """Generic converter function. The base implementation raises an error for unsupported types."""
    raise TypeError(f"to_ase does not support conversion for type {type(structure).__name__}")

@to_ase_converter.register
def _(structure: core.Structure, frame_index: int = 0) -> ase.Atoms:
    """
    Converter implementation for mbe_automation.storage.Structure.
    """
    
    if not 0 <= frame_index < structure.n_frames:
        raise IndexError(
            f"frame_index {frame_index} is out of bounds for a structure with "
            f"{structure.n_frames} frames."
        )

    if structure.positions.ndim == 3:
        positions = structure.positions[frame_index]
    else:
        positions = structure.positions

    cell = None
    if structure.periodic and structure.cell_vectors is not None:
        if structure.cell_vectors.ndim == 3:
            cell = structure.cell_vectors[frame_index]
        else:
            cell = structure.cell_vectors

    if structure.atomic_numbers.ndim == 2:
        atomic_numbers = structure.atomic_numbers[frame_index]
    else:
        atomic_numbers = structure.atomic_numbers

    if structure.masses.ndim == 2:
        masses = structure.masses[frame_index]
    else:
        masses = structure.masses

    return ase.Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=structure.periodic,
        masses=masses
    )

@to_ase_converter.register
def _(structure: PhonopyAtoms, frame_index: int = 0) -> ase.Atoms:
    """
    Converter implementation for phonopy.structure.atoms.PhonopyAtoms.
    The frame_index argument is ignored but is kept for a consistent signature.
    """

    return ase.Atoms(
        numbers=structure.numbers,
        scaled_positions=structure.scaled_positions,
        cell=structure.cell,
        pbc=True,  # PhonopyAtoms are always periodic
        masses=structure.masses
    )

# The public-facing function `to_ase` handles all input variations and then dispatches.
@overload
def to_ase(structure: core.Structure, frame_index: int = 0) -> ase.Atoms: ...

@overload
def to_ase(*, dataset: str, key: str, frame_index: int = 0) -> ase.Atoms: ...

@overload
def to_ase(structure: PhonopyAtoms, frame_index: int = 0) -> ase.Atoms: ...

def to_ase(
    structure: core.Structure | PhonopyAtoms | None = None,
    *,
    dataset: str | None = None,
    key: str | None = None,
    frame_index: int = 0
) -> ase.Atoms:
    """
    Convert a supported structure object or a file-based structure to an ASE Atoms object.

    This function can be called in multiple ways:
    1.  By providing a `mbe_automation.storage.Structure` object.
    2.  By providing a `phonopy.structure.atoms.PhonopyAtoms` object.
    3.  By providing a `dataset` path and a `key` to read a structure from a file.
    """

    if structure is not None and (dataset is not None or key is not None):
        raise ValueError("Provide either a 'structure' object or both 'dataset' and 'key', not both.")

    if structure is None:
        if dataset and key:
            structure = core.read_structure(dataset, key)
        else:
            raise ValueError("Provide either a structure object or both 'dataset' and 'key'.")

    return to_ase_converter(structure, frame_index=frame_index)

@overload
def to_pymatgen(
    structure: core.Structure,
    frame_index: int = 0
) -> Union[pymatgen.core.Structure, pymatgen.core.Molecule]: ...

@overload
def to_pymatgen(
    *, 
    dataset: str, 
    key: str, 
    frame_index: int = 0
) -> Union[pymatgen.core.Structure, pymatgen.core.Molecule]: ...

def to_pymatgen(
    structure: core.Structure | None = None,
    *,
    dataset: str | None = None,
    key: str | None = None,
    frame_index: int = 0
) -> Union[pymatgen.core.Structure, pymatgen.core.Molecule]:
    """Converts a single frame to a Pymatgen object.

    - Returns a `pymatgen.core.Structure` for periodic systems.
    - Returns a `pymatgen.core.Molecule` for non-periodic systems.

    Can be called in two ways:
    1.  By providing a Structure object directly.
    2.  By providing a dataset path and a key to read the structure.
    """
    if structure is not None and (dataset is not None or key is not None):
        raise ValueError("Provide either a 'structure' object or 'dataset'/'key', not both.")

    if (dataset is not None and key is None) or (dataset is None and key is not None):
         raise ValueError("Both 'dataset' and 'key' must be provided together.")

    if structure is None:
        structure = core.read_structure(
            dataset=dataset,
            key=key
        )

    if not 0 <= frame_index < structure.n_frames:
        raise IndexError(
            f"frame_index {frame_index} is out of bounds for a structure with "
            f"{structure.n_frames} frames."
        )

    if structure.positions.ndim == 3:
        positions = structure.positions[frame_index]
    else:
        positions = structure.positions

    if structure.atomic_numbers.ndim == 2:
        atomic_numbers = structure.atomic_numbers[frame_index]
    else:
        atomic_numbers = structure.atomic_numbers

    if structure.periodic:
        if structure.cell_vectors is None:
            raise ValueError("Periodic structure must have cell_vectors.")
            
        if structure.cell_vectors.ndim == 3:
            cell = structure.cell_vectors[frame_index]
        else:
            cell = structure.cell_vectors

        return pymatgen.core.Structure(
            lattice=cell,
            species=atomic_numbers,
            coords=positions,
            coords_are_cartesian=True
        )
    else:
        return pymatgen.core.Molecule(
            species=atomic_numbers,
            coords=positions
        )


def to_dynasor_mode_projector(
        dataset: str,
        key: str
):
    
    fc = core.read_force_constants(
        dataset=dataset,
        key=key
    )
    primitive = to_ase(fc.primitive)
    supercell = to_ase(fc.supercell)
    mp = dynasor.ModeProjector(
        primitive=primitive,
        supercell=supercell,
        force_constants=fc.force_constants
    )
    return mp


@overload
def to_phonopy(
    force_constants: core.ForceConstants
) -> phonopy.Phonopy: ...

@overload
def to_phonopy(*, dataset: str, key: str) -> phonopy.Phonopy: ...

def to_phonopy(
    force_constants: core.ForceConstants | None = None,
    *,
    dataset: str | None = None,
    key: str | None = None,
) -> phonopy.Phonopy:
    """
    Create a phonopy instance from stored or in-memory force constants.

    Can be called in two ways:
    1. By providing a ForceConstants object directly.
    2. By providing a dataset path and a key to read the data.
    """

    if force_constants is not None and (dataset is not None or key is not None):
        raise ValueError(
            "Provide either a 'force_constants' object or 'dataset'/'key', not both."
        )
    if (dataset is not None and key is None) or (dataset is None and key is not None):
        raise ValueError("Both 'dataset' and 'key' must be provided together.")

    if force_constants is not None:
        fc_data = force_constants
    elif dataset is not None and key is not None:
        fc_data = core.read_force_constants(dataset, key)
    else:
        raise ValueError("Either 'force_constants' or both 'dataset' and 'key' must be provided.")

    primitive_ph = PhonopyAtoms(
        numbers=fc_data.primitive.atomic_numbers,
        masses=fc_data.primitive.masses,
        cell=fc_data.primitive.cell_vectors,
        positions=fc_data.primitive.positions,
    )
    ph = phonopy.Phonopy(
        unitcell=primitive_ph,
        supercell_matrix=fc_data.supercell_matrix
    )
    ph.force_constants = fc_data.force_constants

    return ph


