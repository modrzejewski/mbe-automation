import io
from typing import overload
import h5py
import ase
import ase.io.trajectory
import dynasor
import nglview.adaptor

import mbe_automation.storage.core

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
    Present a trajectory from a dataset file as an ASE-compatible object.

    The object supports indexing to retrieve ase.Atoms objects for each frame
    by reading data from the file on-demand. It is compatible with trajectories
    saved by the 'save_trajectory' function.
    """
    def __init__(self, dataset: str, key: str):
        """
        Initialize the trajectory by opening the dataset.

        Args:
            dataset (str): Path to the dataset.
            key (str): Key to the trajectory group within the dataset.
        """
        self.storage = h5py.File(dataset, "r")
        group = self.storage[key]

        self.positions = group["positions (Å)"]
        self.atomic_numbers = group["atomic_numbers"][:]
        self.masses = group["masses (u)"][:]
        
        self.periodic = group.attrs["periodic"]
        self.n_frames = group.attrs["n_frames"]

        self.cell_dataset = None
        if self.periodic:
            self.cell_dataset = group["cell_vectors (Å)"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def __len__(self):
        """Return the total number of frames."""
        return self.n_frames

    def __getitem__(self, i):
        """
        Get atoms object or a slice of the trajectory.
        """
        if isinstance(i, slice):
            return SlicedTrajectory(self, i)
        return self._get_atoms(i)

    def _get_atoms(self, index):
        """
        Return a single frame as an ase.Atoms object.
        """
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError("Trajectory index out of range")

        positions = self.positions[index]
        
        cell = None
        if self.periodic:
            if self.cell_dataset.ndim == 3:
                cell = self.cell_dataset[index]
            else:
                cell = self.cell_dataset[:]

        atoms = ase.Atoms(
            numbers=self.atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=self.periodic,
            masses=self.masses
        )
        return atoms

    def close(self):
        """Close the dataset file."""
        self.storage.close()


@overload
def to_ase_atoms(
        structure: mbe_automation.storage.core.Structure,
        frame_index: int = 0
) -> ase.Atoms: ...

@overload
def to_ase_atoms(*, dataset: str, key: str, frame_index: int = 0) -> ase.Atoms: ...

def to_ase_atoms(
    structure: mbe_automation.storage.core.Structure | None = None,
    *,
    dataset: str | None = None,
    key: str | None = None,
    frame_index: int = 0
) -> ase.Atoms:
    """Convert a single frame from a Structure to an ASE Atoms object.

    Can be called in two ways:
    1.  By providing a Structure object directly.
    2.  By providing a dataset path and a key to read the structure.
    """
    if structure is not None and (dataset is not None or key is not None):
        raise ValueError("Provide either a 'structure' object or 'dataset'/'key', not both.")

    if (dataset is not None and key is None) or (dataset is None and key is not None):
         raise ValueError("Both 'dataset' and 'key' must be provided together.")

    if structure is None:
        structure = mbe_automation.storage.core.read_structure(
            dataset,
            key
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


def to_dynasor_mode_projector(
        dataset: str,
        key: str
):
    
    fc = mbe_automation.storage.core.read_force_constants(
        dataset=dataset,
        key=key
    )
    primitive = to_ase_atoms(fc.primitive)
    supercell = to_ase_atoms(fc.supercell)
    mp = dynasor.ModeProjector(
        primitive=primitive,
        supercell=supercell,
        force_constants=fc.force_constants
    )
    return mp


class NGLViewTrajectory(nglview.adaptor.Trajectory, nglview.adaptor.Structure):
    """Adaptor for using mbe_automation.storage.Structure with nglview."""

    def __init__(self, trajectory_struct: mbe_automation.storage.core.Structure):

        if not isinstance(trajectory_struct, mbe_automation.storage.core.Structure):
            raise TypeError("Input must be an mbe_automation.storage.Structure object")
            
        self.trajectory_struct = trajectory_struct
        self.ext = "pdb"

    @property
    def n_frames(self) -> int:
        return self.trajectory_struct.n_frames

    def get_coordinates(self, index: int):
        return self.trajectory_struct.positions[index]

    def get_structure_string(self) -> str:

        first_frame_ase = to_ase_atoms(self.trajectory_struct, frame_index=0)
        with io.StringIO() as f:
            first_frame_ase.write(f, format="pdb")
            return f.getvalue()
