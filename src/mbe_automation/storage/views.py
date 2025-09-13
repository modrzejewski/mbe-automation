import h5py
import ase
import ase.io.trajectory

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
