# API Data Classes

The core data structures of `mbe_automation` are available as Python classes. They can be imported directly from the top-level package:

```python
from mbe_automation import (
    ForceConstants,
    Structure,
    Trajectory,
    MolecularCrystal,
    FiniteSubsystem,
    Dataset
)
```

## Physical Content

| Class | Description |
| :--- | :--- |
| **`ForceConstants`** | Represents 2nd order force constants used to compute phonon properties like frequencies and dynamical matrix eigenvectors. |
| **`Structure`** | Represents an atomistic structure (positions, atomic numbers, unit cell). Can hold a single frame or a sequence of frames of equal size (e.g., from a short trajectory or a collection of configurations). |
| **`Trajectory`** | Represents the time-evolution of an atomistic system, typically resulting from a Molecular Dynamics simulation. It includes time-dependent properties like positions, velocities, kinetic energies, and thermodynamic variables. |
| **`MolecularCrystal`** | Represents a periodic crystal structure with additional topological information about its constituent molecules (e.g., connectivity, centers of mass, molecule indices). Serves as an intermediate necessary for finite cluster extraction. |
| **`FiniteSubsystem`** | Represents a finite cluster of molecules extracted from a periodic structure or trajectory. Used to generate training data for fragment-based methods. |
| **`Dataset`** | A container class that holds a collection of `Structure` or `FiniteSubsystem` objects. Aggregates data for machine learning training sets. |

## ForceConstants

Represents harmonic force constants.

### Methods

#### `read`
```python
@classmethod
def read(
    cls,
    dataset: str,
    key: str
) -> ForceConstants
```
Loads force constants from an HDF5 dataset.
*   `dataset`: Path to the HDF5 file.
*   `key`: Path to the group within the HDF5 file.

#### `frequencies_and_eigenvectors`
```python
def frequencies_and_eigenvectors(
    self,
    k_point: npt.NDArray[np.floating]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.complex128]]
```
Calculates phonon frequencies and eigenvectors at a given k-point.
*   `k_point`: The wave vector at which to calculate phonon properties.

## Structure

Represents an atomistic structure or a sequence of structures.

### Methods

#### `read`
```python
@classmethod
def read(
    cls,
    dataset: str,
    key: str
) -> Structure
```
Loads a structure from an HDF5 dataset.
*   `dataset`: Path to the HDF5 file.
*   `key`: Path to the group within the HDF5 file.

#### `save`
```python
def save(
    self,
    dataset: str,
    key: str,
    only: List[str] | None = None
) -> None
```
Saves the structure to an HDF5 dataset.
*   `dataset`: Path to the HDF5 file.
*   `key`: Path to the group within the HDF5 file.
*   `only`: Optional list of specific properties to update (e.g., `["feature_vectors"]`).

#### `from_xyz_file`
```python
@classmethod
def from_xyz_file(
    cls,
    read_path: str,
    transform_to_symmetrized_primitive: bool = True,
    symprec: float = 1e-3
) -> Structure
```
Creates a Structure from an XYZ file.
*   `read_path`: Path to the XYZ file.
*   `transform_to_symmetrized_primitive`: Whether to convert the structure to its symmetrized primitive cell.
*   `symprec`: Symmetry tolerance for the conversion.

#### `to_ase_atoms`
```python
def to_ase_atoms(
    self,
    frame_index: int = 0
) -> ase.Atoms
```
Converts a specific frame of the structure to an `ase.Atoms` object.
*   `frame_index`: The index of the frame to convert.

#### `subsample`
```python
def subsample(
    self,
    n: int,
    algorithm: str = "farthest_point_sampling",
    rng: np.random.Generator | None = None
) -> Structure
```
Selects a representative subset of frames based on feature vectors.
*   `n`: The number of frames to select.
*   `algorithm`: The subsampling algorithm ("farthest_point_sampling" or "kmeans").
*   `rng`: Random number generator for reproducibility.

#### `select`
```python
def select(
    self,
    indices: npt.NDArray[np.integer]
) -> Structure
```
Returns a new Structure containing only the specified frames.
*   `indices`: Array of frame indices to keep.

#### `random_split`
```python
def random_split(
    self,
    fractions: Sequence[float],
    rng: np.random.Generator | None = None
) -> Sequence[Structure]
```
Randomly splits the frames into multiple Structure objects.
*   `fractions`: Sequence of fractions summing to 1.0 (e.g., `[0.8, 0.2]`).
*   `rng`: Random number generator for reproducibility.

#### `run_model`
```python
def run_model(
    self,
    calculator: CALCULATORS,
    energies: bool = True,
    forces: bool = True,
    feature_vectors_type: str = "none",
    exec_params: ParallelCPU | None = None,
    overwrite: bool = False
) -> None
```
Executes a calculator on the structure's frames. Updates the structure in-place.
*   `calculator`: The calculator instance (e.g., MACE, DFTB).
*   `energies`: Whether to compute and store energies (in `ground_truth`).
*   `forces`: Whether to compute and store forces (in `ground_truth`).
*   `feature_vectors_type`: Type of feature vectors to generate ("none", "averaged_environments", "atomic_environments").
*   `exec_params`: Parallel execution parameters.
*   `overwrite`: Whether to overwrite existing data.

#### `to_molecular_crystal` (alias: `detect_molecules`)
```python
def to_molecular_crystal(
    self,
    reference_frame_index: int = 0,
    assert_identical_composition: bool = True,
    bonding_algo: NearNeighbors | None = None
) -> MolecularCrystal
```
Converts a periodic Structure into a MolecularCrystal by detecting molecules.
*   `reference_frame_index`: Frame index to use for molecule detection.
*   `assert_identical_composition`: Whether to require all molecules to have the same composition.
*   `bonding_algo`: Algorithm for determining atomic connectivity (defaults to VESTA-style cutoffs).

#### `extract_all_molecules`
```python
def extract_all_molecules(
    self,
    bonding_algo: NearNeighbors | None = None,
    reference_frame_index: int = 0,
    calculator: ASECalculator | None = None
) -> List[Structure]
```
Extracts all individual molecules from the crystal structure.
*   `bonding_algo`: Algorithm for determining atomic connectivity.
*   `reference_frame_index`: Frame index to use for extraction.
*   `calculator`: Optional calculator to attach to extracted molecules.

#### `extract_unique_molecules`
```python
def extract_unique_molecules(
    self,
    calculator: ASECalculator,
    energy_thresh: float = 1.0E-5,
    bonding_algo: NearNeighbors | None = None,
    reference_frame_index: int = 0
) -> List[Structure]
```
Extracts only unique molecules based on potential energy similarity.
*   `calculator`: Calculator used to compute energies for uniqueness check.
*   `energy_thresh`: Energy threshold (eV/atom) for considering molecules identical.
*   `bonding_algo`: Algorithm for determining connectivity.
*   `reference_frame_index`: Frame index to use.

#### `extract_relaxed_unique_molecules`
```python
def extract_relaxed_unique_molecules(
    self,
    dataset: str,
    key: str,
    calculator: ASECalculator,
    config: Minimum,
    energy_thresh: float = 1.0E-5,
    bonding_algo: NearNeighbors | None = None,
    reference_frame_index: int = 0,
    work_dir: Path | str = Path("./")
) -> None
```
Extracts unique molecules and relaxes their geometry, saving results to HDF5.
*   `dataset`: Output HDF5 file path.
*   `key`: Output HDF5 group path.
*   `calculator`: Calculator for relaxation.
*   `config`: Relaxation parameters.
*   `energy_thresh`: Threshold for uniqueness.
*   `work_dir`: Working directory for temporary files.

#### `to_mace_dataset`
```python
def to_mace_dataset(
    self,
    save_path: str,
    level_of_theory: str | dict[Literal["target", "baseline"], str],
    atomic_energies: dict | None = None
) -> None
```
Exports the structure to a MACE-compatible XYZ training file.
*   `save_path`: Path to save the XYZ file.
*   `level_of_theory`: Theoretical model key(s) for energies and forces.
*   `atomic_energies`: Dictionary of isolated atom energies for baseline subtraction.

#### `atomic_energies`
```python
def atomic_energies(
    self,
    calculator: CALCULATORS
) -> dict[np.int64, np.float64]
```
Calculates ground-state energies for isolated atoms present in the structure.
*   `calculator`: Calculator instance to use.

## Trajectory

Represents a time-evolution of a system. Inherits from `Structure`.

### Methods

#### `read`
```python
@classmethod
def read(
    cls,
    dataset: str,
    key: str
) -> Trajectory
```
Loads a trajectory from an HDF5 dataset.
*   `dataset`: Path to the HDF5 file.
*   `key`: Path to the group within the HDF5 file.

#### `save`
```python
def save(
    self,
    dataset: str,
    key: str,
    only: List[str] | None = None
) -> None
```
Saves the trajectory to an HDF5 dataset.
*   `dataset`: Path to the HDF5 file.
*   `key`: Path to the group within the HDF5 file.
*   `only`: Optional list of specific properties to update.

#### `subsample`
```python
def subsample(
    self,
    n: int,
    algorithm: str = "farthest_point_sampling",
    rng: np.random.Generator | None = None
) -> Trajectory
```
Selects a representative subset of frames based on feature vectors.
*   `n`: The number of frames to select.
*   `algorithm`: The subsampling algorithm.
*   `rng`: Random number generator.

#### `run_model`
```python
def run_model(
    self,
    calculator: CALCULATORS,
    energies: bool = True,
    forces: bool = True,
    feature_vectors_type: str = "none",
    exec_params: ParallelCPU | None = None,
    overwrite: bool = False
) -> None
```
Executes a calculator on the trajectory frames. See `Structure.run_model`.

## MolecularCrystal

Represents a periodic crystal with molecular topology information.

### Methods

#### `read`
```python
@classmethod
def read(
    cls,
    dataset: str,
    key: str
) -> MolecularCrystal
```
Loads a molecular crystal from an HDF5 dataset.

#### `save`
```python
def save(
    self,
    dataset: str,
    key: str
) -> None
```
Saves the molecular crystal to an HDF5 dataset.

#### `subsample`
```python
def subsample(
    self,
    n: int,
    algorithm: str = "farthest_point_sampling",
    rng: np.random.Generator | None = None
) -> MolecularCrystal
```
Selects a representative subset of frames.

#### `extract_finite_subsystems` (alias: `extract_finite_subsystem`)
```python
def extract_finite_subsystems(
    self,
    filter: FiniteSubsystemFilter | None = None
) -> List[FiniteSubsystem]
```
Extracts finite clusters of molecules (e.g., dimers) from the crystal.
*   `filter`: Configuration object defining selection rules (distances, numbers).

## FiniteSubsystem

Represents a finite cluster of molecules.

### Methods

#### `read`
```python
@classmethod
def read(
    cls,
    dataset: str,
    key: str
) -> FiniteSubsystem
```
Loads a finite subsystem from an HDF5 dataset.

#### `save`
```python
def save(
    self,
    dataset: str,
    key: str,
    only: List[str] | None = None
) -> None
```
Saves the finite subsystem to an HDF5 dataset.

#### `subsample`
```python
def subsample(
    self,
    n: int,
    algorithm: str = "farthest_point_sampling",
    rng: np.random.Generator | None = None
) -> FiniteSubsystem
```
Selects a representative subset of frames.

#### `run_model`
```python
def run_model(
    self,
    calculator: CALCULATORS,
    energies: bool = True,
    forces: bool = True,
    feature_vectors_type: str = "none",
    exec_params: ParallelCPU | None = None,
    overwrite: bool = False
) -> None
```
Executes a calculator on the subsystem frames.

#### `random_split`
```python
def random_split(
    self,
    fractions: Sequence[float],
    rng: np.random.Generator | None = None
) -> Sequence[FiniteSubsystem]
```
Randomly splits the frames into multiple FiniteSubsystem objects.

#### `to_mace_dataset`
```python
def to_mace_dataset(
    self,
    save_path: str,
    level_of_theory: str | dict[Literal["target", "baseline"], str],
    atomic_energies: dict | None = None
) -> None
```
Exports to MACE-compatible XYZ training file.

## Dataset

A collection of `Structure` or `FiniteSubsystem` objects.

### Methods

#### `append`
```python
def append(
    self,
    structure: Structure | FiniteSubsystem
)
```
Adds a structure or finite subsystem to the dataset.
*   `structure`: The object to add.

#### `statistics`
```python
def statistics(
    self,
    level_of_theory: str
) -> None
```
Prints mean and standard deviation of energy per atom for the specified level of theory.
*   `level_of_theory`: The key for the energy data in `ground_truth`.

#### `to_mace_dataset`
```python
def to_mace_dataset(
    self,
    save_path: str,
    level_of_theory: str | dict[Literal["target", "baseline"], str],
    atomic_energies: dict | None = None
) -> None
```
Exports the entire dataset to a single MACE-compatible XYZ training file.
*   `save_path`: Path to save the XYZ file.
*   `level_of_theory`: Theoretical model key(s).
*   `atomic_energies`: Dictionary of isolated atom energies.

#### `unique_elements` (Property)
```python
@property
def unique_elements(self) -> npt.NDArray[np.int64]
```
Returns a sorted NumPy array of unique atomic numbers present across all structures in the dataset.
