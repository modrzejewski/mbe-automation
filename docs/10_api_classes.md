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

This chapter provides an overview of the physical content each class represents and a summary of the methods available for processing them.

## Physical Content

| Class | Description |
| :--- | :--- |
| **`ForceConstants`** | Represents 2nd order force constants used to compute phonon properties like frequencies and dynamical matrix eigenvectors. |
| **`Structure`** | Represents an atomistic structure (positions, atomic numbers, unit cell). Can hold a single frame or a sequence of frames of equal size (e.g., from a short trajectory or a collection of configurations). |
| **`Trajectory`** | Represents the time-evolution of an atomistic system, typically resulting from a Molecular Dynamics simulation. It includes time-dependent properties like positions, velocities, kinetic energies, and thermodynamic variables. |
| **`MolecularCrystal`** | Represents a periodic crystal structure with additional topological information about its constituent molecules (e.g., connectivity, centers of mass, molecule indices). Serves as an intermediate necessary for finite cluster extraction. |
| **`FiniteSubsystem`** | Represents a finite cluster of molecules extracted from a periodic structure or trajectory. Used to generate training data for fragment-based methods. |
| **`Dataset`** | A container class that holds a collection of `Structure` or `FiniteSubsystem` objects. Aggregates data for machine learning training sets. |

## Methods Summary

The following table summarizes the key methods available across these classes.

| Method | Description | Available In |
| :--- | :--- | :--- |
| **`read`** | Method to load the object from an HDF5 dataset. | `ForceConstants`, `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`save`** | Saves the object to an HDF5 dataset. | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`subsample`** | Selects a representative subset of frames (e.g., using Farthest Point Sampling or k-means on feature vectors). | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`run_model`** | Executes a calculator on fixed structures. Computed energies and forces are stored in `ground_truth` (indexed by the calculator's `level_of_theory`), while feature vectors are stored directly on the structure for subsampling. | `Structure`, `Trajectory`, `FiniteSubsystem` |
| **`to_mace_dataset`** | Exports the data (structures, energies, forces) to MACE-compatible XYZ files for model training. | `Structure`, `FiniteSubsystem`, `Dataset` |
| **`random_split`** | Randomly splits the frames into multiple objects (e.g., for creating training and validation sets). | `Structure`, `FiniteSubsystem` |
| **`to_ase_atoms`** | Converts a specific frame into an `ase.Atoms` object. | `Structure` |
| **`extract_..._molecules`** | Extracts isolated molecules (all, unique, or relaxed unique) from a periodic structure. | `Structure` |
| **`extract_finite_subsystems`** | Extracts finite clusters of molecules (e.g., dimers, trimers) based on distance or number criteria. | `MolecularCrystal` |
| **`frequencies_and_...`** | Calculates phonon frequencies and eigenvectors at a given k-point. | `ForceConstants` |
| **`append`** | Adds a structure or subsystem to the dataset collection. | `Dataset` |
| **`unique_elements`** | Returns a sorted array of unique atomic numbers present in the object. | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem`, `Dataset` |
| **`atomic_energies`** | Calculates ground-state energies for all unique isolated atoms (used for generating MLIP baselines). | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem`, `Dataset` |

### Usage Notes
*   **`subsample`**: Requires feature vectors to be present (computed via `run_model` or during an MD simulation) to calculate distances in chemical space.
*   **`run_model`**: Can be used to evaluate ground truth (energies and forces) or to generate descriptors (feature vectors) for subsampling.
