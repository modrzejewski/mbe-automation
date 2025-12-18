# API Classes

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

| Class | Physical Meaning |
| :--- | :--- |
| **`ForceConstants`** | Represents interatomic force constants (typically 2nd order) derived from a crystal structure, used to compute phonon properties like frequencies and eigenvectors. |
| **`Structure`** | Represents an atomistic structure (positions, atomic numbers, unit cell). It can hold a single frame or a sequence of frames (e.g., from a short trajectory or a collection of configurations). |
| **`Trajectory`** | Represents the time-evolution of an atomistic system, typically resulting from a Molecular Dynamics simulation. It includes time-dependent properties like positions, velocities, kinetic energies, and thermodynamic variables. |
| **`MolecularCrystal`** | Represents a periodic crystal structure with additional topological information about its constituent molecules (e.g., connectivity, centers of mass, molecule indices). It serves as a bridge between the periodic crystal and its finite molecular components. |
| **`FiniteSubsystem`** | Represents a finite cluster of molecules extracted from a periodic crystal or trajectory. It is used to study local interactions or to generate training data for fragment-based methods. |
| **`Dataset`** | A container class that holds a collection of `Structure` or `FiniteSubsystem` objects. It is primarily used to aggregate data for generating machine learning training sets. |

## Methods Summary

The following table summarizes the key methods available across these classes.

| Method | Description | Available In |
| :--- | :--- | :--- |
| **`read`** | Factory method to load the object from an HDF5 dataset (wrapper around `mbe_automation.storage`). | `ForceConstants`, `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`save`** | Saves the object to an HDF5 dataset. | `MolecularCrystal`, `FiniteSubsystem` |
| **`subsample`** | Selects a representative subset of frames (e.g., using Farthest Point Sampling or k-means on feature vectors). | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`run_model`** | Executes a calculator (e.g., MACE, HF, DFT) on fixed, precomputed structures to compute energies, forces, and feature vectors. | `Structure`, `Trajectory`, `FiniteSubsystem` |
| **`to_mace_dataset`** | Exports the data (structures, energies, forces) to MACE-compatible XYZ files for model training. | `Structure`, `FiniteSubsystem`, `Dataset` |
| **`random_split`** | Randomly splits the frames into multiple objects (e.g., for creating training and validation sets). | `Structure`, `FiniteSubsystem` |
| **`to_ase_atoms`** | Converts a specific frame into an `ase.Atoms` object. | `Structure` |
| **`extract_..._molecules`** | Extracts isolated molecules (all, unique, or relaxed unique) from a periodic structure. | `Structure` |
| **`extract_finite_subsystems`** | Extracts finite clusters of molecules (e.g., dimers, trimers) based on distance or number criteria. | `MolecularCrystal` |
| **`frequencies_and_...`** | Calculates phonon frequencies and eigenvectors at a given k-point. | `ForceConstants` |
| **`append`** | Adds a structure or subsystem to the dataset collection. | `Dataset` |

### Usage Notes
*   **`subsample`**: Requires feature vectors to be present (computed via `run_model` or during an MD simulation) to calculate distances in chemical space.
*   **`run_model`**: Can be used to evaluate ground truth (energies and forces) or to generate descriptors (feature vectors) for subsampling.
