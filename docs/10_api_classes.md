# API Data Classes

The core data structures of `mbe_automation` are available as Python classes. They can be imported directly from the top-level package:

```python
from mbe_automation import (
    AtomicReference,
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
| **`ForceConstants`** | Second order force constants and associated physical quantities used to compute phonon properties. Needed for frequencies and dynamical matrix eigenvectors. |
| **`Structure`** | Atomistic structure (positions, atomic numbers, cell vectors). Can hold a single frame or a sequence of frames of equal size (e.g., from a short trajectory or a collection of configurations). |
| **`Trajectory`** | Time evolution of an atomistic system generated with molecular dynamics. Includes time-dependent properties like positions, velocities, kinetic energies, and thermodynamic variables. |
| **`MolecularCrystal`** | Periodic crystal structure with additional topological information about its constituent molecules (e.g., connectivity, centers of mass, molecule indices). Serves as an intermediate necessary for finite cluster extraction. |
| **`FiniteSubsystem`** | Finite clusters of molecules extracted from a periodic structure or trajectory. Used to generate training data for fragment-based methods. |
| **`Dataset`** | A container class that holds a collection of `Structure` or `FiniteSubsystem` objects. Aggregates data for machine learning training sets. |
| **`AtomicReference`** | Isolated atom energies required to generate reference energy for machine-learning interatomic potentials. Can store data at multiple levels of theory. |

## Methods Summary

The following table summarizes the key methods available across these classes.

| Method | Description | Available In |
| :--- | :--- | :--- |
| **`read`** | Method to load the object from an HDF5 dataset. | `AtomicReference`, `ForceConstants`, `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`save`** | Saves the object to an HDF5 dataset. | `AtomicReference`, `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`subsample`** | Selects a representative subset of frames (e.g., using Farthest Point Sampling or k-means on feature vectors). | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`run`** | Executes a calculator on fixed structures. Computed energies and forces are stored in `ground_truth` (indexed by the calculator's `level_of_theory`), while feature vectors are stored directly on the structure for subsampling. | `AtomicReference`, `Structure`, `Trajectory`, `FiniteSubsystem` |
| **`to_mace_dataset`** | Exports the data (structures, energies, forces) to MACE-compatible XYZ files for model training. | `Structure`, `FiniteSubsystem`, `Dataset` |
| **`random_split`** | Randomly splits the frames into multiple objects (e.g., for creating training and validation sets). | `Structure`, `FiniteSubsystem` |
| **`to_ase_atoms`** | Converts a specific frame into an `ase.Atoms` object. | `Structure` |
| **`extract_..._molecules`** | Extracts isolated molecules (all, unique, or relaxed unique) from a periodic structure. | `Structure` |
| **`extract_finite_subsystems`** | Extracts finite clusters of molecules (e.g., dimers, trimers) based on distance or number criteria. | `MolecularCrystal` |
| **`frequencies_and_...`** | Calculates phonon frequencies and eigenvectors at a given k-point. | `ForceConstants` |
| **`append`** | Adds a structure or subsystem to the dataset collection. | `Dataset` |
| **`unique_elements`** | Returns a sorted array of unique atomic numbers present in the object. | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem`, `Dataset` |
| **`atomic_reference`** | Calculates ground-state energies for all unique isolated atoms (used for generating MLIP baselines). | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem`, `Dataset` |

### Usage Notes
*   **`subsample`**: Requires feature vectors to be present (computed via `run` or during an MD simulation) to calculate distances in chemical space.
*   **`run`**: Can be used to evaluate ground truth (energies and forces) or to generate descriptors (feature vectors) for subsampling.
