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
| **`from_xyz_file`** | Creates a structure object from an XYZ file. | `Structure` |
| **`from_atomic_numbers`** | Creates an `AtomicReference` from a list of atomic numbers and a calculator. | `AtomicReference` |
| **`subsample`** | Selects a representative subset of frames (e.g., using Farthest Point Sampling or k-means on feature vectors). | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` |
| **`select`** | Returns a new object containing only the specified frames (by index). | `Structure` |
| **`run`** | Executes a calculator on fixed structures. Computed energies and forces are stored in `ground_truth` (indexed by the calculator's `level_of_theory`), while feature vectors are stored directly on the structure for subsampling. | `Structure`, `Trajectory`, `FiniteSubsystem` |
| **`to_mace_dataset`** | Exports the data (structures, energies, forces) to MACE-compatible XYZ files for model training. | `Structure`, `Trajectory`, `FiniteSubsystem`, `Dataset` |
| **`random_split`** | Randomly splits the frames into multiple objects (e.g., for creating training and validation sets). | `Structure`, `FiniteSubsystem` |
| **`to_molecular_crystal`** | Converts a periodic structure into a `MolecularCrystal` by detecting connected molecules. | `Structure` |
| **`to_ase_atoms`** | Converts a specific frame into an `ase.Atoms` object. | `Structure`, `Trajectory` |
| **`to_pymatgen`** | Converts the structure (or a frame) to a Pymatgen object. | `Structure`, `Trajectory` |
| **`lattice`** | Returns the Pymatgen lattice object for a given frame. | `Structure`, `Trajectory` |
| **`extract_all_molecules`** | Extracts all molecules from a periodic structure. | `Structure` |
| **`extract_unique_molecules`** | Extracts symmetry-unique molecules from a periodic structure. | `Structure` |
| **`extract_relaxed_unique_molecules`** | Extracts and relaxes symmetry-unique molecules from a periodic structure. | `Structure` |
| **`extract_finite_subsystems`** | Extracts finite clusters of molecules (e.g., dimers, trimers) based on distance or number of molecules. | `MolecularCrystal` |
| **`positions`** | Returns positions of specific molecules in the crystal. | `MolecularCrystal` |
| **`atomic_numbers`** | Returns atomic numbers of specific molecules in the crystal. | `MolecularCrystal` |
| **`frequencies_and_eigenvectors`** | Calculates phonon frequencies and eigenvectors at a given k-point. | `ForceConstants` |
| **`append`** | Adds a structure or subsystem to the dataset collection. | `Dataset` |
| **`statistics`** | Prints statistical summaries of the dataset (e.g. mean/std of energies). | `Dataset` |
| **`display`** | Visualizes properties of the object (e.g. energy fluctuations). | `Trajectory` |
| **`available_energies`** | Lists methods (levels of theory) for which energies are available. | `Structure`, `Trajectory` |
| **`available_forces`** | Lists methods (levels of theory) for which forces are available. | `Structure`, `Trajectory` |
| **`energies_at_level_of_theory`** | Returns energies at a given level of theory. | `Structure`, `Trajectory` |
| **`forces_at_level_of_theory`** | Returns forces at a given level of theory. | `Structure`, `Trajectory` |
| **`unique_elements`** | Returns a sorted array of unique atomic numbers present in the object. | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem`, `Dataset` |
| **`atomic_reference`** | Calculates ground-state energies for all unique isolated atoms (used for generating MLIP baselines). | `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem`, `Dataset` |
| **`levels_of_theory`** | Lists the levels of theory available in the atomic reference. | `AtomicReference` |

### Usage Notes
*   **`subsample`**: Requires feature vectors to be present (computed via `run` or during an MD simulation) to calculate distances in chemical space.
*   **`run`**: Can be used to evaluate ground truth (energies and forces) or to generate descriptors (feature vectors) for subsampling.
