```
+------------------------------------------------------------+
| This is experimental code with lots of bugs intended for   |
| internal use only. Will be ready as soon as we publish the |
| methodology.                                               |
+------------------------------------------------------------+
```

# `mbe-automation`

Automate the modeling of thermodynamic properties in organic molecular crystals using machine-learning interatomic potentials (MLIPs). 

The program integrates several scientific codes into a unified workflow, including:

*   **MACE:** Primary MLIP model for energies and forces.
*   **phonopy:** Phonon calculations and vibrational properties.
*   **pymatgen:** Crystal structure analysis and manipulation.
*   **ASE:** Geometry relaxation and molecular dynamics simulations.
*   **PySCF** and **GPU4PySCF:** Mean-field electronic structure calculations.
*   **MRCC** and **beyond-rpa**: High-fidelity data points from correlated wave-function theory.

## Table of Contents

### Setup

*   [Installation Guide](./00_installation.md)

### Basics

*   [API Data Classes](./01_api_classes.md)
*   [Calculators](./02_calculators.md)
*   [Configuration Classes](./03_configuration_classes.md)
*   [Working with HDF5 Datasets](./04_working_with_hdf5_datasets.md)
*   [Computational Bottlenecks](./05_bottlenecks.md)

### Workflows

*   [Quasi-Harmonic Calculation](./06_quasi_harmonic.md)
*   [Molecular Dynamics](./07_molecular_dynamics.md)
*   [MD sampling and phonon sampling](./08_training_set.md)

### Cookbooks

*   [Extracting frequencies and eigenvectors of the dynamical matrix](./09_cookbook_frequencies_eigenvectors.md)
*   [Training Set from MACE MD + r2SCAN Energies & Forces](./10_cookbook_mace_md_dftb_energies.md)
*   [Delta Learning Dataset Creation](./11_cookbook_delta_learning_dataset.md)
*   [Adding Atomic Reference Energies to Existing MACE Training Data](./12_cookbook_export_with_atomic_energies.md)
