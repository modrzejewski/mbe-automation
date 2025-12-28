```
+------------------------------------------------------------+
| This is experimental code with lots of bugs intended for   |
| internal use only. Will be ready as soon as we publish the |
| methodology.                                               |
+------------------------------------------------------------+
```

# `mbe-automation`

Automate the modeling of thermodynamic properties in organic molecular crystals using machine-learning interatomic potentials (MLIPs). 
The general computational scheme facilitated by this program involves the following steps:

```
+-------------------------------------------------------------+
| Generate training set configurations for a periodic system  |
| using a baseline MLIP (e.g., MACE).                         |
+--------------------------+----------------------------------+
                           |
                           v
+--------------------------+----------------------------------+
| Extract finite subsystems composed of one or more molecules.|
+--------------------------+----------------------------------+
                           |
                           v
+--------------------------+----------------------------------+
| Use these finite subsystems to generate high-accuracy data  |
| points using quantum chemical wave function methods.        |
+--------------------------+----------------------------------+
                           |
                           v
+--------------------------+----------------------------------+
| Perform training of a delta-learning layer on top of the    |
| baseline model.                                             |
+--------------------------+----------------------------------+
                           |
                           v
+--------------------------+----------------------------------+
| Carry out the final calculation (e.g., quasi-harmonic       |
| thermodynamics or molecular dynamics) using the newly       |
| trained model.                                              |
+-------------------------------------------------------------+
```

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

*   [API Data Classes](./10_api_classes.md)
*   [Calculators](./11_calculators.md)
*   [Working with HDF5 Datasets](./09_working_with_hdf5_datasets.md)
*   [Computational Bottlenecks](./06_bottlenecks.md)

### Workflows

*   [Quasi-Harmonic Calculation](./01_quasi_harmonic.md)
*   [Molecular Dynamics](./02_molecular_dynamics.md)
*   [MD sampling and phonon sampling](./03_training_set.md)

### Cookbooks

*   [Semi-empirical MD + MACE Features](./04_cookbook_dftb_mace.md)
*   [Extracting frequencies and eigenvectors of the dynamical matrix](./05_frequencies_eigenvectors.md)
*   [Training Set from MACE MD + DFTB Energies](./07_cookbook_mace_md_dftb_energies.md)
*   [Delta Learning Dataset Creation](./08_cookbook_delta_learning_dataset.md)
