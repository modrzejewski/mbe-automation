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

The program integrates several scientific libraries into a unified workflow, including:

*   **MACE (Multi-ACE):** As the primary MLIP model for calculating energies and forces.
*   **phonopy:** For phonon calculations and the analysis of vibrational properties.
*   **pymatgen:** For crystal structure analysis and manipulation.
*   **ASE:** For geometry relaxation and molecular dynamics simulations.
*   Quantum-chemical programs for correlated electronic structure calculations.

## Installation

For instructions on how to install the program, please see the [Installation Guide](./00_installation.md).

## Workflows

The program is organized into three main workflows:

1.  [Quasi-Harmonic Calculation](./01_quasi_harmonic.md)
2.  [Molecular Dynamics](./02_molecular_dynamics.md)
3.  [Training Set Creation](./03_training_set.md)

For a detailed discussion of performance considerations, see the [Computational Bottlenecks](./06_bottlenecks.md) section.
