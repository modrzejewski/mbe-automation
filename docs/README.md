# mbe-automation Documentation

`mbe-automation` is a program for modeling the thermodynamic properties of molecular crystals using machine-learning interatomic potentials (MLIPs). It provides automated, "black-box" workflows for complex computational tasks that are essential in the study of molecular crystals.

The general computational scheme facilitated by this program is an automation of the following process applied to molecular crystals:

1.  Generate training set configurations for a periodic system using a baseline MLIP (e.g., MACE).
2.  Extract finite subsystems composed of one or more molecules.
3.  Use these finite subsystems to generate high-accuracy data points using quantum chemical wave function methods.
4.  Perform training of a delta-learning layer on top of the baseline model.
5.  Carry out the final calculation (e.g., quasi-harmonic thermodynamics or molecular dynamics) using the newly trained model.

The program integrates several powerful scientific libraries into a unified workflow, including:

*   **ASE (Atomic Simulation Environment):** For handling atomic structures and running simulations.
*   **MACE (Multi-ACE):** As the primary MLIP model for calculating energies and forces.
*   **phonopy:** For phonon calculations and the analysis of vibrational properties.
*   **pymatgen:** For crystal structure analysis and manipulation.

## Installation

For instructions on how to install the program, please see the [Installation Guide](./00_installation.md).

## Workflows

The program is organized into three main workflows:

1.  [Quasi-Harmonic Calculation](./01_quasi_harmonic.md)
2.  [Molecular Dynamics](./02_molecular_dynamics.md)
3.  [Training Set Creation](./03_training_set.md)
