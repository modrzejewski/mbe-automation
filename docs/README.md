# mbe-automation Documentation

`mbe-automation` is a program for modeling the thermodynamic properties of molecular crystals using machine-learning interatomic potentials (MLIPs). It provides automated, "black-box" workflows for complex computational tasks that are essential in the study of molecular crystals.

The program integrates several powerful scientific libraries into a unified workflow, including:

*   **ASE (Atomic Simulation Environment):** For handling atomic structures and running simulations.
*   **MACE (Multi-ACE):** As the primary MLIP model for calculating energies and forces.
*   **phonopy:** For phonon calculations and the analysis of vibrational properties.

## Workflows

The program is organized into three main workflows, each designed for a specific computational task:

1.  [Quasi-Harmonic Calculation](./01_quasi_harmonic.md): For calculating thermodynamic properties like free energy and heat capacity.
2.  [Molecular Dynamics](./02_molecular_dynamics.md): For running MD simulations to compute properties such as sublimation energy.
3.  [Training Set Creation](./03_training_set.md): For generating a diverse set of configurations for delta-learning an MLIP.
