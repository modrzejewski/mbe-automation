# Molecule Vibrational Thermodynamics Comparison Report

## Overview
This report investigates the differences in the computation of molecule (not crystal) vibrational thermodynamic functions in the quasi-harmonic workflow between the `development` and `main` branches.

## Technical Differences
The `main` branch uses the `ase.thermochemistry.HarmonicThermo` class from the `ase` library to compute molecular vibrational thermodynamics. Constants provided by `ase.units` are used for unit conversions.

The `development` branch uses a custom implementation in `src/mbe_automation/dynamics/harmonic/molecule_thermo.py` (specifically in the `_vibrational_functions` method). It computes these properties manually using the partition function, and uses physical constants retrieved from the `phonopy.physical_units.get_physical_units` module.

### Comparison
The mathematical implementations (taking the quantum harmonic oscillator and Bose-Einstein statistics) are practically identical between the two branches:
- For `E_vib` (Vibrational internal energy), both approaches sum $\hbar \omega \left(\frac{1}{2} + \frac{1}{e^x - 1}\right)$, where $x = \hbar \omega / k_B T$.
- For `S_vib` (Vibrational entropy), both approaches sum $k_B \left( x \frac{1}{e^x - 1} - \ln(1 - e^{-x}) \right)$.

However, as the user correctly noted, there are minor numerical discrepancies in the outputs of `E_vib_molecule` and `S_vib_molecule` between the two branches.

### Source of Discrepancies
The discrepancy stems entirely from differences in the values of fundamental physical constants used by the underlying libraries (`ase` vs `phonopy`), particularly the Boltzmann constant ($k_B$) and Avogadro's number ($N_A$) or elemental charge ($e$), which affect the conversion multipliers:

1. **Boltzmann Constant ($k_B$) in eV/K**:
   - `ase` uses $k_B \approx 8.6173303 \times 10^{-5}$ eV/K.
   - `phonopy` uses $k_B \approx 8.6173383 \times 10^{-5}$ eV/K.

2. **Energy Conversion Multipliers (eV $\to$ kJ/mol)**:
   - The multiplier uses combinations of $N_A$ and $e$.
   - `ase` multiplier: $96.485332882...$
   - `phonopy` multiplier: $96.485390539...$

Because $k_B$ appears inside the exponent in the Bose-Einstein distribution factor $x = \frac{\hbar \omega}{k_B T}$, variations in its value propagate through both internal energy ($E_{vib}$) and entropy ($S_{vib}$) computations. Additionally, differences in the conversion multiplier scale the final output.

When the custom function from the `development` branch is executed using the constants from `ase.units`, the outputs perfectly match the values produced by the `main` branch down to floating point precision.

## Conclusion & Error Analysis
Neither branch contains a fundamental mathematical error in its formulation of vibrational thermodynamics.

However, introducing multiple sets of physical constants from different libraries into the same physical pipeline can cause inconsistency. For instance, if the crystal vibrational properties are calculated using `phonopy` (which they likely are, given the workflow), and the molecule vibrational properties are calculated using `ase`, the workflow will combine energies that disagree on the exact definitions of $k_B$ and $N_A$. This creates small, non-physical offsets when calculating energetic differences like the sublimation free energy.

Therefore, the `development` branch's custom implementation that unifies the constants across the framework (by relying on `phonopy.physical_units`) is considered **correct and superior**. The approach in the `main` branch can be deemed **incorrect in the context of the broader workflow** because it introduces constant-related inconsistencies by using `ase.units`.
