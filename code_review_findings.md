# Code Review Findings

## Summary of New Features

The recent commit introduces the functionality to perform quasi-harmonic crystal modeling under external pressure. This allows for the calculation of thermodynamic properties of molecular crystals at pressures other than ambient, which is a significant enhancement to the modeling capabilities. The `external_pressure_GPa` parameter has been added to the `FreeEnergy` configuration class to control this feature.

## Identified Issues

| File Path                                                 | Line Number | Description of the Issue                                                                                                                              | Severity Level |
| --------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `src/mbe_automation/dynamics/harmonic/core.py`            | 403-407     | The code was minimizing the Helmholtz Free Energy (F) instead of the Gibbs Free Energy (G = F + PV), completely ignoring the external pressure term.         | Critical       |
| `src/mbe_automation/dynamics/harmonic/core.py`            | 405         | The variable `G_tot_crystal` was misleadingly named, as it actually contained the Helmholtz Free Energy. This could lead to confusion and future bugs. | Medium         |
| `src/mbe_automation/workflows/quasi_harmonic.py`          | 251-252     | When `eos_sampling` was set to "volume" or "uniform_scaling", the pressure for the geometry optimizer was incorrectly reset to zero, ignoring external pressure. | High           |
| `src/mbe_automation/workflows/quasi_harmonic.py`          | 329-331     | The code would have crashed with a `NameError` due to a typo (`F_tot_diff` instead of `G_tot_diff`) in the final accuracy check.                     | Critical       |
