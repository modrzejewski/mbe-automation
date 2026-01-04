# Code Review Findings

## Summary
The review of the thermal expansion code in the quasi-harmonic workflow revealed three critical issues that will cause runtime crashes or data corruption.

## Findings Table

| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/workflows/quasi_harmonic.py` | 246 | **Syntax Error / Typo**: The module path `mbe_automation.dynamics.haronic.eos` contains a typo. It should be `mbe_automation.dynamics.harmonic.eos`. This will cause an `AttributeError` or `ImportError`. | Critical |
| `src/mbe_automation/workflows/quasi_harmonic.py` | 258, 264 | **Logic Error / Data Corruption**: The `df_thermal_expansion` DataFrame returned by `fit_thermal_expansion_properties` uses a default `RangeIndex` (0, 1, ...), losing the original temperature indices. When concatenated with `df_crystal_eos` (which preserves the full index) or other DataFrames, rows will be misaligned if any temperature points were filtered out (e.g., due to optimization failure). This results in associating thermal expansion properties with the wrong temperatures. | Critical |
| `src/mbe_automation/dynamics/harmonic/core.py` | 144 | **Runtime Error**: The `phonons` function passes `interp_mesh` directly to `phonons.run_mesh`. If `interp_mesh` is a float (the default `150.0`), Phonopy will crash because it expects a list of 3 integers. Logic to convert the mesh density to a grid is missing. | Critical |
