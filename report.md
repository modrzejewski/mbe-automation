# Quasi-Harmonic Workflow Failed-Temperature Handling Report

## Observations

1. **Intended NaN Behavior vs. Crashing:** In the quasi-harmonic workflow, it's expected that for some temperatures, an equilibrium volume might not be successfully predicted. When this happens, values for volume and related thermal properties (like `V_crystal`, `H_tot_crystal`, `C_V_vib_crystal`) are populated with NaNs when evaluating thermal expansion and sublimation properties. However, code in `data.py` and `crystal_thermo.py` is not robust against handling these NaNs, which causes crashes downstream.
2. **Type Conversions on NaNs (`data.py`):** The logic accessing single-value attributes from `df_crystal` and `df_molecule` using `.iloc[0]` fails when encountering `np.nan` values because it attempts to explicitly cast them to integers (`int()`). Specifically, fields like `n_atoms_primitive_cell`, `n_atoms_conventional_cell`, and `n_atoms_molecule` crash with a `ValueError` when they return NaNs. Also, checking properties like `unit_cell_type` via `.nunique() == 1` might fail unexpectedly when the subset evaluated to empty. Furthermore, calculations like `beta = n_atoms_formula_unit / n_atoms_unit_cell` and `Z = n_atoms_unit_cell // n_atoms_molecule` are not properly protected against NaN operands.
3. **Differentiation with NaNs (`crystal_thermo.py`):** The function `fit_thermal_expansion_properties` uses numerical differentiation (`np.gradient` and `CubicSpline.derivative`). The `scipy.interpolate.CubicSpline` implementation throws a `ValueError: y must contain only finite values` if any `np.nan` is present in the `y` array, failing the thermal expansion calculation for the entire array.

## Proposed Suggestions

1. **Graceful Handling of Empty/NaN Slices in `data.py`:**
   - Instead of forcefully indexing `[0]` on the columns, `df[col].dropna()` should be applied first.
   - Conditional logic like `if not df[col].dropna().empty` should be added to handle cases where a variable is entirely empty due to `NaN`s, explicitly substituting `np.nan` (or `None`/`0` where applicable) to prevent integer-casting errors.
   - The scalar formulas depending on unit cell divisibility (`Z = n_atoms_unit_cell // n_atoms_molecule`) should conditionally evaluate to avoid runtime errors on `NaN` division, appropriately propagating `NaN` down into resulting arrays.
2. **Filtering Out NaNs for Numerical Differentiation in `crystal_thermo.py`:**
   - Update `fit_thermal_expansion_properties` to `dropna` early using a subset of all necessary thermodynamic variable columns.
   - This ensures that finite differences and `CubicSpline` only process dense, finite numbers.
   - The returned dataframe should map cleanly back to the original index `index=df_valid.index`, successfully relying on the later `df_thermal_expansion.reindex(df_crystal_eos.index)` in `quasi_harmonic.py` to restore the proper gaps/NaNs for invalid temperatures in the top-level QHA loop.

## Code Rating

- **Architecture:** The separation of concerns between `quasi_harmonic.py`, `crystal_thermo.py`, and `data.py` is quite good and logically cohesive. Using pandas dataframes for temperatures is standard and scales well.
- **Robustness:** Currently, the edge-case handling for missing EOS points is brittle. The workflow was explicitly designed to pad failed points with `NaN`s (via `reindex`), but downstream algorithms fundamentally assume dense validity (int conversions and splines).
- **Test Coverage:** Existing tests in the `tests/` directory are helpful. However, testing the edge cases of QHA workflow logic (where properties contain isolated NaNs) is currently lacking and would be a highly valuable addition.

Overall Rating: **7/10** (Robustness improvements would greatly stabilize the pipeline).
