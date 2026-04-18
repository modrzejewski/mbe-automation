# Quasi-Harmonic Workflow Data Handling Report: Debye Model Fitting

## Overview
This report details how data points are handled within the quasi-harmonic workflow when the user requests Debye model fitting for unit cell volumes (`volume_curve='debye'`), and specifically how the code behaves when some equation of state (EOS) points are removed as "low quality" (due to imaginary modes or other filtering criteria).

## Data Flow & Filtering in `mbe_automation/dynamics/harmonic/core.py`

1. **Initial Point Filtering (`equilibrium_curve` function)**
   Data from different volume points are gathered into a single DataFrame `df_eos`. A boolean mask `good_points` is created based on criteria requested by the user:
   - `filter_out_imaginary_acoustic`
   - `filter_out_imaginary_optical`
   - `filter_out_broken_symmetry`

2. **EOS Fitting per Temperature**
   For each temperature `T`, the code extracts the data for `good_points` corresponding to that temperature. It then attempts to fit the equation of state (e.g., polynomial, spline, Birch-Murnaghan, Vinet) to the `G` vs `V` data.
   If the minimum is found (`fit.min_found`), it calculates the equilibrium volume `V_min` and `G_min`. These are stored in `V_eos[i]`, `G_tot_eos[i]`, and the `min_found` array is updated to `True` for index `i`.

3. **Debye Model Fit (`_fit_debye_model` function)**
   After generating the `df` DataFrame with columns like `"T (K)"`, `"V_eos (Å³∕unit cell)"`, and `"min_found"`, the code attempts to fit the Debye model:
   ```python
   def _fit_debye_model(
       df: pd.DataFrame,
       debye_model: mbe_automation.dynamics.harmonic.eec.DebyeModel,
       filter_out_extrapolated_minimum: bool,
   ):
       df_fit = df[
           (df["T (K)"] <= debye_model.max_fit_temperature_K) &
           (df["min_found"])
       ]
       # ...
       if len(df_fit) >= 3:
           debye_model.fit(...)
   ```
   **Observation:** The Debye model is only fitted using temperatures where an EOS minimum was successfully found (`df["min_found"] == True`). If low-quality data points at a specific temperature prevent the EOS from finding a minimum, that temperature is correctly excluded from the Debye model fit.

4. **Debye Model Prediction**
   If the Debye model is successfully fitted (i.e., `debye_model.initialized` is True), it predicts the volume `V_debye` for *all* temperatures.

## Downstream Handling in `mbe_automation/workflows/quasi_harmonic.py`

1. **Fallback Mechanism**
   If the Debye model could not be fitted (e.g., because too many low-T EOS points lacked a minimum, resulting in `len(df_fit) < 3`), the workflow gracefully falls back to `eos_minimum`.

2. **Temperature Filtering for Valid Equilibrium Volumes**
   The code marks a column `valid_equilibrium`. For the Debye curve, a temperature is considered valid when `V_debye` lies strictly inside the sampled volume range `(V_lo, V_hi)`.
   ```python
    if effective_volume_curve == "debye":
        in_range = (
            (df_crystal_eos["V_debye (Å³∕unit cell)"] > V_lo) &
            (df_crystal_eos["V_debye (Å³∕unit cell)"] < V_hi)
        )
        df_crystal_eos["valid_equilibrium"] = in_range
        df_crystal_eos.drop(columns=["p_thermal_crystal (GPa)"], inplace=True)
   ```
   If `debye` is used, all temperatures are included as long as their predicted Debye volume is within the sampled range. Crucially, this drops the `p_thermal_crystal (GPa)` column completely since it was computed at the EOS minimum volume, which is not the Debye equilibrium volume.

## Missing Data Issue / Bug Evaluation
The user previously experienced an issue where the following KeyError occurred:
```
KeyError: 'V_debye (Å³∕unit cell)'
```

**Is this error still possible on the current branch?**
**No, this issue has been resolved in the latest commits.**

**Explanation of the Fix:**
The original `KeyError` was caused by a subtle unicode character mismatch between how the column was written in `core.py` and how it was read in `quasi_harmonic.py`:
- `core.py` (line 749) used `Å` (U+212B, Angstrom sign).
- `quasi_harmonic.py` (line 284) attempted to read using `Å` (U+00C5, Latin Capital Letter A with ring above).

In the latest commit to the `EECv2` branch, this has been corrected. `quasi_harmonic.py` now correctly accesses the column using the exact same unicode character `Å` (U+212B) used during assignment in `core.py`:
```python
        in_range = (
            (df_crystal_eos["V_debye (Å³∕unit cell)"] > V_lo) &
            (df_crystal_eos["V_debye (Å³∕unit cell)"] < V_hi)
        )
```
Since the column names now perfectly match, the `KeyError` will no longer occur. Additionally, the fallback mechanism ensures that if `debye_model` is not initialized (and thus the column is not added), the loop falls back to `eos_minimum` and safely avoids calling the column entirely.

## Findings Summary
1. **Debye Fitting is Robust:** The actual Debye fit correctly excludes temperatures where EOS fitting failed or had insufficient high-quality data points.
2. **Unicode Mismatch Bug Fixed:** The `KeyError: 'V_debye (Å³∕unit cell)'` caused by a subtle unicode character mismatch (U+212B vs U+00C5) has been successfully resolved.
3. **`p_thermal` safely dropped:** `p_thermal` is safely dropped for the Debye curve to prevent erroneous effective pressure evaluation.

## Final Recommendations
The `debye` option now correctly provides a smoother volume curve at low temperatures, properly handling cases where some data points are excluded as low quality.

**Minor Precaution on S_vib Spline:**
To be completely rigorous, the cubic spline evaluation `interpolated_harmonic_props.S_vib_at_T(T, derivative=True)` requires at least 3 valid volume points at a given temperature. If a specific temperature was completely excluded from the EOS fit due to imaginary modes (meaning it has `< 3` valid points), but its Debye-predicted volume randomly falls inside `[V_lo, V_hi]`, the `debye` path might still attempt to evaluate the spline and crash. It may be wise to assert `n_volumes >= 3` gracefully or enforce that `valid_equilibrium` also requires sufficient points for the `S_vib` spline.
