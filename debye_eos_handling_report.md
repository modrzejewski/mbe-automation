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
            (df_crystal_eos["V_debye (Å³∕unit cell)"] > V_lo) &
            (df_crystal_eos["V_debye (Å³∕unit cell)"] < V_hi)
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

This error occurred at line 285 in `quasi_harmonic.py`:
```python
(filtered_df["V_debye (Å³∕unit cell)"] > V_lo) &
```

**Is this error still possible on the current branch?**
**No, this specific `KeyError` is no longer possible for `V_debye (Å³∕unit cell)`.**

Why?
In `mbe_automation/dynamics/harmonic/core.py`, the `V_debye (Å³∕unit cell)` column is *only* added to `df` if the Debye model is successfully initialized:
```python
    if debye_model.initialized:
        V_debye, alpha_V_debye = debye_model.predict(temperatures)
        df["V_debye (Å³∕unit cell)"] = V_debye
```

If the Debye model fails to initialize (e.g., `len(df_fit) < 3` because too many points were removed as low quality), `V_debye (Å³∕unit cell)` is never added. Notice the character `Å` (U+212B) vs `Å` (U+00C5) in the column name might have been a previous issue, but the main protection is the fallback.

In `quasi_harmonic.py`, there is a fallback mechanism:
```python
    effective_volume_curve = config.volume_curve
    if config.volume_curve == "debye":
        if not interpolated_harmonic_props.debye_model.initialized:
            print(
                "WARNING: volume_curve='debye' was requested but the Debye model "
                "could not be fitted (insufficient low-T data). "
                "Falling back to volume_curve='eos_minimum'."
            )
            effective_volume_curve = "eos_minimum"
```
Because of this fallback, if the Debye model is not initialized, `effective_volume_curve` changes to `"eos_minimum"`. Consequently, the block `if effective_volume_curve == "debye":` is **skipped**, and the code evaluating `df_crystal_eos["V_debye (Å³∕unit cell)"]` is never reached.

*Wait! Look closely at the characters!*
In `mbe_automation/dynamics/harmonic/core.py` (line 749):
```python
        df["V_debye (Å³∕unit cell)"] = V_debye
```
Notice the Angstrom sign is `Å` (U+212B, Angstrom sign).

In `mbe_automation/workflows/quasi_harmonic.py` (line 284):
```python
            (df_crystal_eos["V_debye (Å³∕unit cell)"] > V_lo) &
```
Notice the Angstrom sign is `Å` (U+00C5, Latin Capital Letter A with ring above).

**THIS IS THE EXACT CAUSE OF THE KEYERROR!** Even if `debye_model` is initialized and `V_debye (Å³∕unit cell)` is added to the dataframe, the workflow script tries to access `V_debye (Å³∕unit cell)` with the wrong unicode character, causing the `KeyError` regardless of initialization status.

**Yes, the error is definitely still possible on this branch!**

## Findings Summary
1. **Debye Fitting is Robust:** The actual Debye fit correctly excludes temperatures where EOS fitting failed or had insufficient high-quality data points.
2. **Unicode Mismatch Bug:** The `KeyError: 'V_debye (Å³∕unit cell)'` is caused by a subtle unicode character mismatch. `core.py` sets the column using `Å` (U+212B), but `quasi_harmonic.py` attempts to read it using `Å` (U+00C5).
3. **`p_thermal` safely dropped:** `p_thermal` is safely dropped for the Debye curve.
4. **Crashing Dependencies:** The cubic spline for `S_vib(V)` will still crash if fewer than 3 high-quality volume points are available at that temperature, because `valid_equilibrium` does not check `min_found` for `debye`.

## Recommendations
The `debye` option provides a smoother volume curve, especially at low temperatures.

**Fix Unicode Bug:** Modify `quasi_harmonic.py` to use the correct unicode character (U+212B) when accessing the `V_debye` column.

Change lines 284-285 in `src/mbe_automation/workflows/quasi_harmonic.py`:
```python
        in_range = (
            (df_crystal_eos["V_debye (Å³∕unit cell)"] > V_lo) &
            (df_crystal_eos["V_debye (Å³∕unit cell)"] < V_hi)
        )
```
And around line 292:
```python
            out_vols  = df_crystal_eos.loc[~in_range, "V_debye (Å³∕unit cell)"].tolist()
```

**Fix S_vib Spline Issue:** To fully protect the Debye evaluation loop from bad data, ensure that any temperature processed has at least 3 valid points in `exact_at_sampled_volume` for the `S_vib` spline.
