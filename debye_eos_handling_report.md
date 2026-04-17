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
   For each temperature `T`, the code extracts the data for `good_points` corresponding to that temperature using the mask `good_points & select_T[i]`. It then attempts to fit the equation of state (e.g., polynomial, spline, Birch-Murnaghan, Vinet) to the `G` vs `V` data.

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
   If the Debye model could not be fitted (e.g., because too many low-T EOS points lacked a minimum, resulting in `len(df_fit) < 3`), the workflow gracefully falls back to `eos_minimum`:
   ```python
   if config.volume_curve == "debye":
       if not interpolated_harmonic_props.debye_model.initialized:
           print("WARNING: volume_curve='debye' was requested but the Debye model could not be fitted...")
           effective_volume_curve = "eos_minimum"
   ```

2. **Temperature Filtering for `data_frames_at_T` Construction**
   This is where a critical difference occurs between `eos_minimum` and `debye`:
   ```python
   if effective_volume_curve == "debye":
       filtered_df = df_crystal_eos # Includes all temperatures
   else: # "eos_minimum"
       filtered_df = df_crystal_eos[df_crystal_eos["min_found"]] # Excludes T without minimum
   ```
   If `debye` is used, **all temperatures are initially included** in `filtered_df`, regardless of whether `min_found` is True for that temperature.

3. **Trust Region / Range Check Filtering**
   For `debye`, a subsequent check filters out temperatures where the predicted Debye volume `V_debye` falls outside the interpolation range of the sampled volumes `[V_lo, V_hi]`.
   ```python
   if effective_volume_curve == "debye":
       V_lo = interpolated_harmonic_props.sampled_volumes.min()
       V_hi = interpolated_harmonic_props.sampled_volumes.max()
       in_range = (
           (filtered_df["V_debye (Å³∕unit cell)"] > V_lo) &
           (filtered_df["V_debye (Å³∕unit cell)"] < V_hi)
       )
       # Warning is printed and out-of-range rows are skipped
       filtered_df = filtered_df[in_range]
   ```

## Missing Data Issue / Bug in Workflow
There is a fundamental issue with how the `p_thermal` and thermodynamic properties are handled when `effective_volume_curve == "debye"`.

In the `for i, row in filtered_df.iterrows():` loop, the code uses `row["p_thermal_crystal (GPa)"]`.

However, if an EOS fit failed (or was excluded because of imaginary modes leading to `< 3` good points) for a specific temperature `T`, `min_found` for that `T` is `False`.

When `min_found` is `False`, `p_thermal_eos[i]` is left as `np.nan` (because it is initialized with `np.full(n_temperatures, np.nan)` and only updated `if min_found[i]:`).

Because `filtered_df` for `debye` includes rows where `min_found` might be `False`, the loop will attempt to use `np.nan` for `row["p_thermal_crystal (GPa)"]`:
```python
p_effective = (
    row["p_thermal_crystal (GPa)"] +
    config.pressure_GPa
)
```
This will result in an effective pressure of `NaN`, which is then passed to the optimizer: `optimizer._pressure_GPa = p_effective`. The relaxation will crash or silently propagate `NaN` coordinates.

Additionally, to compute `dSdV_vib_crystal`, the code retrieves `interpolated_harmonic_props.S_vib_at_T(T, derivative=True)`. This function internally fetches `df_T = self.exact_at_sampled_volume[mask]`. If points were filtered out due to imaginary modes, this dataframe might have fewer than 3 valid points. In fact, `S_vib_at_T` explicitly throws an error:
```python
if n_volumes < 3:
    raise ValueError(f"Cannot fit S_vib(V) at T={temperature_K} K. Need at least 3 volumes...")
```

## Findings Summary
1. **Debye Fitting is Robust:** The actual Debye fit correctly excludes temperatures where EOS fitting failed or had insufficient high-quality data points.
2. **Debye Evaluation Injects Invalid Data:** The quasi-harmonic workflow attempts to evaluate structural relaxations and thermodynamic properties for *all* temperatures when `volume_curve='debye'`, even those where the original EOS minimum could not be found due to low-quality points.
3. **Crashing Dependencies:** For temperatures where `min_found` is False, `p_thermal` is `NaN`, causing relaxation with `NaN` pressure. Further, the cubic spline for `S_vib(V)` will crash if fewer than 3 high-quality volume points are available at that temperature.

## Recommendations
The `debye` option is designed to provide a smoother volume curve, especially at low temperatures. However, it still requires the underlying thermal pressure (`p_thermal`) and entropy derivatives (`dS/dV`) to be well-defined at the evaluation temperatures to proceed with the constrained relaxations and property evaluations.

If `p_thermal` is `NaN` because the EOS could not find a minimum (e.g., due to filtering imaginary modes), it is physically impossible to apply the "effective pressure" relaxation technique used in `eos_sampling='pressure'`.

**Fix:** The code should filter out temperatures where `min_found` is `False` *even* when using the `debye` volume curve, OR the algorithm needs to be modified to interpolate/extrapolate `p_thermal` and `dS/dV` using the valid temperatures. Given the current architecture, simply requiring `min_found` is the safest and most consistent approach.

Modify `src/mbe_automation/workflows/quasi_harmonic.py` around line 274:

```python
    if effective_volume_curve == "debye":
        # CHANGE: Even for debye, we can only proceed if we successfully found a minimum
        # because we rely on p_thermal and S_vib splines which are only valid when min_found=True.
        filtered_df = df_crystal_eos[df_crystal_eos["min_found"]]
    elif config.filter_out_extrapolated_minimum:
        filtered_df = df_crystal_eos[df_crystal_eos["min_found"] & (df_crystal_eos["min_extrapolated"] == False)]
    else:
        filtered_df = df_crystal_eos[df_crystal_eos["min_found"]]
```
This ensures that only valid temperatures are passed to the relaxation and property evaluation loop, preventing `NaN` pressure crashes and `S_vib` spline errors.
