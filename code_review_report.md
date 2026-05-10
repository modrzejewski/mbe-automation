# Code Review Report for `cold_curve` branch

The `cold_curve` branch introduces several changes focused on expanding the Empirical Electronic Energy Correction (EEC) capabilities, specifically tracking multiple variations of the cold curve (MLIP vs external baselines, raw vs corrected). Overall, the code changes are sound and successfully expand the tracking capabilities to evaluate these new baselines. Below is a comprehensive list of observations and suggestions based on syntax, physics, and implementation details.

## General Observations

1. **New EEC Structure:** The `EEC` dataclass was significantly updated. Instead of holding a single `cold_curve` and `baseline_cold_curve` and a single `param`, it now stores separate parameters (`param_mlip`, `param_external`) and separate curves (`cold_curve_baseline_mlip`, `cold_curve_baseline_external`, `cold_curve_corrected_mlip`, `cold_curve_corrected_external`). It also stores multiple components for energies (`E_el_mlip`, `E_el_external`, and their corrected versions).
2. **Backwards Compatibility / Storage:** In `src/mbe_automation/storage/core.py`, both `_save_eec` and `_read_eec` have been updated to persist these new properties. It correctly attempts to use `.get` to retrieve properties that might not be present in older HDF5 files to maintain backward compatibility, although for a few attributes (like the changed name of `param`), reading old files might fail if `param_external` or other keys are not accessed safely.
3. **Display/Plotting Updates:** In `src/mbe_automation/dynamics/harmonic/display.py`, the plotting logic was updated to render multiple cold curves when available (`cold_curve_mlip`, `cold_curve_mlip_corrected`, `cold_curve_external`, `cold_curve_external_corrected`), using consistent coloring and line styles.
4. **Keyword Expansion for RPA:** `src/mbe_automation/readout/rpa.py` expanded the `MethodDependentKeywords` from just `Method == "JCTC2024" or Method == "ph-RPA(3)"` to include `Method == "RPA+ph"`.
5. **String Formatting Changes (Angstroms):** In `eos.py` and across several other files, the character for Angstroms `Å` (U+00C5) was changed to `Å` (U+212B, Angstrom sign). This is an aesthetic/consistency change that spans dictionaries, plot labels, and print statements.

## Physics & Functionality

### 1. External Cold Curve
The `baseline_cold_curve` function in `eos.py` was renamed to `external_cold_curve`. The previous logic essentially built an idealized equation of state from user-specified `V0`, `B0`, `B0_prime`, and optionally `E0`. This remains valid.

### 2. EEC Parameter Calculation (`_eec_param`)
In `src/mbe_automation/dynamics/harmonic/eec.py`, the code now computes `param_mlip` and `param_external` separately. The `EEC.from_sampled_eos_curve` constructs the correction correctly:
- First, it evaluates the pure MLIP correction using `_eec_param(..., electronic_energy_source="mlip")`.
- Then, if an external baseline is active, it evaluates the external correction using `_eec_param(..., electronic_energy_source="external")`.
- The residual energies `E_el_mlip_corrected` and `E_el_external_corrected` are calculated by summing the raw MLIP energies with `_eec_value`. The comments explain that since the spline fits `E_el_mlip` perfectly, the `- E_mlip_spline(V)` inside `_eec_value` perfectly cancels `E_el_mlip` at the sampled points. This leaves `E_external_base(V_sampled) + E_corr(V_sampled)` for the external curve. This logic is physically and mathematically sound.

### 3. Display Priorities
When rendering the plots, `display.py` sets an explicit `_COLD_CURVE_REFERENCE_PRIORITY`. The priority favors corrected versions:
1. `cold_curve_mlip_corrected`
2. `cold_curve_external_corrected`
3. `cold_curve_mlip`
4. `cold_curve_external`
This correctly determines the reference $V_0$ and the reference $E_{el}$ to shift the plot energies such that the minimum is zero.

## Code Review Suggestions

### A. Backward Compatibility in HDF5 Storage
In `src/mbe_automation/storage/core.py`, the `_read_eec` function currently looks like this:
```python
def _read_eec(group: h5py.Group):
    # ...
    electronic_energy_source = group.attrs.get("electronic_energy_source", "none")
    param_mlip = float(group.attrs.get("param_mlip", 0.0))
    param_external = float(group.attrs["param_external"]) if "param_external" in group.attrs else None
    # ...
```
While this reads the new fields safely, the old `param` attribute (which was present when `reference_state_forcing` was `"linear"`, `"inverse_volume"`, or `"rigid_shift"`) is no longer being read.
If a user tries to read an older HDF5 file with `reference_state_forcing != "none"`, the `param_mlip` will default to `0.0`. The old `param` is stored differently based on the type (e.g., `param (kJ∕mol∕Å³)`).
**Suggestion:** Add a fallback block to read the old `param` and assign it to `param_mlip` if `param_mlip` is `0.0` but `reference_state_forcing != "none"`:
```python
    if "param_mlip" in group.attrs:
        param_mlip = float(group.attrs["param_mlip"])
    else:
        # Fallback for old files
        if reference_state_forcing == "linear" and "param (kJ∕mol∕Å³)" in group.attrs:
            param_mlip = float(group.attrs["param (kJ∕mol∕Å³)"])
        elif reference_state_forcing == "inverse_volume" and "param (kJ∕mol⋅Å³)" in group.attrs:
            param_mlip = float(group.attrs["param (kJ∕mol⋅Å³)"])
        elif reference_state_forcing == "rigid_shift" and "param (Å³∕unit cell)" in group.attrs:
            param_mlip = float(group.attrs["param (Å³∕unit cell)"])
        elif "param" in group.attrs:
            param_mlip = float(group.attrs["param"])
        else:
            param_mlip = 0.0
```

Also, note that old files stored `E_el_raw_sampled (kJ∕mol∕unit cell)` instead of `E_el_mlip (kJ∕mol∕unit cell)` and `F_vib_sampled` instead of `F_vib_mlip`. `_read_eec` must handle this:
```python
        E_el_mlip = group["E_el_mlip (kJ∕mol∕unit cell)"][:] if "E_el_mlip (kJ∕mol∕unit cell)" in group else group["E_el_raw_sampled (kJ∕mol∕unit cell)"][:]
        F_vib_mlip = group["F_vib_mlip (kJ∕mol∕unit cell)"][:] if "F_vib_mlip (kJ∕mol∕unit cell)" in group else group["F_vib_sampled (kJ∕mol∕unit cell)"][:]
```

### B. Unit Sign Changes (`Å` vs `Å`)
You've changed the strings to use the Angstrom sign (`Å` U+212B) instead of the Latin capital letter A with ring above (`Å` U+00C5). While `Å` is the mathematically correct symbol for Angstroms, in HDF5 data formats and API returns, this change breaks backwards compatibility with any script that parses the dictionary keys or HDF5 group names (e.g., `B0 (kJ∕mol∕Å³)` became `B0 (kJ∕mol∕Å³)`).
**Suggestion:** If old HDF5 files contain keys using `Å`, reading them with `group["V_sampled (Å³∕unit cell)"]` will raise a `KeyError`. You may want to provide a fallback:
```python
def _get_dataset(group, new_key, old_key):
    if new_key in group:
        return group[new_key][:]
    elif old_key in group:
        return group[old_key][:]
    else:
        raise KeyError(f"Neither '{new_key}' nor '{old_key}' found in group.")
```

### C. `display.py` Variable Referencing
In `src/mbe_automation/dynamics/harmonic/display.py`, you reference `ref_V0` on line 413:
```python
            bbox_to_anchor=(ref_V0, 0.98),
```
If `cold_curve` is empty, `ref_V0` will be `None`. While `ax_cold` wouldn't exist and this line is nested inside `if ax_cold is not None:`, ensure that `ref_V0` handles the case where it might not be properly populated but `ax_cold` is somehow active.

## Final Recommendation
The physics of the baseline/external cold curve are correctly implemented, with the analytical splines nicely canceling out the numerical differences to provide a smooth adjusted curve. The main concern is **backward compatibility** with existing HDF5 databases that expect the previous string names for dataset parameters (`Å` vs `Å`, `E_el_raw_sampled` vs `E_el_mlip`). Updating the `_read_eec` function in `mbe_automation/storage/core.py` to support fallback keys will make this branch robust for merging.
