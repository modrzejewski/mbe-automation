# Verification Report: `rebase_to_reference` Functionality in Empirical Electronic Energy Correction (EEC)

This report details the comprehensive verification of the `rebase_to_reference` functionality in the `mbe_automation` library, specifically focusing on the `EECConfig`, `rebase_volume_to_reference` function, the `EEC` class, and its integration within quasi-harmonic workflows.

## 1. `EECConfig` Functionality
The `rebase_to_reference` type in `EECConfig` activates the implicit volume correction. The configuration correctly recognizes this mode:

- **Validation:** When `reference_state_forcing="rebase_to_reference"`, both `T_ref` and `V_ref` are mandated (validated via `__post_init__`).
- **Flags:** `config.is_implicit_volume_correction` successfully returns `True` for this mode.
- **Enabled State:** `config.is_enabled` successfully returns `True`, ensuring the correction pipeline is activated.
- **Integration:** The function successfully handles edge cases, such as raising an error if `T_ref` is not found within the input temperature array.

## 2. `rebase_volume_to_reference` Logic
The `rebase_volume_to_reference` function rigidly shifts a quasi-harmonic volume curve so it anchors at the reference point: $V_{corrected}(T) = V_{ref} - V(T_{ref}) + V(T)$.

- **Test Outcome:**
  - **Inputs:** $T = [100.0, 200.0, 300.0]$, $V = [1000.0, 1020.0, 1050.0]$
  - **Config:** `T_ref = 200.0`, `V_ref = 1100.0`
  - **Expected:** $1100.0 - 1020.0 + [1000.0, 1020.0, 1050.0] = [1080.0, 1100.0, 1130.0]$
  - **Actual Output:** `[1080. 1100. 1130.]`
- **Result:** The logic correctly shifts the volume curve, maintaining the exact shape of thermal expansion (constant $dV/dT$).

## 3. Integration with the `EEC` Class
In the context of the `EEC` class, `rebase_to_reference` is designed to be an *implicit* volume correction that does not alter the underlying electronic energy surface analytically.

- **Energy Correction (`_eec_value`):** For `rebase_to_reference`, the energy correction is strictly `0.0`.
  - **Verification:** `eec.evaluate([1000.0, 1100.0])` returns `[0. 0.]`.
- **Pressure Correction (`_eec_pressure`):** For `rebase_to_reference`, the pressure correction is strictly `0.0`.
  - **Verification:** `eec.evaluate_pressure([1000.0, 1100.0])` returns `[0. 0.]`.
- **Parameter Extraction (`_eec_param`):** The implicit correction does not fit an electronic energy parameter.
  - **Verification:** During `EEC.from_sampled_eos_curve`, `param_mlip` successfully short-circuits to `0.0`. `E_el_mlip_corrected` remains completely identical to `E_el_mlip`.

## 4. Integration within Quasi-Harmonic Workflows
The `rebase_to_reference` mode hooks directly into the quasi-harmonic workflow core logic (`src/mbe_automation/workflows/quasi_harmonic.py` and `src/mbe_automation/dynamics/harmonic/core.py`).

- **Volume Curve Selection:** In `equilibrium_curve`, if `is_implicit_volume_correction` is True, it correctly triggers `rebase_volume_to_reference` to populate the `"V_rebased (Å³∕unit cell)"` column in the dataframe.
- **Workflow Hook (`effective_volume_curve`):** In `mbe_automation/workflows/quasi_harmonic.py`, `effective_volume_curve = "rebase_to_reference"` is correctly set if the implicit correction is active.
- **Validity Mask:**
  - The interpolation logic checks if the EOS minimum is found and extrapolated. Since the rebase is a pure algebraic shift, `valid_equilibrium` falls back correctly to the `eos_minimum` validity mask.
  - The `p_thermal_crystal (GPa)` is correctly dropped since the rebased volume is no longer the minimum of $G(V,T)$, avoiding inconsistent geometry relaxations.
- **Geometry Relaxation:** During the geometry relaxation loop for EOS sampling under pressure (`config.eos_sampling == "pressure"`), the `p_effective` does **not** include an EEC evaluation pressure because `is_implicit_volume_correction` blocks it, which is correct as the correction pressure is zero. The workflow instead correctly relies on scaling the cell to `V_rebased` manually.

## Conclusion
The `rebase_to_reference` implicit volume correction behaves consistently with its design:
1. It precisely rigid-shifts the V(T) volume curve.
2. It correctly sets a 0.0 value for energy and pressure corrections.
3. It seamlessly integrates into the quasi-harmonic workflow without interfering with the $p_{thermal}$ minimizations.
