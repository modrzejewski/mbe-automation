# Code Review: Thermal Expansion in Quasi-Harmonic Workflow

## Summary
A code review was performed on the recent changes related to thermal expansion properties within the quasi-harmonic workflow. The review covered the workflow logic, equation of state (EOS) fitting, and configuration management.

## Findings

| Severity | File Path | Line Number | Description |
| :--- | :--- | :--- | :--- |
| **High** | `src/mbe_automation/dynamics/harmonic/eos.py` | 179 | **Potential Crash in Nonlinear Fit**: The `fit` function attempts to use parameters from a polynomial fit (`poly_fit.G_min`, etc.) to initialize the nonlinear `curve_fit` (Vinet/Birch-Murnaghan) even if the polynomial fit failed (`min_found=False`). If `poly_fit` fails, its attributes are `np.nan`, causing `curve_fit` to be initialized with NaNs, which may lead to runtime errors. A check `if not poly_fit.min_found: return poly_fit` should be added before the nonlinear fit block. |
| **Low** | `src/mbe_automation/workflows/quasi_harmonic.py` | 240 | **Ignored Configuration Parameter**: The `save_plots` parameter from the configuration is ignored when generating EOS curve plots. The `equilibrium_curve` function is called without checking `config.save_plots`, and it unconditionally generates and saves the `eos_curves.png` plot via `mbe_automation.dynamics.harmonic.display.eos_curves`. |

## Detailed Analysis

### 1. Unsafe Initialization of Nonlinear EOS Fit
In `src/mbe_automation/dynamics/harmonic/eos.py`, the `fit` function structure is as follows:

```python
    poly_fit = polynomial_fit(V, G)

    if equation_of_state in linear_fit:
        return poly_fit

    if equation_of_state in nonlinear_fit:
        G_initial = poly_fit.G_min
        V_initial = poly_fit.V_min
        # ...
        popt, pcov = scipy.optimize.curve_fit(
                eos_func,
                # ...
                p0=np.array([G_initial, V_initial, B_initial, B_prime_initial]),
                # ...
```

If `polynomial_fit` returns a result with `min_found=False` (e.g., due to insufficient data points), `poly_fit.G_min` and others are set to `np.nan`. Passing these `NaN` values to `scipy.optimize.curve_fit` via `p0` is unsafe.

**Recommendation:**
Add a guard clause:
```python
    if not poly_fit.min_found:
        return poly_fit
```

### 2. `save_plots` Configuration Ignored
The `FreeEnergy` configuration class defines `save_plots: bool = True`. However, in `src/mbe_automation/workflows/quasi_harmonic.py`, the plotting logic within `equilibrium_curve` (in `core.py`) is hardcoded to always save the plot if the function is called, as it constructs a save path regardless of the config setting.

**Recommendation:**
Update `src/mbe_automation/dynamics/harmonic/core.py` to accept a `save_plots` argument in `equilibrium_curve` and conditionally pass `save_path` to `display.eos_curves`.
