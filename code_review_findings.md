# Code Review: Thermal Expansion in Quasi-Harmonic Workflow (Machine Learning Branch)

## Summary
A re-evaluation of the code related to thermal expansion properties was performed on the `machine-learning` branch (commit `5eb0b90`). The previous issue regarding the ignored `save_plots` parameter has been resolved. A critical logic error was identified in the equation of state (EOS) fitting routine, which prevented the usage of nonlinear EOS models. This error has been corrected in the accompanying code changes.

## Findings and Fixes

| Severity | File Path | Line Number | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| **High** | `src/mbe_automation/dynamics/harmonic/eos.py` | 179 | **Logic Error Disables Nonlinear EOS**: The conditional logic incorrectly returned the polynomial fit result when it succeeded, even if a nonlinear EOS (e.g., "vinet") was requested. Conversely, it attempted to run nonlinear fits with invalid parameters if the polynomial fit failed. | **FIXED** |

## Detailed Analysis

### 1. Fixed Logic in `fit` Function
In `src/mbe_automation/dynamics/harmonic/eos.py`, the `fit` function logic was flawed:

```python
    # OLD CODE
    if (
            equation_of_state in linear_fit or
            (equation_of_state in nonlinear_fit and poly_fit.min_found)
    ):
        return poly_fit
```

This forced the use of the polynomial fit whenever it was successful, ignoring the user's request for nonlinear models like Vinet or Birch-Murnaghan.

**The Correction:**
The logic has been updated to:
1.  Return the polynomial fit immediately if a linear fit is requested.
2.  Return the polynomial fit if it failed (regardless of request), as it is needed to initialize the nonlinear fit.
3.  Proceed to the nonlinear fit block if requested and if initialization data is available (i.e., polynomial fit succeeded).

```python
    # NEW CODE
    if equation_of_state in linear_fit:
        return poly_fit

    if not poly_fit.min_found:
        return poly_fit

    if equation_of_state in nonlinear_fit:
        # ... proceed with nonlinear fit
```
