# Code Review Findings: Quasi-Harmonic Workflow Analysis

## Summary
The quasi-harmonic workflow was analyzed to determine the impact of using a non-strictly increasing series of temperatures. While the data generation phases (calculation of phonons and equilibrium curves) are robust to unsorted temperatures, the final thermal expansion analysis contains hard constraints and algorithmic dependencies that require strictly increasing temperatures.

## Critical Findings

### 1. Explicit Constraint (Hard Failure)
**File:** `src/mbe_automation/dynamics/harmonic/eos.py`
**Location:** `fit_thermal_expansion_properties` function

The code contains an explicit assertion that forces the program to crash if temperatures are not strictly increasing:

```python
assert np.all(np.diff(T) > 0.0), "Temperatures must be strictly increasing."
```

### 2. Algorithmic Dependencies (Hidden Assumptions)
If the explicit assertion were removed, the code would still fail or produce incorrect results due to the requirements of the underlying numerical methods used for differentiation (`dX/dT`).

**File:** `src/mbe_automation/dynamics/harmonic/eos.py`

*   **`scipy.interpolate.CubicSpline`**: Used in `_fit_thermal_expansion_properties_cspline` and `_hybrid_derivative`. This class raises a `ValueError` if the independent variable `x` (temperature) is not strictly increasing.
*   **`numpy.gradient`**: Used in `_fit_thermal_expansion_properties_finite_diff`. This function assumes that the array indices correspond to the physical ordering of the grid points. If `T` is unsorted, `np.gradient` calculates derivatives based on array adjacency rather than physical temperature adjacency, leading to physically meaningless results (e.g., negative or fluctuating intervals).

### 3. Finite Difference Logic Assumptions
**File:** `src/mbe_automation/dynamics/harmonic/eos.py`

The logic for selecting finite difference schemes implicitly assumes that the input array is sorted:

*   The code applies a **forward difference** at the "lowest temperature endpoint" (assumed to be index `0`).
*   The code applies a **backward difference** at the "highest temperature endpoint" (assumed to be index `-1`).

If the temperature array is not sorted (e.g., `[300, 100, 200]`), index `0` is not the global minimum, and index `-1` is not the global maximum. Applying specific boundary schemes to these internal points would be mathematically incorrect.

### 4. Visualization Minor Issues
**File:** `src/mbe_automation/dynamics/harmonic/display.py`
**Location:** `eos_curves`

While the plotting logic generally handles unsorted temperatures correctly (using `vmin`/`vmax` for color mapping), the generation of secondary axis ticks might produce an unrepresentative subset of labels if the input array is unsorted, as it selects indices linearly from the array (`np.linspace(0, len-1, ...)`).

## Conclusion
The requirement for strictly increasing temperatures is not just a superficial check but a fundamental requirement of the numerical differentiation algorithms implemented in `eos.py`. The rest of the workflow (phonon calculations, EOS sampling) is technically compatible with unsorted inputs, but the final analysis step is not.
