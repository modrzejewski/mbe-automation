# Code Review Findings

## Summary
The following issues were identified during the static analysis of the codebase, focusing on `src/mbe_automation/api/classes.py`, `src/mbe_automation/ml/delta.py`, and `src/mbe_automation/ml/mace.py`.

## Critical Issues (Potential Runtime Errors)

### 1. `Dataset` object is not iterable in `export_to_mace`
*   **Location:** `src/mbe_automation/api/classes.py`, line 597
*   **Description:** The `Dataset.export_to_mace` method calls `_export_to_mace` passing `dataset=self`. The `_export_to_mace` function then attempts to iterate over this object (`for x in dataset:`). Since `Dataset` is a dataclass and not iterable, this will raise a `TypeError`.
*   **Fix:** Pass `dataset=self.structures` instead of `dataset=self`.

```python
# src/mbe_automation/api/classes.py

# Current:
def export_to_mace(self, ...):
    _export_to_mace(
        dataset=self,  # <--- Error: self is not iterable
        ...
    )

# Correction:
def export_to_mace(self, ...):
    _export_to_mace(
        dataset=self.structures, # <--- Fix
        ...
    )
```

## Logical Errors

### 1. Inconsistent Linear Regression Target in `delta.py`
*   **Location:** `src/mbe_automation/ml/delta.py`, function `_energy_shifts_linear_regression`
*   **Description:** The linear regression calculates atomic shifts `x` by fitting them to `E_target - sum(n * E_atomic_baseline)`. However, the actual training data for the MACE delta model (in `export_to_mace`) uses `E_target - E_baseline` as the target potential energy.
    *   This discrepancy means the "IsolatedAtom" energies (derived from `x`) will effectively include the baseline binding energy, whereas the bulk training data (derived from `E_target - E_baseline`) does not.
    *   For a delta model $E_{\Delta} = E_{target} - E_{baseline}$, the atomic shifts for the delta model should ideally approximate $E_{target}(atom) - E_{baseline}(atom)$.
    *   The current implementation fits $E_{shift} \approx E_{target}(structure) - E_{sum\_atomic\_baseline}(structure)$.
*   **Recommendation:** Change the regression target `b` to `E_target - E_baseline`.

```python
# src/mbe_automation/ml/delta.py

# Current:
E_delta = E_target[i] - np.sum(n * E_atomic_baseline)

# Recommendation:
# Use the actual baseline energy of the structure
E_baseline = _baseline_energies(structures) # Ensure this is passed or available
E_delta = E_target[i] - E_baseline[i]
```

## Major Issues (Code Quality / Python Antipatterns)

### 1. Mutable Default Arguments
*   **Location:** `src/mbe_automation/api/classes.py`
    *   `Dataset.export_to_mace`: `save_paths` (list) and `fractions` (numpy array).
    *   `_export_to_mace`: `save_paths` (list) and `fractions` (numpy array).
*   **Description:** Using mutable objects as default argument values is dangerous because they are created only once at function definition time. Modifications to these arguments (e.g., `save_paths.append(...)`) will persist across subsequent function calls.
*   **Fix:** Use `None` as the default value and initialize the list/array inside the function.

```python
# src/mbe_automation/api/classes.py

def export_to_mace(
        ...,
        save_paths: List[str] | None = None,
        fractions: npt.NDArray[np.float64] | None = None,
):
    if save_paths is None:
        save_paths = ["train.xyz", "validate.xyz", "test.xyz"]
    if fractions is None:
        fractions = np.array([0.90, 0.05, 0.05])
    ...
```

## Minor Issues / Observations

*   **Type Hinting:** In `src/mbe_automation/api/classes.py`, `mbe_automation.ml.delta.export_to_mace` is called with `save_path` (singular), but the `export_to_mace` method in `Dataset` takes `save_paths` (plural list). The logic correctly iterates and passes single paths, but careful attention is needed to ensure the length of `save_paths` matches the number of splits (3). The code asserts this, which is good.
*   **Data Validation:** In `src/mbe_automation/ml/delta.py`, `_baseline_forces` correctly raises a `ValueError` if forces are inconsistent (present in some frames but not others). This is a good safety check.
