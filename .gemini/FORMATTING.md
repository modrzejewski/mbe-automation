# Preferred Code Formatting

## Function Definitions

*   **Line Breaks:** Always place a line break after every argument definition, including the last one.
*   **Type Hints:** All arguments must have type hints.
*   **Numpy Arrays:** Type hints for numpy arrays must specify both the array type and the numerical type (e.g., `npt.NDArray[np.float64]`).

### Example

```python
def find_degenerate_frequencies(
    freqs: npt.NDArray[np.float64], 
    tolerance: float = 1e-4,
) -> List[List[int]]:
```

## Import Ordering

*   **Project Imports:** Imports from `nomore_ase` and `mbe_automation` should be placed **after** all other imports (e.g., standard library, `numpy`, `scipy`).
*   **Separation:** Include a blank line between the general imports and the project-specific imports.

## Module Usage

*   **Full Paths:** When calling functions from `mbe_automation`, use the full module path (e.g., `mbe_automation.dynamics.harmonic.bands.reorder`).
    *   **Rationale:** Function names are designed to be extremely brief, but sound precise when read with the full module path.
    *   **Exception:** This rule can be relaxed if using the full module path would result in an extremely long line.

## Function Calls

*   **Explicit Arguments:** When calling functions from `mbe_automation` or `nomore_ase`, always use explicit argument passing (e.g., `func(arg1=val1, arg2=val2)`).
*   **Line Breaks:** Always place a line break after every argument in the function call, including the last one.

### Example

```python
mbe_automation.dynamics.harmonic.bands.reorder(
    band_indices=band_indices,
    frequencies=frequencies,
    eigenvectors=eigenvectors,
)
```
