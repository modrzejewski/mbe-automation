# Report on `api.classes._run_model` Redundant Computation Bug

## Summary
The function `mbe_automation.api.classes._run_model` fails to correctly implement the `overwrite=False` logic. While it correctly identifies which frames require computation, it does not use this information to filter the structure passed to the calculator. Consequently, all frames (or all user-selected frames) are re-computed regardless of their completion status, leading to unnecessary computational expense and overwriting of existing data.

## Code Analysis

The issue is located in `src/mbe_automation/api/classes.py` within the `_run_model` function:

```python
    # (1) The function correctly calculates the subset of frames that need computation.
    frames_to_compute = _frames_to_compute(
        structure=structure,
        level_of_theory=calculator.level_of_theory,
        overwrite=overwrite,
        selected_frames=selected_frames,
    )

    # ... (omitted checks) ...

    if len(frames_to_compute) == 0:
        print(f"Found zero uncompleted frames to process with {calculator.level_of_theory}.")
        return

    # ... (omitted setup) ...

    # (2) HERE IS THE BUG:
    # The calculation_structure is derived from 'structure' or 'selected_frames',
    # completely ignoring 'frames_to_compute'.
    calculation_structure = structure
    if selected_frames is not None:
        calculation_structure = structure.select(selected_frames)

    # (3) The calculator is run on the full set of frames instead of the subset.
    E_pot, F, d, statuses = mbe_automation.calculators.run_model(
        structure=calculation_structure,
        # ...
    )
```

Because `calculation_structure` includes already-computed frames, the calculator re-evaluates them. The results are then assigned back to the structure, effectively behaving as if `overwrite=True` was set (for the subset of frames defined by `selected_frames` or the whole structure).

## Reproduction
A reproduction script was created to verify this behavior.
- **Scenario:** A structure with 10 frames, where 5 are marked as `CALCULATION_STATUS_COMPLETED` in `ground_truth`.
- **Action:** Call `_run_model` with `overwrite=False`.
- **Expected Result:** The calculator should receive a structure with only 5 frames.
- **Actual Result:** The calculator received a structure with all 10 frames.

## Impact
- **Performance:** Significant waste of computational resources (CPU/GPU time) when resuming calculations or filling in missing data.
- **Data Integrity:** Existing data is silently overwritten, which might be undesirable if the new calculation parameters (e.g., calculator settings) are different but the level of theory label is the same.

## Recommendation
The `_run_model` function should be refactored to:
1. Construct `calculation_structure` using `frames_to_compute`.
2. Update the result assignment logic to map the outputs (which will match the size of `frames_to_compute`) back to the correct indices in the original structure's `ground_truth`.
