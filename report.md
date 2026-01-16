# Report on `api.classes._run_model` Redundant Computation Bug

## Summary
The function `mbe_automation.api.classes._run_model` failed to correctly implement the `overwrite=False` logic. While it correctly identified which frames required computation, it did not use this information to filter the structure passed to the calculator. Consequently, all frames (or all user-selected frames) were re-computed regardless of their completion status, leading to unnecessary computational expense and overwriting of existing data.

**Update:** The bug has been fixed in the `fault-tolerance` branch (commit `efba2db`). The `_run_model` function now correctly creates a subset of the structure containing only the uncompleted frames before passing it to the calculator.

## Code Analysis (Original Issue)

The issue was located in `src/mbe_automation/api/classes.py` within the `_run_model` function:

```python
    # (1) The function correctly calculates the subset of frames that need computation.
    frames_to_compute = _frames_to_compute(
        structure=structure,
        level_of_theory=calculator.level_of_theory,
        overwrite=overwrite,
        selected_frames=selected_frames,
    )

    # ... (omitted checks) ...

    # (2) HERE WAS THE BUG:
    # The calculation_structure was derived from 'structure' or 'selected_frames',
    # completely ignoring 'frames_to_compute'.
    calculation_structure = structure
    if selected_frames is not None:
        calculation_structure = structure.select(selected_frames)

    # (3) The calculator was run on the full set of frames instead of the subset.
    E_pot, F, d, statuses = mbe_automation.calculators.run_model(
        structure=calculation_structure,
        # ...
    )
```

## Fix Analysis

The updated code now correctly selects the subset of frames:

```python
    # To avoid calculation on frames with COMPLETE status,
    # create a new view of the structure with the subset
    # of frames where the data are missing
    calculation_structure = structure.select(frames_to_compute)

    E_pot, F, d, statuses = mbe_automation.calculators.run_model(
        structure=calculation_structure,
        # ...
    )
```

Additionally, the result assignment logic has been updated to map the outputs (which match the size of `frames_to_compute`) back to the correct indices in the original structure's `ground_truth`.

## Verification
A reproduction script confirmed the fix:
- **Scenario:** A structure with 10 frames, where 5 are marked as `CALCULATION_STATUS_COMPLETED` in `ground_truth`.
- **Action:** Call `_run_model` with `overwrite=False`.
- **Expected Result:** The calculator should receive a structure with only 5 frames.
- **Actual Result (Fixed):** The calculator received a structure with 5 frames.

## Impact
- **Performance:** Computational resources are now efficiently used, as already completed frames are skipped.
- **Data Integrity:** Existing data for completed frames is preserved when `overwrite=False`.
