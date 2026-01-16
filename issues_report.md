# Issues Report for `machine-learning` branch

This report updates the status of previously identified issues in the `machine-learning` branch after the recent patch.

## Status Summary

| Issue ID | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **1** | Crash in `run_model` with `energies=False` | **FIXED** | Safe guard added to skip status update if energies/forces are not computed. |
| **2** | Feature Vector Logic Flaw | **RESOLVED** | Fixed by allowing `energies=False` with `overwrite=True`, enabling FV-only computation. |
| **3** | `Structure.save` ignores geometry | **MITIGATED** | Warning added to docstring. Logic remains unchanged (geometry changes are ignored in default mode). |
| **4** | `AtomicReference` validation | **FIXED** | Explicit validation added for missing elements. |
| **5** | Missing import in `from_xyz_file` | **INVALID** | Verified that `structure` package exposes `crystal` submodule. |
| **6** | Ambiguous Import in `AtomicReference` | **INVALID** | Verified that `mbe_automation.storage.core` resolves correctly at runtime. |

## Remaining Concerns

### 1. Data Integrity Risk in `Structure.save`

**Location:** `src/mbe_automation/storage/core.py` inside `_save_structure`.

**Description:**
While a warning has been added to the documentation, the code still silently ignores geometry updates (positions, cell vectors, atomic numbers) when `update_mode="update_ground_truth"` (default) is used on an existing dataset key.
If a user modifies `structure.positions` in memory and calls `save()`, the HDF5 file will store the *new* energies (computed on new positions) associated with the *old* positions (retained in file). This leads to a mismatch between geometry and properties.

**Recommendation:**
Consider implementing a check that compares the in-memory geometry with the stored geometry when `save_basics` is skipped. If they differ, raise a `ValueError` or print a runtime warning, prompting the user to use `update_mode="replace"`.
