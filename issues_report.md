# Issues Report for `machine-learning` branch

This report summarizes the status of issues identified in the `machine-learning` branch following recent updates.

## Status Summary

| Issue ID | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **1** | Crash in `run_model` with `energies=False` | **FIXED** | Safe guard added to skip status update if energies/forces are not computed. |
| **2** | Feature Vector Logic Flaw | **RESOLVED** | Fixed by allowing `energies=False` with `overwrite=True`, enabling FV-only computation. |
| **3** | `Structure.save` ignores geometry | **MITIGATED** | Warning added to docstring. Logic remains unchanged (geometry changes are ignored in default mode). |
| **4** | `AtomicReference` validation | **FIXED** | Explicit validation added for missing elements. |

## Verification Notes

*   **Imports:** Potential issues regarding missing or ambiguous imports (`mbe_automation.structure.crystal`, `mbe_automation.storage.core`) were investigated. These were determined to be **false positives** as the submodules are correctly exposed via their respective `__init__.py` files or loaded into the namespace.
*   **Dependencies:** The `run_model` function and MACE integration depend on several optional packages (`mace-torch`, `pyscf`, `phonopy`, etc.). Ensure these are installed in the target environment.

## Remaining Risks

### Data Integrity in `Structure.save`

**Location:** `src/mbe_automation/storage/core.py` inside `_save_structure`.

**Observation:**
The default behavior of `Structure.save` (`update_mode="update_ground_truth"`) effectively locks the geometry (positions, cell, atomic numbers) once the key exists in the HDF5 file. If a user modifies the structure in memory and calls `save()`, the function will save the *newly computed properties* but associate them with the *old geometry* on disk.

**Mitigation:**
A warning has been added to the docstring advising users to use `update_mode="replace"` if the geometry has changed.

**Recommendation:**
For future hardening, consider adding a runtime check that compares the in-memory geometry with the stored geometry (if `save_basics` is skipped). If they differ, raise a `ValueError` or warning to prevent accidental data corruption.
