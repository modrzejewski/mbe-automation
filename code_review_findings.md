# Code Review: Ground Truth Operations

## Summary
This review focuses on the implementation of `ground_truth` operations (read/write/modify) within `Structure`, `Trajectory`, and related classes. The review identifies a **critical data loss issue** in trajectory subsampling and a **synchronization risk** during partial updates (`save(only=...)`).

## Critical Issues

### 1. Data Loss in Trajectory Subsampling
**Severity:** **CRITICAL**
**Location:** `src/mbe_automation/api/classes.py` -> `_subsample_trajectory`

When subsampling a `Trajectory` object (e.g., via `Trajectory.subsample`), a new `_Trajectory` object is instantiated to hold the selected frames. The constructor call for this new object **omits the `ground_truth` argument**.

*   **Consequence:** The subsampled trajectory object loses all ground truth data (energies/forces for different levels of theory), resetting `ground_truth` to `None`.
*   **Fix Required:** Pass `ground_truth=traj.ground_truth.select_frames(selected_indices) if traj.ground_truth is not None else None` to the `_Trajectory` constructor.

### 2. Inconsistent Updates with `_save_only`
**Severity:** **SEVERE**
**Location:** `src/mbe_automation/storage/core.py` -> `_save_only`

The `_save_only` function allows updating specific datasets (e.g., `only=['potential_energies']`) in an existing HDF5 group. However, it fails to maintain the "hard link" between the active `level_of_theory` and the `ground_truth` group.

*   **Scenario:** A user updates `potential_energies` for a structure with `level_of_theory="DFT"`, calling `save(..., only=['potential_energies'])`.
*   **Issue:** The `E_pot` dataset is updated in the structure root, but the corresponding entry in the `ground_truth` group (`ground_truth/E_DFT...`) is **not updated**.
*   **Consequence:** Data desynchronization. The `ground_truth` archive becomes stale and no longer matches the "active" data for that level of theory.
*   **Fix Required:** `_save_only` must either:
    1.  Automatically imply `ground_truth` update when `potential_energies` or `forces` are modified and a `level_of_theory` is set.
    2.  Explicitly call `_hard_link_to_ground_truth` after updates.

## Major Issues

### 3. Missing `ground_truth` in `Trajectory.empty`
**Severity:** **MAJOR**
**Location:** `src/mbe_automation/storage/core.py` -> `Trajectory.empty`

The factory method `Trajectory.empty` creates a new trajectory instance but does not accept a `ground_truth` argument.

*   **Consequence:** Users cannot initialize a fresh `Trajectory` with existing ground truth data using this method. They must instantiate the class manually or set `ground_truth` after creation, which breaks the factory pattern's utility.
*   **Fix Required:** Add `ground_truth: GroundTruth | None = None` to the `Trajectory.empty` signature and pass it to the constructor.

## Minor Issues & Suggestions

### 4. Redundant Logic in `_hard_link_to_ground_truth`
**Severity:** Minor
**Location:** `src/mbe_automation/storage/core.py`

`_hard_link_to_ground_truth` deletes the existing key in the `ground_truth` group before writing. This is technically redundant if `h5py`'s standard overwrite mechanics (or a check for existence) were used, but more importantly, it creates a momentary state where data is deleted before being re-written. If the write fails (e.g., disk quota), data is lost.

### 5. `Structure.save` Documentation
**Severity:** Minor
**Location:** `src/mbe_automation/storage/core.py`

The docstring for `Structure.save` (and `Trajectory.save`) mentions the `only` argument but does not warn users that omitting `ground_truth` from the `only` list while updating energies/forces might lead to inconsistent ground truth records in HDF5.

## Findings Table

| File Path | Line Number (approx) | Severity | Description |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/api/classes.py` | 620-645 | **CRITICAL** | `_subsample_trajectory` constructor call missing `ground_truth`. |
| `src/mbe_automation/storage/core.py` | 1180-1230 | **SEVERE** | `_save_only` fails to call `_hard_link_to_ground_truth`. |
| `src/mbe_automation/storage/core.py` | 134 | **MAJOR** | `Trajectory.empty` missing `ground_truth` argument. |
| `src/mbe_automation/storage/core.py` | 1286 | Minor | `_hard_link_to_ground_truth` deletes data before writing (risk of data loss on failure). |
