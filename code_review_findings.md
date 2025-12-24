# Code Review: Ground Truth Operations (Revised)

## Summary
This review analyzes the `ground_truth` operations in the `machine-learning` branch. While recent changes simplified the handling of `level_of_theory` by centralizing results into `ground_truth`, several critical integration issues remain, particularly regarding data persistence and object initialization.

## Critical Issues

### 1. Data Loss in Trajectory Subsampling
**Severity:** **CRITICAL**
**Location:** `src/mbe_automation/api/classes.py` -> `_subsample_trajectory` (lines ~620-660)

The `_subsample_trajectory` function constructs a new `_Trajectory` object representing the subset of frames. However, it fails to pass the `ground_truth` argument to the constructor.
*   **Consequence:** The returned `Trajectory` object has `ground_truth=None`. Any calculated energies/forces associated with the original trajectory are lost during subsampling.
*   **Fix Required:** Slice the ground truth object and pass it to the constructor:
    ```python
    ground_truth=(
        traj.ground_truth.select_frames(selected_indices)
        if traj.ground_truth is not None else None
    )
    ```

### 2. Broken Save Workflow for 'potential_energies'
**Severity:** **CRITICAL**
**Location:** `src/mbe_automation/storage/core.py` -> `_save_only` vs `src/mbe_automation/api/classes.py` -> `_run_model`

The `_run_model` method now exclusively updates `structure.ground_truth` and leaves `structure.E_pot` and `structure.forces` as `None` (or stale). However, `_save_only` explicitly raises a `RuntimeError` if `only=["potential_energies"]` is requested but `structure.E_pot` is `None`.

*   **Scenario:**
    ```python
    structure.run_model(calc, energies=True) # Updates ground_truth, E_pot remains None
    structure.save(..., only=["potential_energies"]) # Raises RuntimeError
    ```
*   **Consequence:** The standard workflow for saving calculation results is broken. Users must now know to save `ground_truth` instead, but the error message is misleading.
*   **Fix Required:**
    1.  Update `structure.E_pot` / `forces` in `_run_model` to reflect the latest calculation (mirroring the data).
    2.  Or, modify `_save_only` to pull from `ground_truth` if `E_pot` is missing but a valid `level_of_theory` is active.

## Major Issues

### 3. "Zombie" Fields: `E_pot` and `forces`
**Severity:** **MAJOR**
**Location:** `src/mbe_automation/storage/core.py` -> `Structure` dataclass

The `Structure` dataclass retains `E_pot` and `forces` fields, but the primary calculation engine (`_run_model`) no longer updates them.
*   **Consequence:** These fields become unreliable ("zombie" data). They might hold `None` or data loaded from a file, but they do not reflect the in-memory state after a calculation. Accessing `structure.E_pot` directly is now a trap for developers.
*   **Fix Required:** Either deprecate these fields and use properties that proxy to `ground_truth`, or ensure they are kept in sync as the "active" level of theory data.

### 4. Missing `ground_truth` in `Trajectory.empty`
**Severity:** **MAJOR**
**Location:** `src/mbe_automation/storage/core.py` -> `Trajectory.empty`

The factory method `Trajectory.empty` initializes a new trajectory but does not accept a `ground_truth` argument.
*   **Consequence:** Prevents initializing an empty trajectory with a pre-existing or known ground truth container.
*   **Fix Required:** Add `ground_truth: GroundTruth | None = None` to the arguments and pass it to the constructor.

## Minor Issues

### 5. `from_ase_atoms` Ignores Ground Truth
**Severity:** Minor
**Location:** `src/mbe_automation/storage/views.py` -> `from_ase_atoms`

When converting from ASE Atoms (used by `Structure.from_xyz_file`), the function does not parse energy/force information (e.g., `REF_energy`, `REF_forces` arrays/info) into the `Structure` or its `ground_truth`.
*   **Consequence:** Loading training data from XYZ files results in structures without their ground truth labels.

## Findings Table

| File Path | Line Number (approx) | Severity | Description |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/api/classes.py` | 650 | **CRITICAL** | `_subsample_trajectory` drops `ground_truth`. |
| `src/mbe_automation/storage/core.py` | 1190 | **CRITICAL** | `_save_only` raises RuntimeError for `potential_energies` because `run_model` doesn't populate `E_pot`. |
| `src/mbe_automation/storage/core.py` | 60 | **MAJOR** | `Structure.E_pot` / `forces` are not updated by calculations, leading to stale data. |
| `src/mbe_automation/storage/core.py` | 150 | **MAJOR** | `Trajectory.empty` factory missing `ground_truth`. |
| `src/mbe_automation/storage/views.py` | 15 | Minor | `from_ase_atoms` fails to import energy/forces from ASE object. |
