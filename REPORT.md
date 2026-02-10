# Code Review Report: Machine Learning Features

## Overview
This report summarizes the findings from a review of the new machine learning features in the `mbe_automation` package. The review covered `src/mbe_automation/ml/`, `src/mbe_automation/calculators/`, and `src/mbe_automation/workflows/`.

## Summary of Findings

| Severity | File | Issue | Description |
| :--- | :--- | :--- | :--- |
| **High** | `src/mbe_automation/workflows/training.py` | Logic Error | In `phonon_sampling`, when using `DISTANCE_SELECTION` for finite subsystems, the code iterates over the extracted subsystems and assumes a 1-to-1 correspondence with the input `distances` list. However, `extract_finite_subsystem` filters out duplicate-sized clusters, potentially causing a mismatch where the wrong distance label is applied to a subsystem in the HDF5 key. |
| **High** | `src/mbe_automation/dynamics/harmonic/refinement.py` | Missing Dependency Source | The `nomore_ase` library is required for `NormalModeRefinement` but its source code is missing from the repository (not in `src/nomore_ase`) and cannot be installed from PyPI or GitHub (404 error). This renders the refinement module unusable. |
| Medium | `src/mbe_automation/calculators/core.py` | Limitation | `run_model` restricts Ray parallelization to multi-GPU setups (`n_workers = min(resources.n_gpus, ...)`). This prevents parallelization on high-core-count CPU-only nodes even if `ray` is available. |
| Low | `src/mbe_automation/ml/dataset.py` | Ambiguity | `get_vacuum_energies` returns energy *differences* (reference - base) but prints messages implying absolute vacuum energies. The function name is slightly misleading. Also, this file appears to be a standalone utility not integrated with the main workflow. |
| Low | `src/mbe_automation/ml/mace.py` | Potential Edge Case | In `to_xyz_training_set`, if `delta_learning` is enabled but one of the models (target/baseline) returns `None` for energies/forces, the resulting training entry will have missing labels. While `_to_xyz_training_set` handles this (by not writing the key), it relies on MACE training to handle unlabeled data or fail gracefully. |
| Low | `src/mbe_automation/structure/clusters.py` | Assumption | `detect_molecules` uses `reference_frame_index=0` to determine molecular connectivity. For MD trajectories where bond topology might change (rare but possible), this assumption might be invalid. |

## Detailed Analysis & Recommendations

### 1. Finite Subsystem Extraction Logic (High Severity)
**Issue:**
In `src/mbe_automation/workflows/training.py`:
```python
    for i, s in enumerate(finite_subsystems):
        if ...:
            distance = config.finite_subsystem_filter.distances[i]
            key = f"{config.root_key}/finite_subsystems/r={distance:.2f}"
```
In `src/mbe_automation/structure/clusters.py`, `extract_finite_subsystem` iterates over sorted distances and only appends a new subsystem if its size (`n_molecules`) is strictly larger than the previous one. This means `len(finite_subsystems)` can be less than `len(distances)`, and the indices `i` will not align with the original `distances` array.

**Recommendation:**
Modify `extract_finite_subsystem` to return a list of tuples `(subsystem, distance_used)` or a dictionary mapping distance to subsystem. Alternatively, change the loop in `phonon_sampling` to iterate over the *extracted* subsystems and retrieve their generation parameters if stored, or simply use `n_molecules` as the unique identifier for the key (as is done for `NUMBER_SELECTION`) since `extract_finite_subsystem` guarantees unique sizes. If the distance label is strictly required in the key, `extract_finite_subsystem` must provide it.

### 2. Ray Parallelization on CPUs (Medium Severity)
**Issue:**
`run_model` in `src/mbe_automation/calculators/core.py` sets `n_workers = min(resources.n_gpus, structure.n_frames)`. If `resources.n_gpus` is 0, `n_workers` is 0, and sequential execution is forced.

**Recommendation:**
Allow `n_workers` to be determined by CPU core count if GPUs are not available, or add a configuration option to force CPU parallelization.
```python
if resources.n_gpus > 0:
    n_workers = min(resources.n_gpus, structure.n_frames)
else:
    n_workers = min(resources.n_cpu_cores, structure.n_frames) # or similar logic
```

### 3. `dataset.py` Clarification (Low Severity)
**Issue:**
The file `src/mbe_automation/ml/dataset.py` seems to be a standalone script for analyzing vacuum energy shifts between models, but its function names and print statements are slightly confusing regarding whether they deal with absolute energies or differences.

**Recommendation:**
Rename `get_vacuum_energies` to `get_vacuum_energy_shifts` or `calculate_vacuum_deltas`. Ensure docstrings and print statements clearly state that differences are being computed. If this file is not used by the automation pipeline, consider moving it to a `scripts/` or `tools/` directory.

### 4. `to_xyz_training_set` Delta Learning (Low Severity)
**Issue:**
When generating delta learning datasets, if a baseline calculation fails (returns None) for a frame, the delta cannot be computed. The current logic implicitly drops the energy/force labels for that frame.

**Recommendation:**
Add a warning log if frames are being written without energy/force labels due to missing baseline/target data. This helps users debug why their training set might be smaller or less effective than expected.

## Validation of `nomore_ase` Interface

A dedicated review of the interface to `nomore_ase` in `src/mbe_automation/dynamics/harmonic/refinement.py` and `bands.py` was performed.

### Findings

1.  **Missing Source Code:** The `nomore_ase` library source code is completely missing from the repository (`src/nomore_ase` does not exist), despite `pixi.toml` referencing it as a local editable dependency. Attempts to locate it on GitHub (https://github.com/Niolon/nomore_ase) resulted in a 404 error, indicating the repository is likely private or deleted.
    *   **Impact:** The `refinement` module relies heavily on `nomore_ase` (e.g., `RefinementEngine`, `SmtbxAdapter`). Without the library, this module will fail with `ImportError` at runtime.

2.  **Interface Consistency:**
    *   The code in `refinement.py` correctly imports and attempts to use classes like `CctbxAdapter`, `NoMoReCalculator`, and `PhononData`.
    *   The usage pattern matches the mocked tests in `tests/nomore/test_adp_comparison.py`, suggesting the interface *might* be correct if the library existed.
    *   **Unit Mismatch Risk:** `refinement.py` converts frequencies from cm⁻¹ to THz for display but passes cm⁻¹ to `NoMoReCalculator`. Without source code, it's impossible to verify if `NoMoReCalculator` expects cm⁻¹, THz, or angular frequency. However, `test_adp_comparison.py` also passes cm⁻¹ (implied by `phonons.frequencies_cm1` usage), suggesting consistency between test and implementation.

3.  **Data Structure Alignment:**
    *   `to_phonon_data` correctly handles atom permutations between Phonopy (primitive cell) and CCTBX/CIF structures. This is crucial for correct ADP calculation.
    *   `compute_band_indices` in `bands.py` correctly adapts Phonopy objects to the `PhonopyASEAdapter` expected by `assign_bands`.

### Recommendations

1.  **Restore `nomore_ase`:** The most critical action is to locate and restore the `nomore_ase` source code to `src/nomore_ase`. If it is a private submodule, ensure the user has access.
2.  **Mock or Vendor:** If the library cannot be open-sourced, consider vendoring a specific version or mocking the interface entirely for public releases to allow the rest of `mbe_automation` to function (the imports are already guarded by `try-except`).
3.  **Verify Units:** Once source is available, explicitly verify the frequency units expected by `NoMoReCalculator`. If it expects angular frequency (rad/s) or THz, the current passing of cm⁻¹ might be incorrect.

## Conclusion
The machine learning features are generally well-structured. However, the `phonon_sampling` logic bug requires immediate attention to prevent data labeling errors. Furthermore, the `nomore_ase` integration is currently broken due to missing dependencies, which blocks any validation of its physics correctness.
