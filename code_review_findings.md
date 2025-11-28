# Code Review Findings

## Summary
The review focused on the validity of new features in the `machine-learning` branch, specifically the structure splitting and subsampling logic, and the random number generator (RNG) seeding consistency.

Several inconsistencies were found in the seeding of random number generators for subsampling algorithms. These have been corrected to ensure that all random operations can be deterministically controlled via a passed `rng` object, satisfying the requirement for "properly seeded with different seed for every function call".

Additionally, a review of new files identified some potential issues in `src/mbe_automation/ml/dataset.py`.

## Issues and Fixes

| File Path | Issue Description | Severity | Action Taken |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/ml/core.py` | `_farthest_point_sampling` used global `np.random.randint`, making it impossible to control seeding via `rng`. | Medium | Updated function to accept `rng` argument and use `rng.integers`. |
| `src/mbe_automation/ml/core.py` | `_kmeans_sampling` used a hardcoded `random_state=42`. | Medium | Updated function to accept `rng` and generate a dynamic seed for `KMeans`. |
| `src/mbe_automation/ml/core.py` | `pca` function used `sklearn.decomposition.PCA` without explicit random state control (defaulting to auto/randomized with internal state). | Low | (Not fixed, but noted as potential minor issue if strict PCA reproducibility is needed). |
| `src/mbe_automation/api/classes.py` | `Structure.subsample` and related methods did not accept `rng` argument to propagate to `ml.core`. | Medium | Updated `subsample` methods in `Structure`, `Trajectory`, `MolecularCrystal`, `FiniteSubsystem` to accept `rng`. |
| `src/mbe_automation/ml/dataset.py` | This file appears to be unused in the package (not imported). It contains `get_vacuum_energies` and `process_trajectory` which seem to be helper scripts for a specific workflow not yet integrated. | Low | Noted in findings. |
| `src/mbe_automation/ml/dataset.py` | `process_trajectory` modifies the input list of Atoms objects in-place, which is a side-effect that might be unexpected. | Low | Noted in findings. |
| `src/mbe_automation/ml/dataset.py` | Explicit import of `mace` at top level creates a hard dependency, whereas other modules handle it gracefully. | Low | Noted in findings. |

## Verification
A test script `test_seeding.py` was created and executed to verify:
1. `farthest_point_sampling` produces identical results with the same seed.
2. `farthest_point_sampling` produces different results with different seeds.
3. `kmeans` sampling produces identical results with the same seed.
4. `kmeans` sampling produces different results with different seeds.
5. `Structure.subsample` correctly propagates the `rng` argument.

All tests passed.
