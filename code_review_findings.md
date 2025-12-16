# Code Review Findings: Machine Learning Branch

## Summary
The machine-learning branch introduces workflows for training set generation (`MDSampling`, `PhononSampling`) and Delta Learning (`ml.delta`). While the architectural integration with `Structure` and `Dataset` classes is sound, there are critical logical errors in the Delta Learning implementation regarding energy reference subtraction and regression targets. Additionally, legacy code and potential runtime errors due to incorrect HDF5 keys were identified.

## Findings

| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/ml/delta.py` | 196 | **Incorrect Regression Target**: The linear regression for energy shifts fits to `E_target - E_atomic_baseline` (atomization energy difference) instead of `E_target - E_baseline` (total delta). This fails to account for the baseline model's interaction energies, defeating the purpose of Delta Learning. | High |
| `src/mbe_automation/ml/delta.py` | 276 | **Reference Molecule Scaling**: The logic `Delta_E_pot = Delta_E_pot - Delta_E_pot_molecule` subtracts a scalar molecular energy difference from the crystal energy difference. It fails to explicitly scale the molecular shift by the number of molecules ($N$) in the target system, which is required for correct extensive property handling. | High |
| `src/mbe_automation/ml/descriptors/mace.py` | 98, 111 | **HDF5 Key Mismatch**: The function `atomic_hdf5` accesses datasets using keys `'cells'` and `'positions'`, but the standard storage format in this project uses units in keys, e.g., `'positions (Å)'`. This will cause a runtime `KeyError`. | High |
| `src/mbe_automation/ml/dataset.py` | 1-27 | **Dead/Legacy Code**: The file contains functions (`get_vacuum_energies`, `process_trajectory`) that operate directly on ASE Atoms and use `tqdm`, bypassing the project's `Structure` API and standard logging. This appears to be deprecated prototyping code. | Medium |
| `src/mbe_automation/configs/training.py` | 51 | **Inconsistent Parameter Type**: `PhononSampling.temperature_K` is typed as a single `float`, whereas `MDSampling` supports arrays for looping over conditions. This restricts `PhononSampling` to single-temperature execution. | Low |
