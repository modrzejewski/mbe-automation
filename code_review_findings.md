# Code Review Findings: `machine-learning` branch

## Overview
The `machine-learning` branch introduces changes to support Delta Learning workflows, including a new `level_of_theory` parameter (replacing `delta_learning`) and a unified `export_to_mace` method for generating training sets. A new `Dataset` class has been added to facilitate batch processing of structures.

## Findings

| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/api/classes.py` | 105 | The `Structure.export_to_mace` method signature includes a mandatory `dataset` argument that is unused and misleading. The method internally creates a dataset containing only `self`. | Medium |
| `src/mbe_automation/api/classes.py` | 117 | `Structure.export_to_mace` passes an undefined variable `reference_molecules` (plural) to `_export_to_mace`, whereas the argument is named `reference_molecule` (singular). This will cause a `NameError`. | High |
| `src/mbe_automation/api/classes.py` | 373 | The `FiniteSubsystem.export_to_mace` method signature includes a mandatory `dataset` argument that is unused and misleading. | Medium |
| `src/mbe_automation/api/classes.py` | 385 | `FiniteSubsystem.export_to_mace` passes an undefined variable `reference_molecules` (plural) to `_export_to_mace`. This will cause a `NameError`. | High |
| `src/mbe_automation/api/classes.py` | 664 | The function `_run_model` attempts to access `mbe_automation.storage.core.DeltaTargetBaseline`. However, `mbe_automation.storage` does not export `core` as an attribute, likely causing an `AttributeError`. It should use `from mbe_automation.storage.core import DeltaTargetBaseline` or rely on `DATA_FOR_TRAINING` imports if available, or fix the import structure. | High |

## Suggestions
1.  **API Cleanup**: Remove the `dataset` argument from `Structure.export_to_mace` and `FiniteSubsystem.export_to_mace`. These methods should only export the object they belong to.
2.  **Bug Fix**: Correct the typo `reference_molecules` -> `reference_molecule` in both `export_to_mace` methods.
3.  **Import Fix**: In `src/mbe_automation/api/classes.py`, add `from mbe_automation.storage.core import DeltaTargetBaseline` to ensure the class is available, or ensure `mbe_automation.storage` properly exposes `core`.
