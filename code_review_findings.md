
# Code Review Findings: `machine-learning` branch

This document summarizes the findings from a review of the changes introduced in the `machine-learning` branch compared to the `main` branch. The review focused on syntax and logical errors.

## Summary

The `machine-learning` branch introduces a `run_model` method to the `Structure` class and a corresponding core implementation in `calculators/core.py`. It also updates the training workflows to use this new method.

A critical issue is the **circular dependency risk** and **missing explicit imports**. The file `src/mbe_automation/api/classes.py` uses `mbe_automation.calculators.run_model` without explicitly importing the `calculators` submodule. While it might work incidentally if `mbe_automation.calculators` is loaded elsewhere, relying on this is unsafe.

Additionally, there are consistency issues with the import of optional dependencies (like `mace`) in the core API.

## Findings Table

| File Path | Line Number | Description | Severity |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/api/classes.py` | 104 | **Missing Explicit Import**: The code calls `mbe_automation.calculators.run_model` but does not import `mbe_automation.calculators`. It relies on the package `__init__.py` or other modules to load `calculators`, which is fragile. Explicitly `import mbe_automation.calculators` to ensure availability. | High |
| `src/mbe_automation/api/classes.py` | 7 | **Hard Dependency on Optional Libraries**: The file imports `MACECalculator` and `ASECalculator` at the top level for type hinting. `MACECalculator` depends on `mace`, which is treated as optional in `workflows/training.py` (guarded by try/except). This hard import makes `mace` mandatory for the core API. Use `typing.TYPE_CHECKING` blocks for type hints to maintain optionality. | Medium |
| `src/mbe_automation/calculators/core.py` | 44 | **Potential Zero Division**: `structure.n_atoms` is used as a divisor (`/ structure.n_atoms`). If `structure.n_atoms` is 0, this will raise a `ZeroDivisionError` or produce `NaN`. While unlikely for valid structures, a check would be robust. | Low |
| `src/mbe_automation/calculators/core.py` | 47 | **Redundant Assertion**: `assert isinstance(calculator, MACECalculator)` is used inside a block guarded by `if compute_feature_vectors:`. The variable `compute_feature_vectors` (line 21) is defined as `(feature_vectors and isinstance(calculator, MACECalculator))`. Thus, the assertion is mathematically redundant, though harmless. | Low |
| `src/mbe_automation/workflows/training.py` | 74 | **Condition Complexity**: The condition `if (config.features_calculator is not None and config.feature_vectors_type != "none")` is repeated multiple times. This logic could be encapsulated or simplified for maintainability. | Low |

## Detailed Analysis

### 1. Missing Import in `api/classes.py`
The `Structure.run_model` method delegates to `mbe_automation.calculators.run_model`. However, `mbe_automation.calculators` is not imported in the file. Python does not automatically populate package namespaces with submodules unless they are explicitly imported or loaded by `__init__.py`. While `src/mbe_automation/__init__.py` does import `calculators`, circular dependencies or specific import orders could cause `api/classes.py` to be executed before `__init__.py` fully completes, leading to an `AttributeError`.

### 2. MACE Dependency
`src/mbe_automation/workflows/training.py` gracefully handles the absence of `mace`:
```python
try:
    from mace.calculators import MACECalculator
except ImportError:
    MACECalculator = None
```
In contrast, `src/mbe_automation/api/classes.py` performs a hard import:
```python
from mace.calculators import MACECalculator
```
This forces `mace` to be installed for the entire `api` module to load, negating the optionality logic in the workflow.
