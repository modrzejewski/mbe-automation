# Code Review Findings

## Summary

A review of the `run_model` implementation in `src/mbe_automation/storage/core.py`, its usages in `src/mbe_automation/workflows/training.py`, and the associated calculator modules revealed several critical issues. These include logic errors that could lead to data corruption, potential runtime crashes due to typos and unsafe imports, and dead code that requires removal.

## Critical Issues

### 1. Logic Error: `feature_vectors_type` Corruption in `_run_model`
**File:** `src/mbe_automation/storage/core.py`
**Location:** `_run_model` function (Lines 1324-1328)

**Description:**
The `structure.feature_vectors_type` attribute is updated unconditionally, regardless of whether feature vectors are actually being computed.

```python
        if average_over_atoms:
            structure.feature_vectors_type = "averaged_environments"
        else:
            structure.feature_vectors_type = "atomic"
```

If `run_model` is called with `feature_vectors_type="none"` (the default), `average_over_atoms` is `False`. The code above incorrectly sets `structure.feature_vectors_type` to `"atomic"`. However, no feature vectors are computed or stored.

**Consequence:**
When this structure is saved, the HDF5 group will have the attribute `feature_vectors_type="atomic"`, but the `feature_vectors` dataset will be missing. Subsequent attempts to read this structure using `Structure.read` (or `read_structure`) will crash because the reader expects the `feature_vectors` dataset to exist when the type is not "none".

**Recommendation:**
Wrap the assignment of `structure.feature_vectors_type` inside the `if feature_vectors:` block.

### 2. Syntax Error: Typo in `training.py`
**File:** `src/mbe_automation/workflows/training.py`
**Location:** Line 200 and 202

**Description:**
There are typos in the variable name for the features calculator.

```python
            if config.feautres_calculator is not None:
                s.cluster_of_molecules.run_model(
                    calculator=config.feautures_calculator,
```

**Consequence:**
This code will raise an `AttributeError` at runtime when executed, as `feautres_calculator` and `feautures_calculator` are not valid attributes of the configuration object (which presumably uses `features_calculator`).

**Recommendation:**
Correct the spelling to `features_calculator`.

### 3. Import Error: Dead Code Reference
**File:** `src/mbe_automation/calculators/__init__.py`
**Location:** Line 7 and 14

**Description:**
The module attempts to import `run_model` from `.batch`.

```python
from .batch import run_model
```

However, `src/mbe_automation/calculators/batch.py` does not define `run_model`. Furthermore, `batch.py` contains incomplete code and syntax errors.

**Consequence:**
Importing `mbe_automation.calculators` will raise an `ImportError` or `SyntaxError`, breaking the package.

**Recommendation:**
Remove `src/mbe_automation/calculators/batch.py` entirely (as it is dead code) and remove the import reference in `src/mbe_automation/calculators/__init__.py`.

### 4. Logic/Usability: Silent Failure for Non-MACE Calculators
**File:** `src/mbe_automation/storage/core.py`
**Location:** `_run_model` (Line 1316)

**Description:**
The code silently disables feature vector calculation if the calculator is not a `MACECalculator`, even if the user explicitly requested them.

```python
    if feature_vectors: feature_vectors = isinstance(calculator, MACECalculator)
```

**Consequence:**
The user may assume features were computed when they were not. While this prevents a crash on `get_descriptors`, a warning should be issued if the user explicitly requested features but they cannot be computed.

**Recommendation:**
Raise a `ValueError` or issue a warning if `feature_vectors` is `True` but the calculator does not support it (i.e., is not a `MACECalculator`).

### 5. Import Safety: Unsafe Top-Level Imports
**File:** `src/mbe_automation/storage/core.py`
**Location:** Lines 11-12

**Description:**
The module performs top-level imports of `mace` and `ase`.

```python
from mace.calculators import MACECalculator
from ase.calculators.calculator import Calculator as ASECalculator
```

**Consequence:**
If `mace` is not installed, the entire `storage` module (and by extension `mbe_automation`) will fail to import. Other parts of the codebase (e.g., `training.py`) attempt to handle `mace` availability gracefully with `try/except` blocks, but this is negated by the hard dependency in `core.py`.

**Recommendation:**
Use `typing.TYPE_CHECKING` for the type hints and import `MACECalculator` inside the methods or within a `try/except` block to make the dependency optional at runtime.

## Code Review Table

| File Path | Line Number | Description | Severity |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/storage/core.py` | 1324-1328 | `structure.feature_vectors_type` is overwritten unconditionally, causing data corruption (type set to "atomic" but no data). | **High** |
| `src/mbe_automation/workflows/training.py` | 200, 202 | Typos: `feautres_calculator` and `feautures_calculator`. | **High** |
| `src/mbe_automation/calculators/__init__.py` | 7 | Broken import: `run_model` does not exist in `.batch`. | **High** |
| `src/mbe_automation/storage/core.py` | 11 | Unsafe top-level import of `MACECalculator` makes the package unusable without MACE. | Medium |
| `src/mbe_automation/storage/core.py` | 1316 | Silent failure: Feature vectors requests are ignored for non-MACE calculators without warning. | Medium |
| `src/mbe_automation/calculators/batch.py` | All | Dead code with syntax errors. | Low |
