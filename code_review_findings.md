# Code Review Findings (Updated)

## Summary

A second review of the `run_model` implementation in `src/mbe_automation/storage/core.py` and its usages was conducted on the latest `machine-learning` branch. The critical syntax errors, dead code, and import issues identified in the previous review have been resolved.

One usability/logic issue remains regarding the handling of non-MACE calculators when feature vectors are requested.

## Remaining Issues

### 1. Logic/Usability: Silent Failure for Non-MACE Calculators
**File:** `src/mbe_automation/storage/core.py`
**Location:** `_run_model` (Line 1316)

**Description:**
The code disables feature vector calculation if the calculator is not a `MACECalculator`, even if the user explicitly requested them by setting `feature_vectors_type` to something other than "none".

```python
    if feature_vectors: feature_vectors = isinstance(calculator, MACECalculator)
```

**Consequence:**
If a user requests feature vectors (e.g., for subsampling) but provides a calculator that doesn't support them (e.g., `GFN2_xTB`), the function silently ignores the request. The `Structure` object retains `feature_vectors_type="none"` (or its previous value) and `feature_vectors=None`. This may lead to confusing downstream errors when the user expects feature vectors to be present (e.g., a `ValueError` in `Structure.subsample` saying "Execute run_model...").

**Recommendation:**
Issue a warning or raise a `ValueError` if `feature_vectors` is `True` (requested) but the calculator does not support it.

## Resolved Issues (Verified)

| File Path | Description | Status |
| :--- | :--- | :--- |
| `src/mbe_automation/storage/core.py` | Logic regarding `feature_vectors_type` assignment was re-evaluated and confirmed correct (guarded by `if feature_vectors:`). | **False Positive (Retracted)** |
| `src/mbe_automation/workflows/training.py` | Typos `feautres_calculator` corrected to `features_calculator`. | **Fixed** |
| `src/mbe_automation/calculators/batch.py` | Dead code file removed. | **Fixed** |
| `src/mbe_automation/calculators/__init__.py` | Broken import of `run_model` removed. | **Fixed** |
