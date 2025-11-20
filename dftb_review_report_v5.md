
# DFTB+ Backend Review Report (v5)

This report updates the findings after the latest commits to the `machine-learning` branch (commit `d9175d0`).

## Resolved Issues
*   **Typo:** `Litaral` -> `Literal` in `src/mbe_automation/configs/structure.py` (Resolved).
*   **DFTB+ Constant Volume Crash:** `__post_init__` validation added (Resolved).
*   **Config Conflict:** `relax_input_cell` removed from `FreeEnergy` config (Resolved).
*   **AttributeError Crash:** The workflow code in `src/mbe_automation/workflows/quasi_harmonic.py` has been updated to correctly use `config.relaxation.cell_relaxation` instead of the removed `config.relax_input_cell`. The logic now properly respects the single source of truth for relaxation parameters.

## Persisting Issues

| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/calculators/dftb.py` | 208 | **File Pollution / Race Condition:** The `relax` function continues to execute in the current working directory and relies on `geo_end.gen`. There is no implementation of temporary directory handling or `work_dir` usage in the calculator wrapper. This remains a critical concurrency risk. | **High** |
