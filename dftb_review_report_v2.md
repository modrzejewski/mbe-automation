
# DFTB+ Backend Review Report (v2)

This report updates the findings after the recent commits to the `machine-learning` branch.

## Resolved Issues
*   **Typo:** The `Litaral` typo in `src/mbe_automation/configs/structure.py` has been fixed.
*   **DFTB+ Constant Volume Crash:** New `__post_init__` checks in `FreeEnergy` and `Minimum` correctly prevent the use of `eos_sampling="volume"` or `cell_relaxation="constant_volume"` with the DFTB+ backend, avoiding the runtime crash.

## Persisting Issues

| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/configs/quasi_harmonic.py` | 50 | **Logic Conflict:** The `relax_input_cell` parameter in `FreeEnergy` remains redundant and conflicting with `Minimum.cell_relaxation`. <br><br>Example scenario: Setting `relax_input_cell="full"` but `relaxation.cell_relaxation="only_atoms"` will result in the workflow labeling the structure as `crystal[opt:atoms,shape,V]` (implying volume relaxation) while only performing atomic position relaxation. The logs and output filenames will contradict the actual physical simulation performed. | **High** |
| `src/mbe_automation/calculators/dftb.py` | 208 | **File Pollution / Race Condition:** The `relax` function still executes the DFTB+ driver in the current working directory (CWD) and relies on the hardcoded output file `geo_end.gen`. <br><br>Risks:<br>1. **Concurrency:** Running multiple workflows or parallel relaxations (e.g., different temperatures/pressures) in the same root directory will cause race conditions where one process overwrites `geo_end.gen` of another.<br>2. **Pollution:** It leaves `geo_end.gen` (and potential `.out` files) in the project root instead of the designated `work_dir`. | **High** |
