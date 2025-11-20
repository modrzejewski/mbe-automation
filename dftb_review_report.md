
# DFTB+ Backend Review Report

This report compiles findings from the review of the new DFTB+ backend implementation in the `quasi_harmonic` workflow. Issues are ranked by severity.

| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/configs/quasi_harmonic.py` | 50 | The `relax_input_cell` parameter in `FreeEnergy` is redundant and conflicts with `Minimum.cell_relaxation` in the `relaxation` sub-config. This creates two sources of truth: one used for logic/labeling (`FreeEnergy`) and one used for actual execution (`Minimum`). This leads to misleading logs (e.g., claiming "opt:atoms,shape,V" while performing "constant_volume" relaxation) and potential runtime errors. | **High** |
| `src/mbe_automation/calculators/dftb.py` | 165 | The `relax` function executes the DFTB+ driver in the current working directory (CWD) and hardcodes the output file name `geo_end.gen`. This causes file pollution in the project root, prevents parallel execution (race conditions on `geo_end.gen`), and ignores the workflow's `work_dir` setting. | **High** |
| `src/mbe_automation/workflows/quasi_harmonic.py` | 249 | Inside the thermal expansion loop, if `eos_sampling="volume"`, the code manually sets `optimizer.cell_relaxation = "constant_volume"` without checking the backend. Since DFTB+ does not support constant volume relaxation, this causes a crash deep in `structure.relax.crystal` at runtime. While `recommended` configs try to avoid this, manual configuration can easily trigger it. | **High** |
| `src/mbe_automation/configs/structure.py` | 113 | Typo in the type hint: `Litaral` instead of `Literal`. | Low |
