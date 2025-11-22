# Code Review Findings: machine-learning branch

## 1. Summary of New Features

The `machine-learning` branch introduces significant capabilities focused on active learning, thermal expansion, and enhanced data management.

### Active Learning & Dataset Generation
*   **New Workflows:** Introduced `MDSampling` and `PhononSampling` workflows to generate diverse training data (structures, forces, energies) for machine learning potentials.
*   **Feature Vector Support:** Added support for computing and storing "feature vectors" (e.g., atomic environments) within MD simulations (`ClassicalMD`) and structure objects.
*   **Subsampling:** Implemented algorithms to subsample trajectories and finite subsystems based on feature vector diversity (e.g., `farthest_point_sampling`), enabling the selection of representative configurations for training.
*   **Finite Subsystems:** Added filters (`FiniteSubsystemFilter`) to extract and manage finite molecular clusters from periodic systems.

### Quasi-Harmonic Thermal Expansion
*   **EOS Sampling:** The `quasi_harmonic` workflow now supports thermal expansion calculations by sampling the Equation of State (EOS) via three methods: `pressure`, `volume`, and `uniform_scaling`.
*   **Thermodynamics:** Added functionality to fit EOS curves and calculate temperature-dependent equilibrium properties, including volume $V(T)$, free energy $F(T)$, bulk modulus $B(T)$, and thermal pressure.
*   **Data Filtering:** Added filters to exclude EOS points with imaginary modes or broken symmetry.

### Data Storage & Structures
*   **HDF5 Expansion:** Enhanced `mbe_automation.storage` to handle complex datasets, including `MolecularCrystal` (with cluster info), `Trajectory` (with drift energies), and `EOSCurves`.
*   **Configuration Refactoring:** Migrated configuration classes (`FreeEnergy`, `Enthalpy`, etc.) to use `dataclasses` with `kw_only=True` and factory methods (`recommended`) for better usability.

---

## 2. Code Review Findings

The following table lists the errors identified in the modified files.

| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/workflows/md.py` | 71 | Missing imports `numpy as np` and `numpy.typing as npt` for type hint `npt.NDArray[np.integer]`. This will cause a `NameError`. | **High** |
| `src/mbe_automation/storage/core.py` | 135 | Typo in argument name `key=keyk` inside `Structure.save`. | **High** |
| `src/mbe_automation/configs/execution.py` | 34 | Invalid type hint `Literal[KNOWN_MODELS]`. `Literal` requires literal values, not a list variable. Use `Literal["model1", "model2"]` or `Literal[*KNOWN_MODELS]` (Python 3.11+). | **High** |
| `src/mbe_automation/configs/training.py` | 20 | Mandatory field `calculator` follows optional field `force_constants_dataset` in `PhononSampling` dataclass. (Violates AGENTS.md and is a syntax error in older Python versions). | **High** |
| `src/mbe_automation/configs/md.py` | 120 | Mandatory field `calculator` follows optional fields `crystal` and `molecule` in `Enthalpy` dataclass. | **High** |
| `src/mbe_automation/configs/quasi_harmonic.py` | 24 | Mandatory field `calculator` follows optional field `molecule` in `FreeEnergy` dataclass. | **High** |
| `src/mbe_automation/dynamics/harmonic/core.py` | 170 | `interp_mesh` (float `150.0` from default config) is passed to `phonons.run_mesh`, which expects a list of 3 integers (e.g., `[10, 10, 10]`). No logic exists to convert the float "distance" to a mesh grid. | **High** |
| `templates/inputs/semiempirical(pbc)/training.py` | 46 | `mace_calc` is used in `FreeEnergy.recommended` but is not defined in the script. | **High** |
| `templates/inputs/semiempirical(pbc)/training.py` | 49 | `relax_input_cell` is passed to `FreeEnergy`, which is not a valid parameter for this class (it expects a `relaxation` object). | **High** |
