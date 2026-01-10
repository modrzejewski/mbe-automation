# Calculators

The `mbe_automation.calculators` module provides interfaces to various computational backends, including machine learning potentials (MACE), density functional theory (PySCF/GPU4PySCF), and semi-empirical methods (DFTB+).

## Why use `mbe_automation.calculators`?

While these calculators inherit from the standard ASE `Calculator` interface, they are enhanced with specific features required by the `mbe_automation` workflow:

1.  **Level of Theory Tracking:** Each calculator instance has a `level_of_theory` property (string). This identifier is used to tag computed energies and forces when they are stored in the HDF5 dataset (under `ground_truth`), ensuring data provenance and enabling multi-fidelity workflows.
2.  **Multi-GPU Parallelization:** The MACE and PySCF-based calculators implement specialized serialization methods that allow `run` to distribute calculations across multiple GPUs using Ray. This parallelization works as a black box for the user: you simply allocate the necessary resources (e.g., via SLURM) and execute `run` on your structures.

## Table of Contents

*   [MACE](#mace)
*   [DeltaMACE](#deltamace)
*   [PySCF (DFT & HF)](#pyscf-dft--hf)
*   [DFTB+ (Semi-empirical)](#dftb-semi-empirical)

## MACE

The `MACE` class wraps the `mace-torch` calculator, providing automatic device selection (CPU/CUDA) and serialization support for Ray actors.

### Adjustable parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str` | - | Path to the MACE model file. |
| `head` | `str` | `"Default"` | Name of the readout head to use.† |

† *Some models, such as `mace-mh-1.model`, require specifying a readout head (e.g., `head="omol"`).*

### Code Example

```python
from mbe_automation import Structure
from mbe_automation.calculators import MACE

# Load a MACE model
# Note: mbe-mh-1 requires specifying the readout head.
# In our tests we have found that "omol" works well for molecular
# crystals.
calc = MACE(model_path="~/models/mace-mh-1.model", head="omol")

# Load a structure
structure = Structure.from_xyz_file("structure.xyz")

# Run calculation
# This updates the structure in-place with energies and forces
# The results are stored in the ground_truth attribute
structure.run(calc)

# Retrieve results using the calculator's level of theory
energy = structure.ground_truth.energies[calc.level_of_theory]
forces = structure.ground_truth.forces[calc.level_of_theory]

print(f"Potential Energy: {energy} eV/atom")
print(f"Forces:\n{forces}")
```

## DeltaMACE

The `DeltaMACE` class implements a delta-learning model. It takes a list of MACE models, where the first is treated as the baseline and the rest are additive corrections (deltas).

### Adjustable parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_paths` | `list[str]` | - | List of paths to MACE model files. The first model is the baseline, and subsequent models are deltas. |
| `head` | `str` | `"Default"` | Name of the readout head to use for all models. |

### Code Example

```python
from mbe_automation import Structure
from mbe_automation.calculators import DeltaMACE

# Define paths to the baseline and delta models
baseline_model = "~/models/mace-mh-1.model"  # e.g., trained on DFTB
delta_model = "~/models/delta_dft.model"      # e.g., trained on DFT - DFTB

# Initialize the delta-learning calculator
# Note: mace-mh-1 requires specifying the readout head
calc = DeltaMACE(
    model_paths=[baseline_model, delta_model],
    head="omol"
)

# Load a structure
structure = Structure.from_xyz_file("structure.xyz")

# Run calculation
structure.run(calc)

# Retrieve results
# The level_of_theory will be baseline_architecture+Δ
energy = structure.ground_truth.energies[calc.level_of_theory]
forces = structure.ground_truth.forces[calc.level_of_theory]

print(f"Delta-learning Energy: {energy} eV/atom")
print(f"Forces:\n{forces}")
```

## PySCF (DFT & HF)

The `PySCFCalculator` provides an interface to PySCF (CPU) and GPU4PySCF (GPU) for Hartree-Fock and DFT calculations. It is designed to be stateless, meaning the underlying backend and system are re-initialized for every `calculate` call. This allows a single calculator instance to be used across different atomic configurations (e.g., during delta learning training set generation).

The module provides factory functions `DFT` and `HF` to simplify initialization.

### Adjustable parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `str` | `"r2scan-d4"` | (DFT only) The density functional method. See "Supported DFT Methods" below. |
| `basis` | `str` | `"def2-tzvp"` | Basis set. See "Supported Basis Sets" below. |
| `kpts` | `list[int]` \| `None` | `None` | k-point mesh for periodic calculations (e.g., `[2, 2, 2]`). If `None`, periodic calculations are performed at the Gamma point. Has no effect on finite systems. |
| `density_fit` | `bool` | `True` | Whether to use density fitting (RI approximation). |
| `auxbasis` | `str` \| `None` | `None` | Auxiliary basis set for density fitting. |
| `verbose` | `int` | `0` | Verbosity level for PySCF output. |
| `max_memory_mb` | `int` \| `None` | `None` | Maximum memory usage in MB. Auto-detected if `None`. |

### Supported DFT Methods

| Range-Separated Hybrids | Global Hybrid GGAs | Pure GGAs | Pure Meta-GGAs |
| :--- | :--- | :--- | :--- |
| `wb97m-v` | `b3lyp-d3` | `pbe-d3` | `r2scan-d4` |
| `wb97x-d3` | `b3lyp-d4` | `pbe-d4` | |
| `wb97x-d4` | `pbe0-d3` | | |
| | `pbe0-d4` | | |

### Supported Basis Sets

| Double Zeta | Triple Zeta | Quadruple Zeta |
| :--- | :--- | :--- |
| `def2-svp` | `def2-tzvp` | `def2-qzvp` |
| `def2-svpd` | `def2-tzvpp` | `def2-qzvpp` |
| | `def2-mtzvpp` | `def2-qzvpd` |
| | `def2-tzvpd` | `def2-qzvppd` |
| | `def2-tzvppd` | |

### Code Example

```python
from mbe_automation import Structure
from mbe_automation.calculators import DFT, HF

# Initialize a DFT calculator (e.g., r2scan-d4 / def2-tzvp)
dft_calc = DFT(model_name="r2scan-d4", basis="def2-tzvp")

# Initialize a Hartree-Fock calculator
hf_calc = HF(basis="def2-tzvpp")

# Load a structure
structure = Structure.from_xyz_file("molecule.xyz")

# Run DFT
structure.run(dft_calc)
energy_dft = structure.ground_truth.energies[dft_calc.level_of_theory]
print(f"DFT Energy: {energy_dft} eV/atom")

# Run HF
structure.run(hf_calc)
energy_hf = structure.ground_truth.energies[hf_calc.level_of_theory]
print(f"HF Energy: {energy_hf} eV/atom")
```

## DFTB+ (Semi-empirical)

The `DFTBCalculator` wraps the ASE `Dftb` calculator. Like the PySCF calculator, it is implemented to be **stateless**. Element-specific parameters (like Slater-Koster files and Hubbard derivatives) are applied dynamically based on the system passed to `calculate`.

Convenience factory functions are provided for common methods.

### Factory Functions

| Function | Description |
| :--- | :--- |
| `GFN2_xTB` | GFN2-xTB method. |
| `DFTB3_D4` | DFTB3 with D4 dispersion (3ob-3-1 parameters). |

The factory functions can be imported from `mbe_automation.calculators` (e.g. `from mbe_automation.calculators import GFN2_xTB`).

Most physics parameters are hardcoded in the factory functions or determined by the chemical system.

### Code Example

```python
from mbe_automation import Structure
from mbe_automation.calculators import GFN2_xTB, DFTB3_D4

calc_xtb = GFN2_xTB()
calc_dftb = DFTB3_D4()

# Load a structure
structure = Structure.from_xyz_file("crystal.xyz")

# Run Calculation
structure.run(calc_xtb)
structure.run(calc_dftb)
energy_xtb = structure.ground_truth.energies[calc_xtb.level_of_theory]
energy_dftb = structure.ground_truth.energies[calc_dftb.level_of_theory]
print(f"GFN2-xTB Energy: {energy_xtb} eV/atom")
print(f"DFTB3-D4 Energy: {energy_dftb} eV/atom")
```
