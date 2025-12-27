# Calculators

The `mbe_automation.calculators` module provides interfaces to various computational backends, including machine learning potentials (MACE), density functional theory (PySCF/GPU4PySCF), and semi-empirical methods (DFTB+).

## MACE

The `MACE` class wraps the `mace-torch` calculator, providing automatic device selection (CPU/CUDA) and serialization support for Ray actors.

### Adjustable parameters

Location: `mbe_automation.calculators.mace.MACE`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str` | - | Path to the MACE model file (`.model`). |
| `device` | `str` \| `None` | `None` | Computation device (e.g., "cpu", "cuda"). Auto-detected if `None`. |
| `head` | `str` | `"default"` | Name of the output head to use.† |

† *Some models, such as `mace-mh-1.model`, require specifying a head (e.g., `head="omol"`).*

### Code Example

```python
import mbe_automation
from mbe_automation.calculators import MACE

# Load a MACE model
# Note: mace-mh-1 requires the "omol" head
calc = MACE(model_path="~/models/mace-mh-1.model", head="omol")

# Load a structure
structure = mbe_automation.Structure.from_xyz_file("structure.xyz")

# Run calculation
# This updates the structure in-place with energies and forces
structure.run_model(calc)

print(f"Potential Energy: {structure.E_pot} eV/atom")
print(f"Forces:\n{structure.forces}")
```

## PySCF (DFT & HF)

The `PySCFCalculator` provides an interface to PySCF (CPU) and GPU4PySCF (GPU) for Hartree-Fock and DFT calculations. It is designed to be **stateless**, meaning the underlying backend and system are re-initialized for every `calculate` call. This allows a single calculator instance to be used across different atomic configurations (e.g., during delta learning training set generation).

The module provides factory functions `DFT` and `HF` to simplify initialization.

### Adjustable parameters

Location: `mbe_automation.calculators.pyscf.DFT` and `mbe_automation.calculators.pyscf.HF`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `str` | `"r2scan-d4"` | (DFT only) The density functional method. Supported methods include `wb97m-v`, `wb97x-d3`, `b3lyp-d4`, `r2scan-d4`, etc. |
| `basis` | `str` | `"def2-tzvp"` | Basis set (e.g., `def2-svp`, `def2-tzvpp`). |
| `kpts` | `list[int]` \| `None` | `None` | k-point mesh for periodic calculations (e.g., `[2, 2, 2]`). If `None`, calculation is treated as isolated/molecule. |
| `density_fit` | `bool` | `True` | Whether to use density fitting (RI approximation). |
| `auxbasis` | `str` \| `None` | `None` | Auxiliary basis set for density fitting. |
| `multigrid` | `bool` | `False` | Whether to use multigrid algorithm for DFT on PBC (GPU4PySCF feature). |
| `verbose` | `int` | `0` | Verbosity level for PySCF output. |
| `max_memory_mb` | `int` \| `None` | `None` | Maximum memory usage in MB. Auto-detected if `None`. |

### Code Example

```python
import mbe_automation
from mbe_automation.calculators import DFT, HF

# Initialize a DFT calculator (e.g., r2scan-d4 / def2-tzvp)
dft_calc = DFT(model_name="r2scan-d4", basis="def2-tzvp")

# Initialize a Hartree-Fock calculator
hf_calc = HF(basis="def2-tzvpp")

# Load a structure
structure = mbe_automation.Structure.from_xyz_file("molecule.xyz")

# Run DFT
structure.run_model(dft_calc)
print(f"DFT Energy: {structure.E_pot} eV/atom")

# Run HF
structure.run_model(hf_calc)
print(f"HF Energy: {structure.E_pot} eV/atom")
```

## DFTB+ (Semi-empirical)

The `DFTBCalculator` wraps the ASE `Dftb` calculator. Like the PySCF calculator, it is implemented to be **stateless**. Element-specific parameters (like Slater-Koster files and Hubbard derivatives) are applied dynamically based on the system passed to `calculate`.

Convenience factory functions are provided for common methods.

### Factory Functions

| Function | Description |
| :--- | :--- |
| `mbe_automation.calculators.dftb.GFN1_xTB` | GFN1-xTB method. |
| `mbe_automation.calculators.dftb.GFN2_xTB` | GFN2-xTB method. |
| `mbe_automation.calculators.dftb.DFTB_Plus_MBD` | DFTB+ with MBD dispersion (3ob-3-1 parameters). |
| `mbe_automation.calculators.dftb.DFTB3_D4` | DFTB3 with D4 dispersion (3ob-3-1 parameters). |

### Adjustable parameters

Location: `mbe_automation.calculators.dftb.DFTBCalculator`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `work_dir` | `Path` | `Path(".")` | Directory where DFTB+ input/output files are written. |

*Note: Most physics parameters are hardcoded in the factory functions or determined by the chemical system.*

### Code Example

```python
import mbe_automation
from mbe_automation.calculators import GFN2_xTB, DFTB3_D4
from pathlib import Path

# Create a GFN2-xTB calculator
# It writes files to the current directory by default
calc_xtb = GFN2_xTB()

# Create a DFTB3-D4 calculator
# Specify a work directory to avoid file conflicts in parallel runs
calc_dftb = DFTB3_D4()
calc_dftb.model_independent_params["directory"] = str(Path("./dftb_work").resolve())

# Load a structure
structure = mbe_automation.Structure.from_xyz_file("crystal.xyz")

# Run Calculation
structure.run_model(calc_xtb)
print(f"GFN2-xTB Energy: {structure.E_pot} eV/atom")
```
