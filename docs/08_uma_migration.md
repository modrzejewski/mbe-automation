# Migration from MACE to UMA (`uma-m-1p1`)

This document summarizes the dependency, calculator, and workflow changes required to replace MACE with the Universal Model for Atoms (`uma-m-1p1`) via `fairchem-core`.

---

## 1. Dependency Conflicts Between MACE and Fairchem

`mace-torch` and `fairchem-core` cannot coexist in a single Python environment due to incompatible version constraints on core equivariant and tensor libraries:

| Package | `mace-torch` (`0.3.16`) | `fairchem-core` (`>=2.19.0`) | Cause of Conflict |
| :--- | :--- | :--- | :--- |
| `e3nn` | `e3nn==0.4.4` | `e3nn>=0.5` | MACE checkpoints depend on Clebsch-Gordan phase conventions fixed in `e3nn 0.4.4`. `fairchem-core` requires API changes introduced in `e3nn 0.5+`. |
| `torch` | `torch>=1.12` (previously pinned to `==2.9.1`) | `torch~=2.8.0` (`2.19`–`2.21`) or `torch~=2.13.0` (`2.22+`) | `fairchem-core` strictly pins minor PyTorch versions (`~=2.8.0` or `~=2.13.0`) to match precompiled custom CUDA/Warp extensions (`nvalchemi-toolkit-ops`). A hard pin of `torch==2.9.1` prevents `fairchem-core` installation. |

### Environment Configuration Changes

* **[`pixi.toml`](../pixi.toml)**:
  * Relaxed `pytorch = "==2.9.1"` to `pytorch = ">=2.8.0"`.
  * Removed `mace-torch = { path = "src/mace" }` and `graph-longrange = { path = "src/graph_electrostatics" }` from `[pypi-dependencies]`.
  * Added `fairchem-core = ">=2.19.0"` to `[pypi-dependencies]`.
* **[`pyproject.toml`](../pyproject.toml)**:
  * Relaxed `"torch==2.9.1"` to `"torch>=2.8.0"` and added `"fairchem-core>=2.19.0"` to core `dependencies`.
  * Removed the `mace` optional dependency group while retaining `uma = ["fairchem-core>=2.19.0"]`.

---

## 2. UMA Calculator Implementation (`src/mbe_automation/calculators/uma.py`)

### 2.1 Local Checkpoint Loading (`uma-m-1p1.pt`)

Previously, [`UMA`](../src/mbe_automation/calculators/uma.py) only called `pretrained_mlip.get_predict_unit(model_name, ...)`, which queries the Hugging Face hub by model name, and the input template noted that loading from a local file path failed when `InferenceSettings` were omitted.

`UMA.__init__` now accepts both a local checkpoint path (`model_path`) and a pretrained model identifier (`model_name`, defaulting to `"uma-m-1p1"`):

```python
from mbe_automation.calculators import UMA

# Load from a local checkpoint file in 64-bit precision
calc = UMA(
    model_path="/path/to/uma-m-1p1.pt",
    task_name="omc",
)
```

When `model_path` (or a file path passed positionally or via `checkpoint_path`) points to a `.pt`, `.pth`, or `.ckpt` file, `UMA` loads the predictor using `fairchem.core.units.mlip_unit.load_predict_unit`:

```python
predictor = load_predict_unit(
    str(self.model_path),
    device=self.device,
    inference_settings=config,
)
```

If no local file is provided, `UMA` falls back to `pretrained_mlip.get_predict_unit(self.model_name, device=self.device, inference_settings=config)`. The `level_of_theory` tag is normalized using the checkpoint filename stem (e.g., `uma_uma-m-1p1_omc_head`) so dataset keys contain no path separators.

### 2.2 64-Bit Floating-Point Precision (`float64`)

Finite-difference phonon calculations, tight geometry relaxations (`max_force_on_atom_eV_A = 1.0E-4`), and normal-mode refinements require 64-bit numerical precision to eliminate force noise. `UMA` configures `InferenceSettings` with:

```python
config = InferenceSettings(
    tf32=False,
    activation_checkpointing=True,
    merge_mole=False,
    compile=False,
    external_graph_gen=False,
    internal_graph_gen_version=2,
    base_precision_dtype=torch.float64,
)
```

* `base_precision_dtype=torch.float64` and `tf32=False`: Enforces strict IEEE 754 double precision across graph construction and message-passing layers.
* `activation_checkpointing=True`: Bounds peak GPU memory usage during `float64` force evaluation on large phonon supercells.
* `merge_mole=False` and `compile=False`: Preserves exact per-configuration evaluation without MoLE weight-merging side effects across distinct supercell geometries.
* `external_graph_gen=False` and `internal_graph_gen_version=2`: Uses the internal on-device neighbor list generator required by `uma-m-1p1`.

### 2.3 H100 / CUDA `float64` `torch.det` Workaround

On NVIDIA Hopper (H100) GPUs with CUDA, `torch.det` on `torch.float64` CUDA tensors fails inside cuSOLVER during cell/rotation determinant checks in `fairchem-core`. [`src/mbe_automation/calculators/uma.py`](../src/mbe_automation/calculators/uma.py) wraps `torch.det` to evaluate `float64` CUDA determinants on the CPU and transfer the scalar result back to the target device:

```python
_original_det = torch.det


def _safe_det(
    tensor: torch.Tensor,
) -> torch.Tensor:
    if tensor.is_cuda and tensor.dtype == torch.float64:
        return _original_det(tensor.cpu()).to(tensor.device)
    return _original_det(tensor)
```

### 2.4 Read-Only `task_name` Property Fix

In `fairchem.core.calculate.ase_calculator.FAIRChemCalculator`, `task_name` is a read-only `@property` backed by `self._task_name`. Assigning `self.task_name = task_name` prior to `super().__init__(predictor, task_name=task_name, **kwargs)` previously raised `AttributeError: property 'task_name' of 'UMA' object has no setter`. Removing the redundant pre-assignment resolves this error while preserving `self.task_name` via the parent class property.

### 2.5 Ray Worker Serialization

`UMA.serialize()` returns `(UMA, {"model_path": self.model_path, "model_name": self.model_name, "device": self.device, "task_name": self.task_name})`, allowing multi-GPU Ray workers in [`calculators/core.py`](../src/mbe_automation/calculators/core.py) to reconstruct calculators initialized from local `.pt` files.

---

## 3. Workflow and Subsystem Adaptations

1. **Multi-GPU Parallelization and Calculator Dispatch ([`src/mbe_automation/calculators/core.py`](../src/mbe_automation/calculators/core.py))**:
   * Replaced `_is_mace` with `_is_uma` so `run_model` enables Ray multi-GPU parallelization when running `UMA` across multiple GPUs.
   * Updated `CALCULATORS` type union to `PySCFCalculator | DFTBCalculator | UMA`.
2. **Isolated Atom Reference Energies ([`src/mbe_automation/calculators/isolated_atoms.py`](../src/mbe_automation/calculators/isolated_atoms.py))**:
   * Replaced `MACE` with `UMA` in `SUPPORTED_CALCULATORS`.
   * Populated `isolated_atom.info["spin"] = ground_state_spin(z) + 1` (spin multiplicity $2S + 1$) and `isolated_atom.info["charge"] = 0` alongside `n_unpaired_electrons`, satisfying `FAIRChemCalculator` metadata requirements.
3. **Phonon Supercell GPU Memory Management ([`src/mbe_automation/dynamics/harmonic/core.py`](../src/mbe_automation/dynamics/harmonic/core.py))**:
   * Added `torch.cuda.empty_cache()` after each displaced supercell force evaluation in `phonons()` to prevent CUDA memory fragmentation and out-of-memory errors during 64-bit `uma-m-1p1` phonon calculations.
4. **Model Summary Display ([`src/mbe_automation/common/display.py`](../src/mbe_automation/common/display.py) & `workflows/`)**:
   * Replaced `mace_summary` with `uma_summary`, displaying `model_name`, `model_path` (when loaded from disk), `task_name`, parameter count, `dtype`, and `device` across `quasi_harmonic`, `md`, `refinement`, and `training` workflows.
5. **Input Templates, Config Presets, and Postprocessing**:
   * Updated [`src/mbe_automation/templates/inputs/uma(pbc)/quasi_harmonic.py`](../src/mbe_automation/templates/inputs/uma(pbc)/quasi_harmonic.py) to initialize `mbe_automation.calculators.UMA(model_path="{mlip_parameters}", task_name="omc")`.
   * Updated [`src/mbe_automation/configs/recommended.py`](../src/mbe_automation/configs/recommended.py), [`src/mbe_automation/single_point.py`](../src/mbe_automation/single_point.py), [`postprocessing/adps_md.py`](../postprocessing/adps_md.py), and test modules to use `UMA`.

