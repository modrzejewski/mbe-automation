# Migration from MACE to UMA and UMA Nitro (`fairchem-core>=2.24.0`)

This document summarizes the dependency, calculator, and workflow changes required to replace MACE with the Universal Model for Atoms (UMA 1.2.1 / `uma-s-1p2p1`) and the optional UMA Nitro inference interface via `fairchem-core>=2.24.0`.

---

## 1. Dependency Conflicts Between MACE and Fairchem

`mace-torch` and `fairchem-core` cannot coexist in a single Python environment due to incompatible version constraints on core equivariant and tensor libraries:

| Package | `mace-torch` (`0.3.16`) | `fairchem-core` (`>=2.24.0`) | Cause of Conflict |
| :--- | :--- | :--- | :--- |
| `e3nn` | `e3nn==0.4.4` | `e3nn>=0.5` | MACE checkpoints depend on Clebsch-Gordan phase conventions fixed in `e3nn 0.4.4`. `fairchem-core` requires API changes introduced in `e3nn 0.5+`. |
| `torch` | `torch>=1.12` (previously pinned to `==2.9.1`) | `torch~=2.13.0` (`2.22+`) | `fairchem-core` pins minor PyTorch versions (`~=2.13.0`) to match precompiled custom CUDA/Warp/CuTeDSL extensions (`nvalchemi-toolkit-ops`, `nvidia-cutlass-dsl`). A hard pin of `torch==2.9.1` prevents `fairchem-core` installation. |

### Environment Configuration Changes

* **[`pixi.toml`](../pixi.toml)**:
  * Relaxed `pytorch = "==2.9.1"` to `pytorch = ">=2.8.0"`.
  * Removed `mace-torch = { path = "src/mace" }` and `graph-longrange = { path = "src/graph_electrostatics" }` from `[pypi-dependencies]`.
  * Added `fairchem-core = ">=2.24.0"` to `[pypi-dependencies]`.
* **[`pyproject.toml`](../pyproject.toml)**:
  * Relaxed `"torch==2.9.1"` to `"torch>=2.8.0"` and added `"fairchem-core>=2.24.0"` to core `dependencies`.
  * Removed the `mace` optional dependency group while retaining `uma = ["fairchem-core>=2.24.0"]`.

---

## 2. UMA Calculator & UMA Nitro Interface (`src/mbe_automation/calculators/uma.py`)

### 2.1 Local Checkpoint Loading & Latest UMA 1.2.1 Models

[`UMA`](../src/mbe_automation/calculators/uma.py) supports loading weights from a local checkpoint file (`model_path`) as well as by pretrained model identifier (`model_name`, defaulting to `"uma-s-1p2p1"`):

| Model / Checkpoint | Tasks | Description | UMA Nitro Support |
| :--- | :--- | :--- | :--- |
| `uma-s-1p2p1` (`uma-s-1p2p1.pt`) | `omc`, `omol`, `omat`, `odac`, `oc20`, `oc22`, `oc25` | Default UMA-S 1.2.1 universal model | Yes |
| `uma-s-1p2p1-ft-d4rev` (`uma-s-1p2p1-ft-d4rev.pt`) | `omol` | UMA-S 1.2.1 fine-tuned on $\omega\text{B97M-D4rev}$ | Yes |
| `uma-s-1p2p1-ft-r2scan` (`uma-s-1p2p1-ft-r2scan.pt`) | `omat` | UMA-S 1.2.1 fine-tuned on $\text{r}^2\text{SCAN}$ (MAD-1.6) | Yes |
| `uma-s-1p2` (`uma-s-1p2.pt`) | `omc`, `omol`, `omat`, `odac`, `oc20`, `oc22`, `oc25` | UMA-S 1.2 model | Automatic fallback to `multigpu` |
| `uma-m-1p1` (`uma-m-1p1.pt`) | `omc`, `omol`, `omat`, `odac`, `oc20` | UMA-M 1.1 medium model | Automatic fallback to `multigpu` |

```python
from mbe_automation.calculators import UMA

# Load from a local UMA 1.2.1 checkpoint file (64-bit precision enforced by default;
# set nitro=True to enable 32-bit UMA Nitro inference)
calc = UMA(
    model_path="/path/to/uma-s-1p2p1.pt",
    task_name="omc",
    nitro=False,
)
```

When `model_path` (or `UMA_MODEL_PATH` in the environment) points to a local `.pt`, `.pth`, or `.ckpt` file, `UMA` loads the predictor using `fairchem.core.units.mlip_unit.load_predict_unit`:

```python
predictor = load_predict_unit(
    str(self.model_path),
    inference_settings=config,
    overrides=overrides,
    device=self.device,
    workers=workers,
)
```

For single-task fine-tuned checkpoints (`uma-s-1p2p1-ft-d4rev.pt` with `omol` or `uma-s-1p2p1-ft-r2scan.pt` with `omat`), `UMA` automatically selects the checkpoint's sole task when `task_name` is left at its default value (`"omc"`).

### 2.2 Precision vs. UMA Nitro (`nitro` and `inference_settings`)

`UMA` defaults to `nitro=False`, which enforces 64-bit IEEE double precision (`"float64"`) required for tight crystal geometry relaxations (`max_force_on_atom_eV_A = 1.0E-4`) and finite-displacement harmonic phonon calculations. Setting `nitro=True` activates the 32-bit standalone `UMANitroModel` in `fairchem-core>=2.24.0` (custom CuTeDSL / Triton / CUDA graph execution with merged MOLE experts on CUDA capability 8.x–12.x GPUs). Because `UMANitroModel` kernels operate in 32-bit precision (`torch.float32` / `tf32=True`), `nitro=True` conflicts with 64-bit precision (`"float64"`).

To prevent multi-node and multi-worker Triton/TorchInductor cache race conditions on shared network home directories (`~/.triton/cache`), `calculators/uma.py` automatically isolates `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` in node-local temporary storage per host, SLURM job, and process ID when not already set in the environment.

Supported `inference_settings` presets include:

* `"float64"` (default when `nitro=False`): 64-bit IEEE double precision (`base_precision_dtype=torch.float64`, `tf32=False`, `activation_checkpointing=True`, `merge_mole=False`, `compile=False`), including a CPU determinant fallback (`_safe_det`) for `torch.det` on `float64` CUDA tensors.
* `"nitro"` / `"default"` (default when `nitro=True`): Runs UMA Nitro for `uma-s-1p2p1`, `uma-s-1p2p1-ft-d4rev`, and `uma-s-1p2p1-ft-r2scan` on supported NVIDIA GPUs, and automatically falls back to `"multigpu"` for other checkpoints (such as `uma-m-1p1.pt`) or devices.
* `"high-precision"`: Full IEEE FP32 precision (`tf32=False`) with merged MOLE weights (`merge_mole=True`) and `torch.compile=True`.
* `"multigpu"`: Merged MOLE weights and `torch.compile=True` with TF32 matrix multiplications on CUDA.
* `"batch"`: Unmerged and uncompiled FP32 mode with activation checkpointing (`merge_mole=False, compile=False, activation_checkpointing=True`) for heterogeneous inputs.

### 2.3 Ray Worker Serialization

`UMA.serialize()` returns `(UMA, {"model_path": ..., "model_name": self.model_name, "device": self.device, "task_name": self.task_name, "nitro": self.nitro, "inference_settings": self.inference_settings, "overrides": self.overrides, "workers": self.workers})`, allowing multi-GPU Ray workers in [`calculators/core.py`](../src/mbe_automation/calculators/core.py) to reconstruct calculators with identical local paths and inference settings.
