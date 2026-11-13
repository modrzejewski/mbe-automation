# Setup & Installation

- [Quick Start: Most Frequent Scenarios](#quick-start-most-frequent-scenarios)
  - [Scenario A: Install with MACE and PySCF on GPU](#scenario-a-install-with-mace-and-pyscf-on-gpu)
  - [Scenario B: Install with UMA on GPU](#scenario-b-install-with-uma-on-gpu)
- [Supported Backends](#supported-backends)

## Quick Start: Most Frequent Scenarios

> **Important:** Before running these commands on an HPC cluster, you may need to load the CUDA module. This is system-specific and beyond the scope of this manual, but typically looks like:

```bash
module load CUDA/12.8
```

### Scenario A: Install with MACE and PySCF on GPU

This is the recommended installation for full functionality on GPU-accelerated systems.

```bash
mkdir -p ~/.virtualenvs
python3 -m venv ~/.virtualenvs/mbe_env
source ~/.virtualenvs/mbe_env/bin/activate
git clone https://github.com/modrzejewski/mbe-automation.git
cd mbe-automation
pip install --no-cache-dir -e ".[mace,gpu-cuda12]"
```
*(Note: If you are on a CUDA 11 system, replace `gpu-cuda12` with `gpu-cuda11`)*

> **Note on Editable Mode**
> The -e flag installs the package in editable mode, making source code changes immediately available. If dependencies in pyproject.toml change, re-run the installation command with the appropriate extras to update them.

### Scenario B: Install with UMA on GPU

If you prefer using UMA instead of MACE. Note that as of April 12th, 2026, MACE and UMA cannot be installed in the same environment due to dependency conflicts.

```bash
mkdir -p ~/.virtualenvs
python3 -m venv ~/.virtualenvs/mbe_env
source ~/.virtualenvs/mbe_env/bin/activate
git clone https://github.com/modrzejewski/mbe-automation.git
cd mbe-automation
pip install --no-cache-dir -e ".[uma]"
```

## Supported Backends

The `mbe-automation` package supports several calculation methods. While some dependencies are installed automatically via `pip`, others (like binary executables) must be installed separately and made available in your system's PATH.

| Method | Required Software/Library |
| :--- | :--- |
| MACE | `mace-torch` (Python package) |
| UMA | `fairchem-core` (Python package) |
| DFT† | `gpu4pyscf`, `pyscf` (Python packages) |
| HF† | `gpu4pyscf`, `pyscf` (Python packages) |
| DFTB+MBD, DFTB3-D4 | `dftb+` (Executable) |
| GFN1-xTB, GFN2-xTB | `dftb+` (Executable) |

† These methods are available on both GPUs (via `gpu4pyscf`) and CPUs (via `pyscf`). The CPU implementation is significantly slower and should only be used for debugging. To use GPU, you must [install the optional `gpu-cuda` dependencies](#quick-start-most-frequent-scenarios) defined in `pyproject.toml` (e.g., `gpu-cuda12` for CUDA 12). See [Supported DFT Methods](01_api.md#pyscf-dft--hf) for a list of available functionals.
