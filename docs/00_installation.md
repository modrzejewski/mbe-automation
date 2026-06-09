# Installation


- [Quick Start: Most Frequent Scenarios](#quick-start-most-frequent-scenarios)
  - [Scenario A: Install with MACE and PySCF on GPU](#scenario-a-install-with-mace-and-pyscf-on-gpu)
  - [Scenario B: Install with UMA on GPU](#scenario-b-install-with-uma-on-gpu)
- [Detailed Step-by-Step Installation](#detailed-step-by-step-installation)
  - [1. Create a Virtual Environment](#1-create-a-virtual-environment)
  - [2. Clone the Repository](#2-clone-the-repository)
  - [3. Install the Program](#3-install-the-program)
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
git clone --recurse-submodules https://github.com/modrzejewski/mbe-automation.git
cd mbe-automation
pip install -e ".[mace,gpu-cuda12]"
```
*(Note: If you are on a CUDA 11 system, replace `gpu-cuda12` with `gpu-cuda11`)*

### Scenario B: Install with UMA on GPU

If you prefer using UMA instead of MACE. Note that as of April 12th, 2026, MACE and UMA cannot be installed in the same environment due to dependency conflicts.

```bash
mkdir -p ~/.virtualenvs
python3 -m venv ~/.virtualenvs/mbe_env
source ~/.virtualenvs/mbe_env/bin/activate
git clone --recurse-submodules https://github.com/modrzejewski/mbe-automation.git
cd mbe-automation
pip install -e ".[uma]"
```

## Detailed Step-by-Step Installation

### 1. Create a Virtual Environment

It is highly recommended to install the program in a dedicated Python virtual environment to avoid conflicts with other packages.

First, create a directory for your virtual environments if it doesn't already exist:

```bash
mkdir -p ~/.virtualenvs
cd ~/.virtualenvs
```

Next, create a new virtual environment. We'll name it `mbe_env` in this example:

```bash
python3 -m venv mbe_env
```

Activate the virtual environment:

```bash
source ~/.virtualenvs/mbe_env/bin/activate
```

Your shell prompt should now be prefixed with `(mbe_env)`, indicating that the virtual environment is active.

### 2. Clone the Repository

Navigate to the directory where you want to store the project and clone the `mbe-automation` repository from GitHub:

```bash
git clone --recurse-submodules https://github.com/modrzejewski/mbe-automation.git
cd mbe-automation
```

The repository depends on several git submodules located in `src/`:

| Submodule | Path |
| :--- | :--- |
| `mace` | `src/mace` |
| `graph_electrostatics` | `src/graph_electrostatics` |
| `nomore_ase` | `src/nomore_ase` |

The `--recurse-submodules` flag initializes and downloads them automatically. If you cloned without it, run:

```bash
git submodule update --init --recursive
```

### 3. Install the Program

With the virtual environment active, install the program and its dependencies in editable mode using `pip`. The `-e` flag allows you to make changes to the source code without needing to reinstall the package.

MACE and UMA are optional dependencies. Include them if needed.

> **Note on Compatibility**
> As of April 12th, 2026, MACE and UMA cannot be installed in the same environment due to incompatibilities in their dependencies. You must choose one or the other, or create separate virtual environments for each.

To install the base program along with MACE:
```bash
pip install -e ".[mace]"
```

To install the base program along with UMA:
```bash
pip install -e ".[uma]"
```

For CUDA 12 systems, also install gpu-cuda12 for GPU acceleration. For example, with MACE:

```bash
pip install -e ".[mace,gpu-cuda12]"
```

The installation process may take several minutes.

> **Note on Editable Mode**
> The `-e` flag installs the package in "editable" mode. This means that any changes you make to the source code will be immediately available without needing to reinstall the package. If you pull the latest changes from the GitHub repository, you will have access to the newest version of the program. However, if the required libraries in `pyproject.toml` have changed, you will need to re-run `pip install -e .` in your virtual environment to ensure all dependencies are up to date.

Once the installation is complete, the `mbe-automation` program is ready to use.

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

† These methods are available on both GPUs (via `gpu4pyscf`) and CPUs (via `pyscf`). The CPU implementation is significantly slower and should only be used for debugging. To use GPU, you must [install the optional `gpu-cuda` dependencies](#3-install-the-program) defined in `pyproject.toml` (e.g., `gpu-cuda12` for CUDA 12). See [Supported DFT Methods](01_api.md#pyscf-dft--hf) for a list of available functionals.
