# Installation

The program integrates several scientific codes into a unified workflow, including:

*   **MACE** and **UMA:** Optional MLIP models for energies and forces.
*   **phonopy:** Phonon calculations and vibrational properties.
*   **pymatgen:** Crystal structure analysis and manipulation.
*   **ASE:** Geometry relaxation and molecular dynamics simulations.
*   **PySCF** and **GPU4PySCF:** Mean-field electronic structure calculations.
*   **MRCC** and **beyond-rpa**: High-fidelity data points from correlated wave-function theory.

- [Create a Virtual Environment](#1-create-a-virtual-environment)
- [Clone the Repository](#2-clone-the-repository)
- [Install the Program](#3-install-the-program)
- [Supported Backends](#4-supported-backends)

## 1. Create a Virtual Environment

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

## 2. Clone the Repository

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

## 3. Install the Program

With the virtual environment active, install the program and its dependencies in editable mode using `pip`. The `-e` flag allows you to make changes to the source code without needing to reinstall the package.

Machine learning interatomic potentials (MLIPs) such as MACE and UMA are optional dependencies. You should explicitly include them if you plan to use them.

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

If you are using a system with CUDA 12, you should also install the optional `gpu-cuda12` dependencies. This includes packages like `cuequivariance` which provide necessary GPU acceleration and optimizations for this environment. For example, to install with MACE and CUDA 12:

```bash
pip install -e ".[mace,gpu-cuda12]"
```

The installation process may take several minutes.

> **Note on Editable Mode**
> The `-e` flag installs the package in "editable" mode. This means that any changes you make to the source code will be immediately available without needing to reinstall the package. If you pull the latest changes from the GitHub repository, you will have access to the newest version of the program. However, if the required libraries in `pyproject.toml` have changed, you will need to re-run `pip install -e .` in your virtual environment to ensure all dependencies are up to date.

Once the installation is complete, the `mbe-automation` program is ready to use.

## 4. Supported Backends

The `mbe-automation` package supports several calculation methods. While some dependencies are installed automatically via `pip`, others (like binary executables) must be installed separately and made available in your system's PATH.

| Method | Required Software/Library |
| :--- | :--- |
| MACE | `mace-torch` (Python package) |
| UMA | `fairchem-core` (Python package) |
| DFT† | `gpu4pyscf`, `pyscf` (Python packages) |
| HF† | `gpu4pyscf`, `pyscf` (Python packages) |
| DFTB+MBD, DFTB3-D4 | `dftb+` (Executable) |
| GFN1-xTB, GFN2-xTB | `dftb+` (Executable) |

† These methods are available on both GPUs (via `gpu4pyscf`) and CPUs (via `pyscf`). The CPU implementation is significantly slower and should only be used for debugging. To use GPU, you must [install the optional `gpu-cuda` dependencies](#3-install-the-program) defined in `pyproject.toml` (e.g., `gpu-cuda12` for CUDA 12). See [Supported DFT Methods](02_calculators.md#supported-dft-methods) for a list of available functionals.
