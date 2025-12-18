# Installation

- [Create a Virtual Environment](#1-create-a-virtual-environment)
- [Clone the Repository](#2-clone-the-repository)
- [Install the Program](#3-install-the-program)
- [Supported Methods](#4-supported-methods)

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
git clone https://github.com/modrzejewski/mbe-automation.git
cd mbe-automation
```
The repository uses git submodules for external dependencies. Download these dependencies by running:

```bash
git submodule update --init --recursive
```

Failure to run this command will result in errors, such as missing parameter files during DFTB calculations.

## 3. Install the Program

With the virtual environment active, install the program and its dependencies in editable mode using `pip`. The `-e` flag allows you to make changes to the source code without needing to reinstall the package.

```bash
pip install -e .
```

This command will read the `pyproject.toml` file and install all the required dependencies.

If you are using a system with CUDA 12, you should instead install the optional `gpu-cuda12` dependencies. This includes packages like `cuequivariance` which provide necessary GPU acceleration and optimizations for this environment.

```bash
pip install -e ".[gpu-cuda12]"
```

The installation process may take several minutes.

> **Note on Editable Mode**
> The `-e` flag installs the package in "editable" mode. This means that any changes you make to the source code will be immediately available without needing to reinstall the package. If you pull the latest changes from the GitHub repository, you will have access to the newest version of the program. However, if the required libraries in `pyproject.toml` have changed, you will need to re-run `pip install -e .` in your virtual environment to ensure all dependencies are up to date.

Once the installation is complete, the `mbe-automation` program is ready to use.

## 4. Supported Methods

The `mbe-automation` package supports several calculation methods. While some dependencies are installed automatically via `pip`, others (like binary executables) must be installed separately and made available in your system's PATH.

| Method | Required Software/Library |
| :--- | :--- |
| MACE | `mace-torch` (Python package) |
| DFT† | `gpu4pyscf`, `pyscf` (Python packages) |
| HF† | `gpu4pyscf`, `pyscf` (Python packages) |
| DFTB+MBD, DFTB3-D4 | `dftb+` (Executable) |
| GFN1-xTB, GFN2-xTB | `dftb+` (Executable) |

† Supported functionals (DFT): `wb97m-v`, `wb97x-d3/d4`, `b3lyp-d3/d4`, `pbe-d3/d4`, `r2scan-d4`. These methods are available on both GPUs (via `gpu4pyscf`) and CPUs (via `pyscf`). The CPU implementation is significantly slower and should only be used for debugging. To use GPU, you must [install the optional `gpu-cuda` dependencies](#3-install-the-program) defined in `pyproject.toml` (e.g., `gpu-cuda12` for CUDA 12).
