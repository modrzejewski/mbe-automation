# Installation

This guide provides step-by-step instructions for installing `mbe-automation` on a Linux-based system, such as a personal laptop or a compute cluster.

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
The repository uses git submodules for external dependencies such as DFTB+ parameter files.

Download these dependencies by running:

```bash
git submodule update --init --recursive
```

Failure to run this command will result in errors, such as missing parameter files during DFTB calculations.

## 3. Install the Program

With the virtual environment active, install the program and its dependencies in editable mode using `pip`. The `-e` flag allows you to make changes to the source code without needing to reinstall the package.

```bash
pip install -e .
```

This command will read the `pyproject.toml` file and install all the required dependencies. The installation process may take several minutes.

> **Note on Editable Mode**
> The `-e` flag installs the package in "editable" mode. This means that any changes you make to the source code will be immediately available without needing to reinstall the package. If you pull the latest changes from the GitHub repository, you will have access to the newest version of the program. However, if the required libraries in `pyproject.toml` have changed, you will need to re-run `pip install -e .` in your virtual environment to ensure all dependencies are up to date.

Once the installation is complete, the `mbe-automation` program is ready to use.

## 4. Supported Methods

The `mbe-automation` package supports several calculation methods. While some dependencies are installed automatically via `pip`, others (like binary executables) must be installed separately and made available in your system's PATH.

| Method | Required Software/Library |
| :--- | :--- |
| MACE | `mace-torch` (Python package) |
| DFTB+MBD, DFTB3-D4 | `dftb+` (Executable) |
| GFN1-xTB, GFN2-xTB | `dftb+` (Executable) |
