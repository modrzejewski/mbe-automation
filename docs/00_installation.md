# Installation

This guide provides step-by-step instructions for installing `mbe-automation` on a Linux-based system, such as a personal laptop or a compute cluster.

## 1. Create a Virtual Environment

It is highly recommended to install the program in a dedicated Python virtual environment to avoid conflicts with other packages.

First, create a directory for the project and navigate into it:

```bash
mkdir mbe-automation-project
cd mbe-automation-project
```

Next, create a new virtual environment. We'll name it `mbe_env` in this example:

```bash
python3 -m venv mbe_env
```

Activate the virtual environment:

```bash
source mbe_env/bin/activate
```

Your shell prompt should now be prefixed with `(mbe_env)`, indicating that the virtual environment is active.

## 2. Clone the Repository

Clone the `mbe-automation` repository from GitHub:

```bash
git clone https://github.com/your-username/mbe-automation.git
cd mbe-automation
```

*Note: Replace `your-username` with the appropriate GitHub username or organization.*

## 3. Install the Program

With the virtual environment active, install the program and its dependencies in editable mode using `pip`. The `-e` flag allows you to make changes to the source code without needing to reinstall the package.

```bash
pip install -e .
```

This command will read the `pyproject.toml` file and install all the required dependencies. The installation process may take several minutes.

Once the installation is complete, the `mbe-automation` program is ready to use.
