# Molecular Dynamics

- [Setup](#setup)
- [Configuration](#configuration)
- [Execution](#execution)
- [Details](#details)
- [Computational Bottlenecks](#computational-bottlenecks)
- [Complete Input Files](#complete-input-files)

This workflow is designed for performing molecular dynamics (MD) simulations to compute properties such as the sublimation energy of a molecular crystal.

## Setup

The initial setup requires importing the necessary modules and defining the system's structures and the MLIP calculator.

```python
import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"
work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("path/to/your/model.model"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)
```

## Configuration

The MD workflow is configured using the `Enthalpy` and `ClassicalMD` classes from `mbe_automation.configs.md`.

```python
md_config = mbe_automation.configs.md.Enthalpy(
    molecule=from_xyz_file(os.path.join(work_dir, xyz_molecule)),
    crystal=from_xyz_file(os.path.join(work_dir, xyz_solid)),
    calculator=mace_calc,
    temperature_K=298.15,
    pressure_GPa=1.0E-4,
    work_dir=os.path.join(work_dir, "properties"),
    dataset=os.path.join(work_dir, "properties.hdf5"),

    md_molecule=mbe_automation.configs.md.ClassicalMD(
        ensemble="NVT",
        time_total_fs=50000.0,
        time_step_fs=1.0,
        time_equilibration_fs=5000.0
    ),

    md_crystal=mbe_automation.configs.md.ClassicalMD(
        ensemble="NPT",
        time_total_fs=50000.0,
        time_step_fs=1.0,
        time_equilibration_fs=5000.0,
        supercell_radius=15.0,
    )
)
```

### Key Parameters for `ClassicalMD`:

| Parameter               | Description                                                                                                                              | Default Value     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `ensemble`              | Thermodynamic ensemble for the simulation ("NVT" or "NPT").                                                                          | "NVT"                 |
| `time_total_fs`         | Total simulation time in femtoseconds.                                                                                               | `50000.0`             |
| `time_step_fs`          | Time step for the integration algorithm.                                                                                             | `0.5`                 |
| `sampling_interval_fs`  | Interval for trajectory sampling.                                                                                                   | `50.0`                |
| `time_equilibration_fs` | Initial period of the simulation discarded to allow the system to reach equilibrium.                                         | `5000.0`              |
| `nvt_algo`              | Thermostat algorithm for NVT simulations. "csvr" (Canonical Sampling Through Velocity Rescaling) is robust for isolated molecules. | "csvr"                |
| `npt_algo`              | Barostat/thermostat algorithm for NPT simulations.                                                                                   | "mtk_full"            |
| `thermostat_time_fs`    | Thermostat relaxation time.                                                                                                          | `100.0`               |
| `barostat_time_fs`      | Barostat relaxation time.                                                                                                            | `1000.0`              |
| `supercell_radius`      | Minimum point-periodic image distance in the supercell (Å).                                                                        | `25.0`                |

## Execution

The MD workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.workflows.md.run(md_config)
```

## Details

The `run` function in `mbe_automation/workflows/md.py` executes the following sequence of operations:

1.  **Molecule Simulation (NVT):**
    *   A molecular dynamics simulation is performed on the isolated `molecule` in the NVT ensemble using `mbe_automation.dynamics.md.core.run`.
    *   Trajectory data is processed by `mbe_automation.dynamics.md.data.molecule` to extract thermodynamic properties.

2.  **Crystal Simulation (NPT):**
    *   Supercell matrix for the `crystal` is determined using `mbe_automation.structure.crystal.supercell_matrix`.
    *   An MD simulation is performed on the crystal supercell in the NPT ensemble, again using `mbe_automation.dynamics.md.core.run`.
    *   Crystal's trajectory data is processed by `mbe_automation.dynamics.md.data.crystal`.

3.  **Sublimation Enthalpy Calculation:**
    *   Sublimation enthalpy is calculated by `mbe_automation.dynamics.md.data.sublimation`, which combines the results from the molecule and crystal simulations.

4.  **Data Storage:**
    *   All final results are compiled into a pandas DataFrame and saved to the HDF5 `dataset` file.

## Computational Bottlenecks

The primary factors influencing the computational cost of this workflow are:

*   **`time_total_fs`**: The total simulation time directly determines the number of integration steps, leading to a linear scaling of the computational cost.
*   **`supercell_radius`**: For the crystal simulation, the number of atoms in the supercell scales with the cube of this radius, significantly impacting the cost of each MD step.
*   **`time_step_fs`**: A smaller time step will increase the total number of steps required for a given `time_total_fs`, thus increasing the computational cost.

## Complete Input Files

### Python Script (`md.py`)

```python
import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"
work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("path/to/your/model.model"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

md_config = mbe_automation.configs.md.Enthalpy(
    molecule=from_xyz_file(os.path.join(work_dir, xyz_molecule)),
    crystal=from_xyz_file(os.path.join(work_dir, xyz_solid)),
    calculator=mace_calc,
    temperature_K=298.15,
    pressure_GPa=1.0E-4,
    work_dir=os.path.join(work_dir, "properties"),
    dataset=os.path.join(work_dir, "properties.hdf5"),

    md_molecule=mbe_automation.configs.md.ClassicalMD(
        ensemble="NVT",
        time_total_fs=50000.0,
        time_step_fs=1.0,
        sampling_interval_fs=50.0,
        time_equilibration_fs=5000.0
    ),

    md_crystal=mbe_automation.configs.md.ClassicalMD(
        ensemble="NPT",
        time_total_fs=50000.0,
        time_step_fs=1.0,
        sampling_interval_fs=50.0,
        time_equilibration_fs=5000.0,
        supercell_radius=15.0,
        supercell_diagonal=True
    )
)

mbe_automation.workflows.md.run(md_config)
```

### SLURM Script (`mace-gpu.py`)

```python
#!/usr/bin/env python3
#SBATCH --job-name="MACE"
#SBATCH -A pl0415-02
#SBATCH --partition=tesla
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1 --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=180gb

import os
import os.path
import sys
import subprocess

InpScript = "{inp_script}"
LogFile = "{log_file}"

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

virtual_environment = os.path.expanduser("~/.virtualenvs/compute-env")
virtual_environment = os.path.realpath(virtual_environment)
activate_env = os.path.realpath(os.path.join(virtual_environment, "bin", "activate"))
cmd = f"module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10 && . {{activate_env}} && python {{InpScript}}"

with open(LogFile, "w") as log_file:
    process = subprocess.Popen(cmd, shell=True, stdout=log_file,
                               stderr=subprocess.STDOUT, bufsize=1,
                               universal_newlines=True)
    process.communicate()
```
