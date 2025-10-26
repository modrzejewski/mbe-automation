# Molecular Dynamics

- [Setup](#setup)
- [NPT/NVT Propagation](#nptnvt-propagation)
- [Configuration details](#configuration-details)
- [Function Call Overview](#function-call-overview)
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

## NPT/NVT Propagation

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

The MD workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.workflows.md.run(md_config)
```

## Configuration details

### `Enthalpy` Class

**Location:** `mbe_automation.configs.md.Enthalpy`

| Parameter | Description | Default Value |
| --- | --- | --- |
| `molecule` | Initial, non-relaxed structure of the isolated molecule. An MD simulation is performed in the NVT ensemble to compute the average potential and kinetic energies. | - |
| `crystal` | Initial, non-relaxed crystal structure. An MD simulation is performed in the NPT ensemble to compute the average potential and kinetic energies, and the average volume. | - |
| `calculator` | MLIP calculator for energies and forces. | - |
| `temperature_K` | Target temperature (in Kelvin) for the MD simulation. | `298.15` |
| `pressure_GPa` | Target pressure (in GPa) for the MD simulation. | `1.0E-4` |
| `md_molecule` | An instance of `ClassicalMD` that configures the MD simulation for the isolated molecule. | `ClassicalMD()` |
| `md_crystal` | An instance of `ClassicalMD` that configures the MD simulation for the crystal. | `ClassicalMD()` |

### `ClassicalMD` Class

**Location:** `mbe_automation.configs.md.ClassicalMD`

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

## Function Call Overview

```
+--------------------------------------+
|             workflows.md             |
|                 run                  |
+--------------------------------------+
                   |
                   |
+--------------------------------------+
|           dynamics.md.core           |   Runs a molecular dynamics
|             run (molecule)           |   simulation for the isolated molecule.
+--------------------------------------+
                   |
                   |
+--------------------------------------+
|           dynamics.md.data           |   Processes the molecule's trajectory
|               molecule               |   to extract thermodynamic properties.
+--------------------------------------+
                   |
                   |
+--------------------------------------+
|         structure.crystal            |   Determines the supercell matrix
|            supercell_matrix          |   for the crystal simulation.
+--------------------------------------+
                   |
                   |
+--------------------------------------+
|           dynamics.md.core           |   Runs a molecular dynamics
|             run (crystal)            |   simulation for the crystal supercell.
+--------------------------------------+
                   |
                   |
+--------------------------------------+
|           dynamics.md.data           |   Processes the crystal's trajectory
|                crystal               |   to extract thermodynamic properties.
+--------------------------------------+
                   |
                   |
+--------------------------------------+
|           dynamics.md.data           |   Calculates the sublimation enthalpy
|              sublimation             |   from the molecule and crystal data.
+--------------------------------------+

```

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

InpScript = "md.py"
LogFile = "md.log"

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

virtual_environment = os.path.expanduser("~/.virtualenvs/compute-env")
virtual_environment = os.path.realpath(virtual_environment)
activate_env = os.path.realpath(os.path.join(virtual_environment, "bin", "activate"))
cmd = f"module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10 && . {activate_env} && python {InpScript}"

with open(LogFile, "w") as log_file:
    process = subprocess.Popen(cmd, shell=True, stdout=log_file,
                               stderr=subprocess.STDOUT, bufsize=1,
                               universal_newlines=True)
    process.communicate()
```
