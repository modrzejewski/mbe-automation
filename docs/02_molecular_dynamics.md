# Molecular Dynamics

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

The `Enthalpy` class configures the overall simulation, while separate `ClassicalMD` instances define the parameters for the individual MD runs.

### Key Parameters for `ClassicalMD`:

| Parameter               | Description                                                                                                                              | Default Value     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `ensemble`              | The thermodynamic ensemble for the simulation ("NVT" or "NPT").                                                                          | "NVT"                 |
| `time_total_fs`         | The total simulation time in femtoseconds.                                                                                               | `50000.0`             |
| `time_step_fs`          | The time step for the integration algorithm.                                                                                             | `0.5`                 |
| `sampling_interval_fs`  | The interval for trajectory sampling.                                                                                                   | `50.0`                |
| `time_equilibration_fs` | The initial period of the simulation that is discarded to allow the system to reach equilibrium.                                         | `5000.0`              |
| `nvt_algo`              | The thermostat algorithm for NVT simulations. "csvr" (Canonical Sampling Through Velocity Rescaling) is robust for isolated molecules. | "csvr"                |
| `npt_algo`              | The barostat/thermostat algorithm for NPT simulations.                                                                                   | "mtk_full"            |
| `thermostat_time_fs`    | The thermostat relaxation time.                                                                                                          | `100.0`               |
| `barostat_time_fs`      | The barostat relaxation time.                                                                                                            | `1000.0`              |
| `supercell_radius`      | The minimum point-periodic image distance in the supercell (Å).                                                                        | `25.0`                |

## Execution

The MD workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.workflows.md.run(md_config)
```

## Programming Aspects

The `run` function in `mbe_automation/workflows/md.py` executes the following sequence of operations:

1.  **Molecule Simulation (NVT):**
    *   A molecular dynamics simulation is performed on the isolated `molecule` in the NVT ensemble using `mbe_automation.dynamics.md.core.run`.
    *   The trajectory data is processed by `mbe_automation.dynamics.md.data.molecule` to extract thermodynamic properties.

2.  **Crystal Simulation (NPT):**
    *   The supercell matrix for the `crystal` is determined using `mbe_automation.structure.crystal.supercell_matrix`.
    *   An MD simulation is performed on the crystal supercell in the NPT ensemble, again using `mbe_automation.dynamics.md.core.run`.
    *   The crystal's trajectory data is processed by `mbe_automation.dynamics.md.data.crystal`.

3.  **Sublimation Enthalpy Calculation:**
    *   The sublimation enthalpy is calculated by `mbe_automation.dynamics.md.data.sublimation`, which combines the results from the molecule and crystal simulations.

4.  **Data Storage:**
    *   All final results are compiled into a pandas DataFrame and saved to the HDF5 `dataset` file.
