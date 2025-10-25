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

This block defines the paths to the input structures and initializes the MACE calculator.

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
```

The `Enthalpy` class configures the overall simulation, while separate `ClassicalMD` instances define the parameters for the individual MD runs for the molecule and the crystal.

### Key Parameters for `ClassicalMD`:

*   `ensemble`: The thermodynamic ensemble for the simulation. The "NVT" ensemble (constant number of particles, volume, and temperature) is used for the isolated molecule. The "NPT" ensemble (constant number of particles, pressure, and temperature) is used for the crystal to allow the cell volume to fluctuate.
*   `time_total_fs`: The total simulation time in femtoseconds. For robust results, a total simulation time of 50 ps (50,000 fs) is recommended.
*   `time_step_fs`: The time step for the integration algorithm. A value of 0.5 fs is a safe choice, especially for PIMD calculations.
*   `time_equilibration_fs`: The initial period of the simulation that is discarded to allow the system to reach thermal equilibrium. A value of 5,000 fs (5 ps) is a typical choice.
*   `nvt_algo`: The thermostat algorithm for NVT simulations. The "csvr" (Canonical Sampling Through Velocity Rescaling) thermostat is recommended as it is robust and suitable for isolated molecules.
*   `npt_algo`: The barostat/thermostat algorithm for NPT simulations. "mtk_full" is a common choice.

## Execution

The MD workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.workflows.md.run(md_config)
```

This command will run the MD simulations, and the results will be saved to the specified HDF5 dataset file.
