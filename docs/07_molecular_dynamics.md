# Molecular Dynamics

- [Setup](#setup)
- [NPT/NVT Propagation](#nptnvt-propagation)
- [Adjustable parameters](#adjustable-parameters)
- [Function Call Overview](#function-call-overview)
- [Computational Bottlenecks](#computational-bottlenecks)
- [Complete Input Files](#complete-input-files)

This workflow is designed for performing molecular dynamics (MD) simulations to compute properties such as the sublimation energy of a molecular crystal.

## Setup

The initial setup requires importing the necessary modules and defining the system's structures and the MLIP calculator.

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/mace.model")
```

## NPT/NVT Propagation

The MD workflow is configured using the `Enthalpy` and `ClassicalMD` classes from `mbe_automation.configs.md`.

```python
md_config = mbe_automation.configs.md.Enthalpy(
    molecule=Structure.from_xyz_file(xyz_molecule),
    crystal=Structure.from_xyz_file(xyz_solid),
    calculator=mace_calc,
    temperatures_K=np.array([298.15]),
    pressures_GPa=np.array([1.0E-4, 1.0]),
    dataset="properties.hdf5",

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
mbe_automation.run(md_config)
```

## Adjustable parameters

Detailed descriptions of the configuration classes can be found in the [Configuration Classes](./03_configuration_classes.md) chapter.

*   **[`Enthalpy`](./03_configuration_classes.md#enthalpy-class)**: Main configuration for the molecular dynamics workflow.
*   **[`ClassicalMD`](./03_configuration_classes.md#classicalmd-class)**: Configuration for the MD simulation parameters.

## Function Call Overview

```
+--------------------------------------+
|           mbe_automation             |
|                run                   |
+--------------------------------------+
                   |
                   |
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

For a detailed discussion of performance considerations, see the [Computational Bottlenecks](./05_bottlenecks.md) section.

## How to read the results

### HDF5 Datasets

The `mbe-automation` program uses the Hierarchical Data Format version 5 (HDF5) for storing large amounts of numerical data. The HDF5 file produced by a workflow contains all the raw and processed data in a hierarchical structure, similar to a file system with folders and files.

### File Structure

You can visualize the structure of the output file using `mbe_automation.tree`.

```python
import mbe_automation

mbe_automation.tree("properties.hdf5")
```

An MD calculation will produce a file with the following structure:

```
properties.hdf5
└── md
    ├── structures
    │   ├── molecule[extracted,0]
    │   ├── molecule[extracted,0,opt:atoms]
    │   └── ... (other structures)
    ├── thermodynamics
    └── trajectories
        ├── crystal[dyn:T=298.15,p=0.00010]
        ├── crystal[dyn:T=298.15,p=1.00000]
        └── molecule[dyn:T=298.15]
```

- **`structures`**: Group containing the input crystal structures and extracted molecules.
- **`thermodynamics`**: Contains thermodynamic properties such as potential energy, kinetic energy, and volume averaged over the MD trajectory.
- **`trajectories`**: Group containing the MD trajectories (positions, velocities, forces, etc.) for each simulation.

## Complete Input Files

### Python Script (`md.py`)

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/model.model")

md_config = mbe_automation.configs.md.Enthalpy(
    molecule=Structure.from_xyz_file(xyz_molecule),
    crystal=Structure.from_xyz_file(xyz_solid),
    calculator=mace_calc,
    temperatures_K=np.array([298.15]),
    pressures_GPa=np.array([1.0E-4, 1.0]),
    dataset="properties.hdf5",

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

mbe_automation.run(md_config)
```

### Bash Script (`run.sh`)

```bash
#!/bin/bash
#SBATCH --job-name="MACE"
#SBATCH -A pl0415-02
#SBATCH --partition=tesla
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1 --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=180gb

module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

python md.py > md.log 2>&1
```
