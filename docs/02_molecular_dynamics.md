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
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/mace.model")
```

## NPT/NVT Propagation

The MD workflow is configured using the `Enthalpy` and `ClassicalMD` classes from `mbe_automation.configs.md`.

```python
md_config = mbe_automation.configs.md.Enthalpy(
    molecule=from_xyz_file(xyz_molecule),
    crystal=from_xyz_file(xyz_solid),
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

### `Enthalpy` Class

**Location:** `mbe_automation.configs.md.Enthalpy`

| Parameter | Description | Default Value |
| --- | --- | --- |
| `molecule` | Initial, non-relaxed structure of the isolated molecule. An MD simulation is performed in the NVT ensemble to compute the average potential and kinetic energies. | - |
| `crystal` | Initial, non-relaxed crystal structure. An MD simulation is performed in the NPT ensemble to compute the average potential and kinetic energies, and the average volume. | - |
| `calculator` | MLIP calculator for energies and forces. | - |
| `md_molecule` | An instance of `ClassicalMD` that configures the MD simulation for the isolated molecule. | - |
| `md_crystal` | An instance of `ClassicalMD` that configures the MD simulation for the crystal. | - |
| `temperatures_K` | Target temperatures (in Kelvin) for the MD simulation. Can be a single float or an array of floats. | `298.15` |
| `pressures_GPa` | Target pressures (in GPa) for the MD simulation. Can be a single float or an array of floats. | `1.0E-4` |
| `unique_molecules_energy_thresh` | Energy threshold (eV/atom) used to detect nonequivalent molecules in the input unit cell. | `1.0E-3`    |
| `relaxation`                     | An instance of `Minimum` that configures the geometry relaxation parameters.                | `Minimum()` |
| `work_dir` | Directory where files are stored at runtime. | `"./"` |
| `dataset` | The main HDF5 file with all data computed for the physical system. | `"./properties.hdf5"` |
| `root_key` | Specifies the root path in the HDF5 dataset where the workflow's output is stored. | `"md"` |
| `verbose` | Verbosity of the program's output. `0` suppresses warnings. | `0` |
| `save_plots` | If `True`, save plots of the simulation results. | `True` |
| `save_csv` | If `True`, save CSV files of the simulation results. | `True` |

### `ClassicalMD` Class

**Location:** `mbe_automation.configs.md.ClassicalMD`

| Parameter               | Description                                                                                                                              | Default Value     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `time_total_fs`         | Total simulation time in femtoseconds.                                                                                               | `50000.0`             |
| `time_step_fs`          | Time step for the integration algorithm.                                                                                             | `0.5`                 |
| `sampling_interval_fs`  | Interval for trajectory sampling.                                                                                                   | `50.0`                |
| `time_equilibration_fs` | Initial period of the simulation discarded to allow the system to reach equilibrium.                                         | `5000.0`              |
| `ensemble`              | Thermodynamic ensemble for the simulation ("NVT" or "NPT").                                                                          | "NVT"                 |
| `nvt_algo`              | Thermostat algorithm for NVT simulations. "csvr" (Canonical Sampling Through Velocity Rescaling) is robust for isolated molecules. | "csvr"                |
| `npt_algo`              | Barostat/thermostat algorithm for NPT simulations.                                                                                   | "mtk_full"            |
| `thermostat_time_fs`    | Thermostat relaxation time.                                                                                                          | `100.0`               |
| `barostat_time_fs`      | Barostat relaxation time.                                                                                                            | `1000.0`              |
| `tchain`                | Number of thermostats in the Nosé-Hoover chain.                                                                                      | `3`                   |
| `pchain`                | Number of barostats in the Martyna-Tuckerman-Klein chain.                                                                            | `3`                   |
| `supercell_radius`      | Minimum point-periodic image distance in the supercell (Å).                                                                        | `25.0`                |
| `supercell_matrix`      | Supercell transformation matrix. If specified, `supercell_radius` is ignored.                                                        | `None`                |
| `supercell_diagonal`    | If `True`, create a diagonal supercell. Ignored if `supercell_matrix` is provided.                                                   | `False`               |

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

For a detailed discussion of performance considerations, see the [Computational Bottlenecks](./06_bottlenecks.md) section.

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
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/model.model")

md_config = mbe_automation.configs.md.Enthalpy(
    molecule=from_xyz_file(xyz_molecule),
    crystal=from_xyz_file(xyz_solid),
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

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

python md.py > md.log 2>&1
```
