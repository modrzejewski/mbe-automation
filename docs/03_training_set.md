# Training Set Creation

- [Setup](#setup)
- [Configuration](#configuration)
  - [MD Sampling](#md-sampling)
  - [Force Constants](#force-constants)
  - [Finite Subsystem Extraction](#finite-subsystem-extraction)
  - [Phonon Filtering](#phonon-filtering)
- [Execution](#execution)
- [Details](#details)
- [Function Call Overview](#function-call-overview)
- [Computational Bottlenecks](#computational-bottlenecks)
- [Complete Input Files](#complete-input-files)

This workflow generates a diverse set of configurations for delta-learning a machine learning interatomic potential. The process is divided into three stages: MD sampling, quasi-harmonic calculations, and phonon sampling.

## Setup

The initial setup involves importing the necessary modules and defining the parameters for the workflow.

```python
import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation
from mbe_automation.configs.md import ClassicalMD
from mbe_automation.structure.clusters import FiniteSubsystemFilter
from mbe_automation.dynamics.harmonic.modes import PhononFilter
from mbe_automation.configs.training import MDSampling, PhononSampling
from mbe_automation.configs.quasi_harmonic import FreeEnergy
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
mlip_parameter_file = "path/to/your/model.model"
temperature_K = 298.15
work_dir = os.path.abspath(os.path.dirname(__file__))
dataset = os.path.join(work_dir, "training_set.hdf5")

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser(mlip_parameter_file),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)
```

## Configuration

### MD Sampling

The first stage generates configurations by running a short molecular dynamics simulation.

```python
md_sampling_config = MDSampling(
    crystal=from_xyz_file(xyz_solid),
    calculator=mace_calc,
    temperature_K=temperature_K,
    pressure_GPa=1.0E-4,
    finite_subsystem_filter=FiniteSubsystemFilter(
        selection_rule="closest_to_central_molecule",
        n_molecules=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
    ),
    md_crystal=ClassicalMD(
        ensemble="NPT",
        time_total_fs=10000.0,
        time_equilibration_fs=1000.0,
        sampling_interval_fs=1000.0,
        supercell_radius=10.0,
    ),
    work_dir=os.path.join(work_dir, "md_sampling"),
    dataset=dataset,
    root_key="training/md_sampling"
)
```

| Parameter                 | Description                                                                                             | Default Value                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `crystal`                 | Initial crystal structure.                                                                          | -                                  |
| `calculator`              | MLIP calculator.                                                                                    | -                                  |
| `temperature_K`           | Target temperature (in Kelvin) for the MD simulation.                                               | `298.15`                           |
| `pressure_GPa`            | Target pressure (in GPa) for the MD simulation.                                                     | `1.0E-4`                           |
| `finite_subsystem_filter` | An instance of `FiniteSubsystemFilter` that defines how finite molecular clusters are extracted.        | `FiniteSubsystemFilter()`          |
| `md_crystal`              | An instance of `ClassicalMD` that configures the MD simulation parameters.                              | -                                  |

#### `ClassicalMD` Parameters

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

### Force Constants

A quasi-harmonic calculation is performed to obtain the force constants required for the phonon sampling stage.

```python
free_energy_config = FreeEnergy(
    crystal=from_xyz_file(xyz_solid),
    calculator=mace_calc,
    thermal_expansion=False,
    relax_input_cell="constant_volume",
    supercell_radius=20.0,
    dataset=dataset,
    root_key="training/quasi_harmonic"
)
```

| Parameter                       | Description                                                                                                                                                                                            | Default Value                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `crystal`                       | Initial, non-relaxed crystal structure.                                                                                                                                                            | -                                               |
| `calculator`                    | MLIP calculator for energies and forces.                                                                                                                                                           | -                                               |
| `thermal_expansion`             | If `True`, performs volumetric thermal expansion calculations. If `False`, uses the harmonic approximation.                                                                                            | `True`                                          |
| `supercell_radius`              | Minimum point-periodic image distance in the supercell for phonon calculations (Å).                                                                                                               | `25.0`                                          |
| `relax_input_cell`              | Relaxation of the input structure: "full", "constant_volume", or "only_atoms".                                                                                                               | `"constant_volume"`                             |

### Finite Subsystem Extraction

This class defines the criteria for extracting finite molecular clusters from a periodic trajectory.

| Parameter                       | Description                                                                                                                                                             | Default Value                               |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `selection_rule`                | The rule for selecting molecules. Options include: `closest_to_center_of_mass`, `closest_to_central_molecule`, `max_min_distance_to_central_molecule`, and `max_max_distance_to_central_molecule`. | `closest_to_central_molecule`               |
| `n_molecules`                   | An array of integers specifying the number of molecules to include in each cluster. Used with `closest_to_center_of_mass` and `closest_to_central_molecule` selection rules. | `np.array([1, 2, ..., 8])`                  |
| `distances`                     | An array of floating-point numbers specifying the cutoff distances (in Å) for molecule selection. Used with `max_min_distance_to_central_molecule` and `max_max_distance_to_central_molecule` rules. | `None`                                      |
| `assert_identical_composition`  | If `True`, the workflow will raise an error if it detects that not all molecules in the periodic structure have the same elemental composition.                            | `True`                                      |

### Phonon Filtering

The final stage generates configurations by sampling from the phonon modes of the crystal.

```python
phonon_sampling_config = PhononSampling(
    calculator=mace_calc,
    temperature_K=temperature_K,
    finite_subsystem_filter=FiniteSubsystemFilter(
        selection_rule="closest_to_central_molecule",
        n_molecules=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
    ),
    phonon_filter=PhononFilter(
        k_point_mesh="gamma",
        freq_min_THz=0.1,
        freq_max_THz=8.0
    ),
    force_constants_dataset=dataset,
    force_constants_key="training/quasi_harmonic/phonons/crystal[opt:atoms,shape]/force_constants",
    time_step_fs=100.0,
    n_frames=20,
    work_dir=os.path.join(work_dir, "phonon_sampling"),
    dataset=dataset,
    root_key="training/phonon_sampling"
)
```

| Parameter                 | Description                                                                          | Default Value |
| ------------------------- | ------------------------------------------------------------------------------------ | ----------------- |
| `calculator`              | MLIP calculator.                                                                 | -                 |
| `temperature_K`           | Temperature (in Kelvin) for the phonon sampling.                                 | `298.15`          |
| `phonon_filter`           | An instance of `PhononFilter` that specifies which phonon modes to sample from.        | `PhononFilter()`  |
| `force_constants_dataset` | Path to the HDF5 file containing the force constants. | `./properties.hdf5` |
| `force_constants_key`     | Key within the HDF5 file where the force constants are stored.                     | -                 |
| `time_step_fs`            | Time step for the trajectory generation.                                         | `100.0`           |
| `n_frames`                | Number of frames to generate for each selected phonon mode.                        | `20`              |

#### `PhononFilter`

| Parameter          | Description                                                                                                                                                                                                                                                          | Default Value |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `k_point_mesh`     | The k-points for sampling the Brillouin zone. Can be `"gamma"` (for the Γ point only), a float (for a Monkhorst-Pack grid defined by a radius), or a 3-element array (for an explicit Monkhorst-Pack mesh).                                                               | `"gamma"`     |
| `selected_modes`   | An array of 1-based indices for selecting specific phonon modes at each k-point. If this is specified, `freq_min_THz` and `freq_max_THz` are ignored.                                                                                                                   | `None`        |
| `freq_min_THz`     | The minimum phonon frequency (in THz) to be included in the sampling.                                                                                                                                                                                              | `0.1`         |
| `freq_max_THz`     | The maximum phonon frequency (in THz) to be included in the sampling. If `None`, all frequencies above `freq_min_THz` are included.                                                                                                                                   | `8.0`         |

## Execution

The workflow is executed by passing the configuration objects to the `run` function.

```python
mbe_automation.workflows.training.run(md_sampling_config)
mbe_automation.workflows.quasi_harmonic.run(free_energy_config)
mbe_automation.workflows.training.run(phonon_sampling_config)
```

## Details

The `run` function in `mbe_automation/workflows/training.py` dispatches to one of two functions based on the configuration type, each designed to generate a diverse set of atomic configurations for training a delta-learning model.

1.  **MD Sampling (`md_sampling`):**
    *   This function runs a molecular dynamics simulation in the NPT ensemble using `mbe_automation.dynamics.md.core.run`. The trajectory provides a set of thermally-accessible configurations at a given temperature and pressure.
    *   From this periodic trajectory, the workflow extracts finite, non-periodic clusters. This is a two-step process:
        1.  **Molecule Detection:** The `detect_molecules` function identifies individual molecules within the periodic supercell. It uses a graph-based algorithm on a reference frame to find connected components of atoms, unwraps the periodic boundary conditions for these molecules, and tracks them across the entire trajectory.
        2.  **Subsystem Extraction:** The `extract_finite_subsystem` function then selects molecules from the unwrapped trajectory based on the criteria defined in the `FiniteSubsystemFilter`.

2.  **Phonon Sampling (`phonon_sampling`):**
    *   This function generates a trajectory by displacing atoms along the normal modes (phonons) of the crystal, using `mbe_automation.dynamics.harmonic.modes.trajectory`. This method is particularly effective at generating distorted geometries that may be energetically unfavorable but are important for teaching the MLIP about repulsive interactions.
    *   The extraction of finite subsystems from the resulting trajectory follows the same two-step process of molecule detection and subsystem extraction as in the MD sampling workflow.

### Finite Subsystem Extraction

The `FiniteSubsystemFilter` provides several `selection_rule` options for how to extract the clusters:

*   **`closest_to_center_of_mass`**: Selects a specified number of molecules (`n_molecules`) whose centers of mass are closest to the center of mass of the entire system.
*   **`closest_to_central_molecule`**: Identifies a "central" molecule (the one closest to the origin) and then selects a specified number of molecules (`n_molecules`) whose centers of mass are closest to that central molecule.
*   **`max_min_distance_to_central_molecule`**: Selects all molecules where the minimum interatomic distance to the central molecule is less than a given `distance`.
*   **`max_max_distance_to_central_molecule`**: Selects all molecules where the maximum interatomic distance to the central molecule is less than a given `distance`.

After the finite subsystems are extracted, if a MACE calculator is provided, the workflow performs an inference calculation to obtain the energies, forces, and feature vectors for each configuration in the subsystem. All data is then saved to the specified HDF5 `dataset`.

## Function Call Overview

### MD Sampling

```
+------------------------------------+
|         workflows.training         |
|            md_sampling             |
+------------------------------------+
                    |
                    |
+------------------------------------+
|          dynamics.md.core          |   Runs a molecular dynamics
|                run                 |   simulation.
+------------------------------------+
                    |
                    |
+------------------------------------+
|         structure.clusters         |   Identifies molecules within
|           detect_molecules         |   the crystal structure.
+------------------------------------+
                    |
                    |
+------------------------------------+
|         structure.clusters         |   Extracts finite molecular clusters
|      extract_finite_subsystem      |   from the periodic trajectory.
+------------------------------------+

```

### Phonon Sampling

```
+------------------------------------+
|         workflows.training         |
|          phonon_sampling           |
+------------------------------------+
                    |
                    |
+------------------------------------+
|      dynamics.harmonic.modes       |   Generates a trajectory by
|             trajectory             |   sampling from phonon modes.
+------------------------------------+
                    |
                    |
+------------------------------------+
|         structure.clusters         |   Identifies molecules within
|           detect_molecules         |   the crystal structure.
+------------------------------------+
                    |
                    |
+------------------------------------+
|         structure.clusters         |   Extracts finite molecular clusters
|      extract_finite_subsystem      |   from the periodic trajectory.
+------------------------------------+

```

## Computational Bottlenecks

The computational cost of this workflow is a combination of its three stages:

*   **MD Sampling**: The cost is primarily determined by the `time_total_fs` and `supercell_radius` parameters of the `md_crystal` configuration.
*   **Quasi-Harmonic Calculation**: This step is generally less expensive than the others, but its cost is influenced by the `supercell_radius` used for the force constant calculation.
*   **Phonon Sampling**: The number of frames (`n_frames`) and the number of phonon modes selected by the `phonon_filter` are the main drivers of the computational cost in this final stage.

## Complete Input Files

### Python Script (`training.py`)

```python
import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation
from mbe_automation.configs.md import ClassicalMD
from mbe_automation.structure.clusters import FiniteSubsystemFilter
from mbe_automation.dynamics.harmonic.modes import PhononFilter
from mbe_automation.configs.training import MDSampling, PhononSampling
from mbe_automation.configs.quasi_harmonic import FreeEnergy
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
mlip_parameter_file = "path/to/your/model.model"
temperature_K = 298.15
work_dir = os.path.abspath(os.path.dirname(__file__))
dataset = os.path.join(work_dir, "training_set.hdf5")

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser(mlip_parameter_file),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

md_sampling_config = MDSampling(
    crystal=from_xyz_file(xyz_solid),
    calculator=mace_calc,
    temperature_K=temperature_K,
    pressure_GPa=1.0E-4,
    finite_subsystem_filter=FiniteSubsystemFilter(
        selection_rule="closest_to_central_molecule",
        n_molecules=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
    ),
    md_crystal=ClassicalMD(
        ensemble="NPT",
        time_total_fs=10000.0,
        time_equilibration_fs=1000.0,
        sampling_interval_fs=1000.0,
        supercell_radius=10.0,
    ),
    work_dir=os.path.join(work_dir, "md_sampling"),
    dataset=dataset,
    root_key="training/md_sampling"
)
mbe_automation.workflows.training.run(md_sampling_config)

free_energy_config = FreeEnergy(
    crystal=from_xyz_file(xyz_solid),
    calculator=mace_calc,
    thermal_expansion=False,
    relax_input_cell="constant_volume",
    supercell_radius=20.0,
    dataset=dataset,
    root_key="training/quasi_harmonic"
)
mbe_automation.workflows.quasi_harmonic.run(free_energy_config)

phonon_sampling_config = PhononSampling(
    calculator=mace_calc,
    temperature_K=temperature_K,
    finite_subsystem_filter=FiniteSubsystemFilter(
        selection_rule="closest_to_central_molecule",
        n_molecules=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
    ),
    phonon_filter=PhononFilter(
        k_point_mesh="gamma",
        freq_min_THz=0.1,
        freq_max_THz=8.0
    ),
    force_constants_dataset=dataset,
    force_constants_key="training/quasi_harmonic/phonons/crystal[opt:atoms,shape]/force_constants",
    time_step_fs=100.0,
    n_frames=20,
    work_dir=os.path.join(work_dir, "phonon_sampling"),
    dataset=dataset,
    root_key="training/phonon_sampling"
)
mbe_automation.workflows.training.run(phonon_sampling_config)
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

InpScript = "training.py"
LogFile = "training.log"

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