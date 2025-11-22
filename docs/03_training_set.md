# Training Set Creation

- [Setup](#setup)
- [Step 1: MD Sampling](#step-1-md-sampling)
- [Step 2: Force Constants Model](#step-2-force-constants-model)
- [Step 3: Phonon Sampling](#step-3-phonon-sampling)
- [Adjustable parameters](#adjustable-parameters)
  - [`MDSampling`](#mdsampling-class)
  - [`PhononSampling`](#phononsampling-class)
  - [`ClassicalMD`](#classicalmd-class)
  - [`FreeEnergy`](#freeenergy-class)
  - [`Minimum`](#minimum-class)
  - [`FiniteSubsystemFilter`](#finitesubsystemfilter-class)
  - [`PhononFilter`](#phononfilter-class)
- [Subsampling](#subsampling)
- [Updates to and Existing Dataset](#updates-to-an-existing-dataset)
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

## Step 1: MD Sampling

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
mbe_automation.workflows.training.run(md_sampling_config)
```

## Step 2: Force Constants Model

A quasi-harmonic calculation is performed to obtain the force constants required for the phonon sampling stage.

```python
free_energy_config = FreeEnergy.recommended(
    model_name="mace",
    crystal=from_xyz_file(xyz_solid),
    calculator=mace_calc,
    thermal_expansion=False,
    supercell_radius=20.0,
    dataset=dataset,
    root_key="training/quasi_harmonic"
)
mbe_automation.workflows.quasi_harmonic.run(free_energy_config)
```

## Step 3: Phonon Sampling

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
    amplitude_scan="random",
    time_step_fs=100.0,
    n_frames=20,
    work_dir=os.path.join(work_dir, "phonon_sampling"),
    dataset=dataset,
    root_key="training/phonon_sampling"
)
mbe_automation.workflows.training.run(phonon_sampling_config)
```

## Adjustable parameters

### `MDSampling` Class

**Location:** `mbe_automation.configs.training.MDSampling`

| Parameter                 | Description                                                                                             | Default Value                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `crystal`                 | Initial crystal structure. From this periodic trajectory, the workflow extracts finite, non-periodic clusters. | - |
| `calculator`              | MLIP calculator.                                                                                    | -                                  |
| `md_crystal`              | An instance of `ClassicalMD` that configures the MD simulation parameters. Defaults used in `MDSampling` differ from standard `ClassicalMD` defaults: `time_total_fs=100000.0`, `supercell_radius=15.0`, `feature_vectors_type="averaged_environments"`. | -                                  |
| `temperature_K`           | Target temperature (in Kelvin) for the MD simulation.                                               | `298.15`                           |
| `pressure_GPa`            | Target pressure (in GPa) for the MD simulation.                                                     | `1.0E-4`                           |
| `finite_subsystem_filter` | An instance of `FiniteSubsystemFilter` that defines how finite molecular clusters are extracted.        | `FiniteSubsystemFilter()`          |
| `work_dir`                | Directory where files are stored at runtime.                                                            | `"./"`                             |
| `dataset`                 | The main HDF5 file with all data computed for the physical system.                                      | `"./properties.hdf5"`              |
| `root_key`                | Specifies the root path in the HDF5 dataset where the workflow's output is stored.                      | `"training/md_sampling"`           |
| `verbose`                 | Verbosity of the program's output. `0` suppresses warnings.                                             | `0`                                |
| `save_plots`              | If `True`, save plots of the simulation results.                                                        | `True`                             |
| `save_csv`                | If `True`, save CSV files of the simulation results.                                                    | `True`                             |

### `PhononSampling` Class

**Location:** `mbe_automation.configs.training.PhononSampling`

| Parameter                 | Description                                                                          | Default Value |
| ------------------------- | ------------------------------------------------------------------------------------ | ----------------- |
| `force_constants_dataset` | Path to the HDF5 file containing the force constants. | `./properties.hdf5` |
| `force_constants_key`     | Key within the HDF5 file where the force constants are stored.                     | `"training/quasi_harmonic/phonons/crystal[opt:atoms,shape]/force_constants"` |
| `calculator`              | MLIP calculator.                                                                 | -                 |
| `temperature_K`           | Temperature (in Kelvin) for the phonon sampling.                                 | `298.15`          |
| `phonon_filter`           | An instance of `PhononFilter` that specifies which phonon modes to sample from. This method is particularly effective at generating distorted geometries that may be energetically unfavorable but are important for teaching the MLIP about repulsive interactions.       | `PhononFilter()`  |
| `finite_subsystem_filter` | An instance of `FiniteSubsystemFilter` that defines how finite molecular clusters are extracted.        | `FiniteSubsystemFilter()`          |
| `amplitude_scan`          | Method for sampling normal-mode coordinates. `"random"` multiplies eigenvectors by a random number on (-1, 1). `"time_propagation"` uses a time-dependent phase factor. | `"random"`                         |
| `time_step_fs`            | Time step for trajectory generation (used only if `amplitude_scan` is `"time_propagation"`).            | `100.0`           |
| `rng`                     | Random number generator for randomized amplitude sampling (used only if `amplitude_scan` is `"random"`). | `np.random.default_rng(seed=42)`   |
| `n_frames`                | Number of frames to generate for each selected phonon mode.                        | `20`              |
| `feature_vectors_type`    | Type of feature vectors to save. Required for subsampling based on feature space distances. Works only with MACE models. | `"averaged_environments"` |
| `work_dir`                | Directory where files are stored at runtime.                                                            | `"./"`                             |
| `dataset`                 | The main HDF5 file with all data computed for the physical system.                                      | `"./properties.hdf5"`              |
| `root_key`                | Specifies the root path in the HDF5 dataset where the workflow's output is stored.                     | `"training/phonon_sampling"` |
| `verbose`                 | Verbosity of the program's output. `0` suppresses warnings.                                             | `0`                                |

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
| `feature_vectors_type`  | Type of feature vectors to save. Options are "none", "atomic_environments", or "averaged_environments". Enables subsampling based on distances in the feature space. Works only with MACE models. | `"none"`                  |

### `FreeEnergy` Class

**Location:** `mbe_automation.configs.quasi_harmonic.FreeEnergy`

| Parameter                       | Description                                                                                                                                                                                            | Default Value                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `crystal`                       | Initial, non-relaxed crystal structure.                                                                                                                                                            | -                                               |
| `calculator`                    | MLIP calculator for energies and forces.                                                                                                                                                           | -                                               |
| `thermal_expansion`             | If `True`, performs volumetric thermal expansion calculations. If `False`, uses the harmonic approximation.                                                                                            | `True`                                          |
| `supercell_radius`              | Minimum point-periodic image distance in the supercell for phonon calculations (Å).                                                                                                               | `25.0`                                          |
| `relaxation`                    | An instance of `Minimum` that configures the geometry relaxation parameters.                                                                                                                       | `Minimum()`                                     |

### `Minimum` Class

**Location:** `mbe_automation.configs.structure.Minimum`

| Parameter                    | Description                                                                                                                                                           | Default Value       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `max_force_on_atom_eV_A`     | Maximum residual force threshold for geometry relaxation (eV/Å).                                                                                                      | `1.0E-4`            |
| `max_n_steps`                | Maximum number of steps in the geometry relaxation.                                                                                                                   | `500`               |
| `cell_relaxation`            | Relaxation of the input structure: "full" (optimizes atomic positions, cell shape, and volume), "constant_volume" (optimizes atomic positions and cell shape at fixed volume), or "only_atoms" (optimizes only atomic positions). | `"constant_volume"` |
| `pressure_GPa`               | External isotropic pressure (in GPa) applied during lattice relaxation.                                                                                               | `0.0`               |
| `symmetrize_final_structure` | If `True`, refines the space group symmetry after each geometry relaxation.                                                                                           | `True`              |
| `backend`                    | Software used to perform the geometry relaxation: "ase" or "dftb".                                                                                                    | `"ase"`             |

### `FiniteSubsystemFilter` Class

**Location:** `mbe_automation.structure.clusters.FiniteSubsystemFilter`

| Parameter                       | Description                                                                                                                                                             | Default Value                               |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `selection_rule`                | The rule for selecting molecules. Options include: `closest_to_center_of_mass` (selects molecules closest to the center of mass of the entire system), `closest_to_central_molecule` (selects molecules closest to the central molecule), `max_min_distance_to_central_molecule` (selects molecules where the minimum interatomic distance to the central molecule is less than a given `distance`), and `max_max_distance_to_central_molecule` (selects molecules where the maximum interatomic distance to the central molecule is less than a given `distance`). | `closest_to_central_molecule`               |
| `n_molecules`                   | An array of integers specifying the number of molecules to include in each cluster. Used with `closest_to_center_of_mass` and `closest_to_central_molecule` selection rules. | `np.array([1, 2, ..., 8])`                  |
| `distances`                     | An array of floating-point numbers specifying the cutoff distances (in Å) for molecule selection. Used with `max_min_distance_to_central_molecule` and `max_max_distance_to_central_molecule` rules. | `None`                                      |
| `assert_identical_composition`  | If `True`, the workflow will raise an error if it detects that not all molecules in the periodic structure have the same elemental composition.                            | `True`                                      |

### `PhononFilter` Class

**Location:** `mbe_automation.dynamics.harmonic.modes.PhononFilter`

| Parameter          | Description                                                                                                                                                                                                                                                          | Default Value |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `k_point_mesh`     | The k-points for sampling the Brillouin zone. Can be `"gamma"` (for the Γ point only), a float (for a Monkhorst-Pack grid defined by a radius), or a 3-element array (for an explicit Monkhorst-Pack mesh).                                                               | `"gamma"`     |
| `selected_modes`   | An array of 1-based indices for selecting specific phonon modes at each k-point. If this is specified, `freq_min_THz` and `freq_max_THz` are ignored.                                                                                                                   | `None`        |
| `freq_min_THz`     | The minimum phonon frequency (in THz) to be included in the sampling.                                                                                                                                                                                              | `0.1`         |
| `freq_max_THz`     | The maximum phonon frequency (in THz) to be included in the sampling. If `None`, all frequencies above `freq_min_THz` are included.                                                                                                                                   | `8.0`         |

## Subsampling

The purpose of subsampling is to select a diverse set of configurations for training a machine learning potential. By choosing a smaller, representative subset of frames from a larger dataset, you can reduce the computational cost of training while ensuring that the model is exposed to a wide range of atomic environments. The `subsample` method, available for `Structure`, `Trajectory`, `MolecularCrystal`, and `FiniteSubsystem` objects, provides a way to do this.

The subsampling process is based on feature vectors, which are numerical representations of the atomic environments in each frame. The method uses algorithms like farthest point sampling to select a diverse set of frames that cover the feature space as broadly as possible. To enable subsampling, you must first save the feature vectors to the dataset by setting the [`feature_vectors_type`](#classicalmd-class) parameter to `"atomic_environments"` or `"averaged_environments"` in the `ClassicalMD` or `PhononSampling` configurations.

### `subsample` Method Parameters

| Parameter   | Description                                                                                                                                       | Default Value                |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| `n`         | The number of frames to select from the dataset.                                                                                                  | -                            |
| `algorithm` | The algorithm to use for subsampling. Options are `"farthest_point_sampling"` and `"kmeans"`. Both methods aim to select a diverse subset of frames by analyzing their feature vectors. | `"farthest_point_sampling"`  |

### Example Usage

The following example demonstrates how to read a `Trajectory` from a dataset, and then use the `subsample` method to select a smaller number of frames.

```python
from mbe_automation.storage import read_trajectory, save_trajectory

# Read the full trajectory from the dataset
full_trajectory = read_trajectory(
    dataset="training_set.hdf5",
    key="training/md_sampling/trajectory"
)

# Subsample the trajectory to select 100 frames
subsampled_trajectory = full_trajectory.subsample(n=100)

# Save the subsampled trajectory to a new key in the dataset
save_trajectory(
    dataset="training_set.hdf5",
    key="training/md_sampling/trajectory_subsampled",
    traj=subsampled_trajectory
)
```

## Updates to an Existing Dataset

The `Structure` and `Trajectory` classes allow updating an existing dataset with new properties, such as feature vectors, potential energies, or forces. This is useful for machine learning workflows where new data needs to be computed for existing geometries.

To update a dataset, first load the structure, run the calculator, and then save the specific properties using the `only` argument in the `save` method.

```python
from mbe_automation.storage import Trajectory
# Assuming mace_calc is already initialized

# Load structure
traj = Trajectory.read(
    dataset="training_set.hdf5",
    key="training/md_sampling/trajectory"
)

# Compute new properties (e.g., feature vectors)
traj.run_neural_network(
    calculator=mace_calc,
    feature_vectors_type="averaged_environments",
    potential_energies=False,
    forces=False
)

# Save only the new feature vectors
traj.save(
    dataset="training_set.hdf5",
    key="training/md_sampling/trajectory",
    only=["feature_vectors"]
)
```

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

For a detailed discussion of performance considerations, see the [Computational Bottlenecks](./06_bottlenecks.md) section.

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

free_energy_config = FreeEnergy.recommended(
    model_name="mace",
    crystal=from_xyz_file(xyz_solid),
    calculator=mace_calc,
    thermal_expansion=False,
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
    amplitude_scan="random",
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
