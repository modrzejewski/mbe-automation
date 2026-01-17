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
from mbe_automation.calculators import MACE

import mbe_automation
from mbe_automation.configs.md import ClassicalMD
from mbe_automation.structure.clusters import FiniteSubsystemFilter
from mbe_automation.dynamics.harmonic.modes import PhononFilter
from mbe_automation.configs.training import MDSampling, PhononSampling
from mbe_automation.configs.quasi_harmonic import FreeEnergy
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
mlip_parameter_file = "path/to/your/mace.model"
temperature_K = 298.15
dataset = "training_set.hdf5"

mace_calc = MACE(model_path=mlip_parameter_file)
```

## Step 1: MD Sampling

The first stage generates configurations by running a short molecular dynamics simulation.

```python
md_sampling_config = MDSampling(
    crystal=Structure.from_xyz_file(xyz_solid),
    calculator=mace_calc,
    temperatures_K=np.array([temperature_K]),
    pressures_GPa=np.array([1.0E-4, 1.0]),
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
    dataset=dataset,
    root_key="training/md_sampling"
)
mbe_automation.run(md_sampling_config)
```

## Step 2: Force Constants Model

A quasi-harmonic calculation is performed to obtain the force constants required for the phonon sampling stage.

```python
free_energy_config = FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(xyz_solid),
    calculator=mace_calc,
    thermal_expansion=False,
    supercell_radius=20.0,
    dataset=dataset,
    root_key="training/quasi_harmonic"
)
mbe_automation.run(free_energy_config)
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
    force_constants_key="training/quasi_harmonic/phonons/force_constants/crystal[opt:atoms,shape]",
    amplitude_scan="random",
    time_step_fs=100.0,
    n_frames=20,
    dataset=dataset,
    root_key="training/phonon_sampling"
)
mbe_automation.run(phonon_sampling_config)
```

## Adjustable parameters

Detailed descriptions of the configuration classes can be found in the [Configuration Classes](./03_configuration_classes.md) chapter.

*   **[`MDSampling`](./03_configuration_classes.md#mdsampling-class)**: Configuration for the MD sampling stage.
*   **[`PhononSampling`](./03_configuration_classes.md#phononsampling-class)**: Configuration for the phonon sampling stage.
*   **[`ClassicalMD`](./03_configuration_classes.md#classicalmd-class)**: Configuration for the MD simulation parameters within `MDSampling`.
*   **[`FreeEnergy`](./03_configuration_classes.md#freeenergy-class)**: Configuration for the force constants calculation.
*   **[`Minimum`](./03_configuration_classes.md#minimum-class)**: Configuration for geometry optimization.
*   **[`FiniteSubsystemFilter`](./03_configuration_classes.md#finitesubsystemfilter-class)**: Configuration for extracting molecular clusters.
*   **[`PhononFilter`](./03_configuration_classes.md#phononfilter-class)**: Configuration for selecting phonon modes.

## Subsampling

The purpose of subsampling is to select a diverse set of configurations for training a machine learning potential. By choosing a smaller, representative subset of frames from a larger dataset, you can reduce the computational cost of training while ensuring that the model is exposed to a wide range of atomic environments. The `subsample` method, available for `Structure`, `Trajectory`, `MolecularCrystal`, and `FiniteSubsystem` objects, provides a way to do this.

The subsampling process is based on feature vectors, which are numerical representations of the atomic environments in each frame. To enable subsampling, you must first save the feature vectors to the dataset by setting the [`feature_vectors_type`](#mdsampling-class) parameter to `"atomic_environments"` or `"averaged_environments"` in the `MDSampling` or `PhononSampling` configurations.

### `subsample` Method Parameters

| Parameter   | Description                                                                                                                                       | Default Value                |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| `n`         | The number of frames to select from the dataset.                                                                                                  | -                            |
| `algorithm` | The algorithm to use for subsampling. Options are `"farthest_point_sampling"` and `"kmeans"`. Both methods aim to select a diverse subset of frames by analyzing their feature vectors. | `"farthest_point_sampling"`  |

### Example Usage

The following example demonstrates how to read a `Trajectory` from a dataset, and then use the `subsample` method to select a smaller number of frames.

```python
from mbe_automation import Trajectory

# Read the full trajectory from the dataset
full_trajectory = Trajectory.read(
    dataset="training_set.hdf5",
    key="training/md_sampling/trajectories/crystal[dyn:T=298.15,p=0.00010]"
)

# Subsample the trajectory to select 100 frames
subsampled_trajectory = full_trajectory.subsample(n=100)

# Save the subsampled trajectory to a new key in the dataset
subsampled_trajectory.save(
    dataset="training_set.hdf5",
    key="training/md_sampling/trajectories/crystal[dyn:T=298.15,p=0.00010]_subsampled"
)
```

*Note: In the example above, the key has been updated to reflect the actual output structure of the MD sampling workflow.*

## Updates to an Existing Dataset

The `Structure` and `Trajectory` classes allow updating an existing dataset with new properties, such as feature vectors, potential energies, or forces. This is useful for machine learning workflows where new data needs to be computed for existing geometries.

To update a dataset, first load the structure, run the calculator, and then save the specific properties using the `only` argument in the `save` method.

```python
from mbe_automation import Trajectory
from mbe_automation.calculators import MACE

# Initialize the calculator that will be used to update the structures
#
# Note we assume here that you will be adding feature vectors.
# The `feature_vectors_type` is not available for other calculators
mace_calc = MACE(
    model_path="path/to/your/mace.model",
    head="omol",
)

# Load structure
traj = Trajectory.read(
    dataset="training_set.hdf5",
    key="training/md_sampling/trajectories/crystal[dyn:T=298.15,p=0.00010]"
)

# Compute new properties
#
# Note that we do not want to re-calculate energies and forces,
# so we set the corresponding keywords to `False`.
traj.run(
    calculator=mace_calc,
    feature_vectors_type="averaged_environments",
    energies=False,
    forces=False
)

# Save the new feature vectors to the same location
#
# By default, the save method uses `update_mode="update_properties"`,
# which will add the missing feature vectors to the existing group
# without modifying other data.
traj.save(
    dataset="training_set.hdf5",
    key="training/md_sampling/trajectories/crystal[dyn:T=298.15,p=0.00010]",
)
```

## Function Call Overview

### MD Sampling

```
+------------------------------------+
|          mbe_automation            |
|               run                  |
+------------------------------------+
                    |
                    |
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
|          mbe_automation            |
|               run                  |
+------------------------------------+
                    |
                    |
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

For a detailed discussion of performance considerations, see the [Computational Bottlenecks](./05_bottlenecks.md) section.

## Complete Input Files

### Python Script (`training.py`)

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation
from mbe_automation.configs.md import ClassicalMD
from mbe_automation.structure.clusters import FiniteSubsystemFilter
from mbe_automation.dynamics.harmonic.modes import PhononFilter
from mbe_automation.configs.training import MDSampling, PhononSampling
from mbe_automation.configs.quasi_harmonic import FreeEnergy
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
mlip_parameter_file = "path/to/your/mace.model"
temperature_K = 298.15
dataset = "training_set.hdf5"

mace_calc = MACE(model_path=mlip_parameter_file)

md_sampling_config = MDSampling(
    crystal=Structure.from_xyz_file(xyz_solid),
    calculator=mace_calc,
    features_calculator=mace_calc,
    temperatures_K=np.array([temperature_K]),
    pressures_GPa=np.array([1.0E-4, 1.0]),
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
    dataset=dataset,
    root_key="training/md_sampling"
)
mbe_automation.run(md_sampling_config)

free_energy_config = FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(xyz_solid),
    calculator=mace_calc,
    thermal_expansion=False,
    supercell_radius=20.0,
    dataset=dataset,
    root_key="training/quasi_harmonic"
)
mbe_automation.run(free_energy_config)

phonon_sampling_config = PhononSampling(
    calculator=mace_calc,
    features_calculator=mace_calc,
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
    force_constants_key="training/quasi_harmonic/phonons/force_constants/crystal[opt:atoms,shape]",
    amplitude_scan="random",
    time_step_fs=100.0,
    n_frames=20,
    dataset=dataset,
    root_key="training/phonon_sampling"
)
mbe_automation.run(phonon_sampling_config)
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

python training.py > training.log 2>&1
```
