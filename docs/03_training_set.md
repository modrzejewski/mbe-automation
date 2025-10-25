# Training Set Creation

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

### Key Parameters for `MDSampling`:

| Parameter                 | Description                                                                                             | Default Value                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `crystal`                 | The initial crystal structure.                                                                          | -                                  |
| `calculator`              | The MLIP calculator.                                                                                    | -                                  |
| `temperature_K`           | The target temperature (in Kelvin) for the MD simulation.                                               | `298.15`                           |
| `pressure_GPa`            | The target pressure (in GPa) for the MD simulation.                                                     | `1.0E-4`                           |
| `finite_subsystem_filter` | Defines how finite molecular clusters are extracted from the periodic simulation.                       | `closest_to_central_molecule`      |
| `md_crystal`              | An instance of `ClassicalMD` that configures the MD simulation parameters.                              | -                                  |

## Step 2: Quasi-Harmonic Calculation

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
    time_step_fs=100.0,
    n_frames=20,
    work_dir=os.path.join(work_dir, "phonon_sampling"),
    dataset=dataset,
    root_key="training/phonon_sampling"
)
mbe_automation.workflows.training.run(phonon_sampling_config)
```

### Key Parameters for `PhononSampling`:

| Parameter                 | Description                                                                          | Default Value |
| ------------------------- | ------------------------------------------------------------------------------------ | ----------------- |
| `calculator`              | The MLIP calculator.                                                                 | -                 |
| `temperature_K`           | The temperature (in Kelvin) for the phonon sampling.                                 | `298.15`          |
| `phonon_filter`           | Specifies which phonon modes to sample from.                                         | `gamma` point     |
| `force_constants_dataset` | The path to the HDF5 file containing the force constants from the quasi-harmonic step. | `./properties.hdf5` |
| `force_constants_key`     | The key within the HDF5 file where the force constants are stored.                     | -                 |
| `time_step_fs`            | The time step for the trajectory generation.                                         | `100.0`           |
| `n_frames`                | The number of frames to generate for each selected phonon mode.                        | `20`              |

## Programming Aspects

The `run` function in `mbe_automation/workflows/training.py` serves as a dispatcher for the two main sampling methods:

1.  **MD Sampling (`md_sampling`):**
    *   This function is called for `MDSampling` configurations.
    *   It runs an MD simulation using `mbe_automation.dynamics.md.core.run`.
    *   If a `finite_subsystem_filter` is provided, it extracts finite clusters by detecting molecules (`detect_molecules`) and then extracting the subsystems (`extract_finite_subsystem`).
    *   It saves all results to the HDF5 `dataset`.

2.  **Phonon Sampling (`phonon_sampling`):**
    *   This function is called for `PhononSampling` configurations.
    *   It generates a trajectory from phonon modes using `mbe_automation.dynamics.harmonic.modes.trajectory`.
    *   It then detects molecules and extracts finite subsystems from the generated trajectory.
    *   All data is saved to the HDF5 `dataset`.
