# Quasi-Harmonic Calculation

- [Setup](#setup)
- [Configuration](#configuration)
- [Execution](#execution)
- [Details](#details)
- [Computational Bottlenecks](#computational-bottlenecks)
- [Complete Input Files](#complete-input-files)

This workflow performs a quasi-harmonic calculation of thermodynamic properties, including the free energy, heat capacities, and equilibrium volume as a function of temperature.

## Setup

The initial setup involves importing the necessary modules and defining the system's initial structures and the machine learning interatomic potential (MLIP) calculator.

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

The workflow is configured using the `FreeEnergy` class from `mbe_automation.configs.quasi_harmonic`.

```python
properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.from_template(
    model_name="MACE",
    crystal=from_xyz_file(os.path.join(work_dir, xyz_solid)),
    molecule=from_xyz_file(os.path.join(work_dir, xyz_molecule)),
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    work_dir=os.path.join(work_dir, "properties"),
    dataset=os.path.join(work_dir, "properties.hdf5")
)
```

### Key Parameters:

| Parameter                       | Description                                                                                                                                                                                            | Default Value                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `crystal`                       | Initial, non-relaxed crystal structure.                                                                                                                                                            | -                                               |
| `molecule`                      | Initial, non-relaxed structure of the isolated molecule. If set to `None`, sublimation free energy is not computed.                                                                           | `None`                                          |
| `calculator`                    | MLIP calculator for energies and forces.                                                                                                                                                           | -                                               |
| `thermal_expansion`             | If `True`, performs volumetric thermal expansion calculations. If `False`, uses the harmonic approximation.                                                                                            | `True`                                          |
| `fourier_interpolation_mesh`    | Mesh for Brillouin zone integration, specified as a 3-component array or a distance in Å.                                                                                                          | `150.0`                                         |
| `temperatures_K`                | Array of temperatures (in Kelvin) for the calculation.                                                                                                                                              | `np.array([298.15])`                            |
| `symmetrize_unit_cell`          | If `True`, refines the space group symmetry after each geometry relaxation.                                                                                                                            | `True`                                          |
| `max_force_on_atom`             | Maximum residual force threshold for geometry relaxation (eV/Å). For MLIPs, `5.0E-3` is recommended; for DFT, `1.0E-4`.                                                                            | `1.0E-4`                                        |
| `supercell_radius`              | Minimum point-periodic image distance in the supercell for phonon calculations (Å).                                                                                                               | `25.0`                                          |
| `supercell_displacement`        | Displacement length (in Å) used for numerical differentiation in phonon calculations.                                                                                                             | `0.01`                                          |
| `volume_range`                  | Scaling factors applied to the reference volume (V0) to sample the F(V) curve.                                                                                                                         | `np.array([0.96, ..., 1.08])`                   |
| `pressure_range`                | External isotropic pressures (in GPa) used to sample cell volumes for the equation of state.                                                                                                            | `np.array([0.2, ..., -0.6])`                    |
| `relax_input_cell`              | Relaxation of the input structure: "full", "constant_volume", or "only_atoms".                                                                                                               | `"constant_volume"`                             |
| `equation_of_state`             | Equation of state used to fit the energy/free energy vs. volume curve: "birch_murnaghan", "vinet", or "polynomial".                                                                                   | `"polynomial"`                                  |
| `eos_sampling`                  | Algorithm for generating points on the equilibrium curve: "pressure" or "volume".                                                                                                                  | `"volume"`                                      |
| `imaginary_mode_threshold`      | Threshold (in THz) for detecting imaginary phonon frequencies.                                                                                                                                     | `-0.1`                                          |

## Execution

The workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.workflows.quasi_harmonic.run(properties_config)
```

## Details

The `run` function in `mbe_automation/workflows/quasi_harmonic.py` orchestrates the calculation through the following sequence of operations:

1.  **Initial Relaxation:**
    *   If a `molecule` is provided, it is relaxed using `mbe_automation.structure.relax.isolated_molecule`.
    *   The `crystal` unit cell is relaxed using `mbe_automation.structure.relax.crystal` to find the reference volume, V0.

2.  **Harmonic Approximation (at V0):**
    *   Supercell matrix is determined based on `supercell_radius` using `mbe_automation.structure.crystal.supercell_matrix`.
    *   Phonon properties of the relaxed cell are computed using `mbe_automation.dynamics.harmonic.core.phonons`.

3.  **Thermal Expansion (if `thermal_expansion=True`):**
    *   The equation of state (EOS) curve is determined by calling `mbe_automation.dynamics.harmonic.core.equilibrium_curve`.
    *   For each temperature, the workflow finds the equilibrium volume, creates a new unit cell, relaxes its geometry, and re-calculates the phonon properties.

4.  **Data Storage:**
    *   All results are saved to the HDF5 file specified by the `dataset` parameter.

## Computational Bottlenecks

The computational cost of this workflow is primarily determined by the following parameters:

*   **`supercell_radius`**: This parameter controls the size of the supercell used for phonon calculations. The number of atoms in the supercell, and thus the computational cost, scales with the cube of this radius.
*   **Number of temperatures**: The phonon calculations are repeated for each temperature in the `temperatures_K` array, leading to a linear scaling of the computational cost with the number of temperature points.
*   **`volume_range` / `pressure_range`**: The number of points in these arrays determines how many volumes are sampled to construct the equation of state, adding another linear scaling factor to the overall cost.

## Complete Input Files

### Python Script (`quasi_harmonic.py`)

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

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.from_template(
    model_name="MACE",
    crystal=from_xyz_file(os.path.join(work_dir, xyz_solid)),
    molecule=from_xyz_file(os.path.join(work_dir, xyz_molecule)),
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    work_dir=os.path.join(work_dir, "properties"),
    dataset=os.path.join(work_dir, "properties.hdf5")
)

mbe_automation.workflows.quasi_harmonic.run(properties_config)
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
