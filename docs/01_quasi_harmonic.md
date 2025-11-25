# Quasi-Harmonic Calculation

- [Setup](#setup)
- [Phonon calculation](#phonon-calculation)
- [Adjustable parameters](#adjustable-parameters)
- [Function Call Overview](#function-call-overview)
- [Computational Bottlenecks](#computational-bottlenecks)
- [How to read the results](#how-to-read-the-results)
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
# Import Minimum if you need to customize relaxation parameters
from mbe_automation.configs.structure import Minimum
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

## Phonon calculation

The workflow is configured using the `FreeEnergy` class from `mbe_automation.configs.quasi_harmonic`.

```python
# Create custom relaxation settings (optional)
relaxation_config = Minimum(
    cell_relaxation="constant_volume",
    max_force_on_atom_eV_A=1.0E-4
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=from_xyz_file(xyz_solid),
    molecule=from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    work_dir=os.path.join(work_dir, "properties"),
    dataset=os.path.join(work_dir, "properties.hdf5"),
    relaxation=relaxation_config
)
```

The workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.workflows.quasi_harmonic.run(properties_config)
```

## Adjustable parameters

### `FreeEnergy` Class

**Location:** `mbe_automation.configs.quasi_harmonic.FreeEnergy`

| Parameter                       | Description                                                                                                                                                                                            | Default Value                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `crystal`                       | Initial, non-relaxed crystal structure. The geometry of the crystal unit cell is relaxed prior to the calculation of the harmonic properties.                                                                                                                                                            | -                                               |
| `calculator`                    | MLIP calculator for energies and forces.                                                                                                                                                           | -                                               |
| `molecule`                      | Initial, non-relaxed structure of the isolated molecule. If set to `None`, sublimation free energy is not computed.                                                                           | `None`                                          |
| `relaxation`                    | An instance of `Minimum` that configures the geometry relaxation parameters.                                                                                                                       | `Minimum()`                                     |
| `temperatures_K`                | Array of temperatures (in Kelvin) for the calculation.                                                                                                                                              | `np.array([298.15])`                            |
| `supercell_radius`              | Minimum point-periodic image distance in the supercell for phonon calculations (Å).                                                                                                               | `25.0`                                          |
| `supercell_matrix`              | Supercell transformation matrix. If specified, `supercell_radius` is ignored.                                                                                                                      | `None`                                          |
| `supercell_diagonal`            | If `True`, create a diagonal supercell. Ignored if `supercell_matrix` is provided.                                                                                                                 | `False`                                         |
| `supercell_displacement`        | Displacement length (in Å) used for numerical differentiation in phonon calculations.                                                                                                             | `0.01`                                          |
| `fourier_interpolation_mesh`    | Mesh for Brillouin zone integration, specified as a 3-component array or a distance in Å.                                                                                                          | `150.0`                                         |
| `thermal_expansion`             | If `True`, performs volumetric thermal expansion calculations by sampling a range of volumes and fitting an equation of state to the F(V) curve. If `False`, uses the harmonic approximation at a fixed reference volume.                                                                                            | `True`                                          |
| `eos_sampling`                  | Algorithm for generating points on the equilibrium curve: "pressure", "volume", or "uniform_scaling".                                                                                                                  | `"volume"`                                      |
| `volume_range`                  | Scaling factors applied to the reference volume (V0) to sample the F(V) curve.                                                                                                                         | `np.array([0.96, ..., 1.08])`                   |
| `pressure_GPa`                  | External isotropic pressure (in GPa) at which equilibrium properties are computed. If non-zero, the equilibrium cell volume is determined by minimizing the Gibbs free energy, G(V) = F(V) + pV.         | `0.0`                                           |
| `thermal_pressures_GPa`         | Range of thermal, effective isotropic pressures (in GPa) applied during cell relaxation to sample cell volumes. This pressure is added to the external `pressure_GPa`.                                      | `np.array([0.2, ..., -0.6])`                    |
| `equation_of_state`             | Equation of state used to fit the energy/free energy vs. volume curve: "birch_murnaghan", "vinet", "polynomial", or "spline".                                                                                   | `"polynomial"`                                  |
| `imaginary_mode_threshold`      | Threshold (in THz) for detecting imaginary phonon frequencies.                                                                                                                                     | `-0.1`                                          |
| `filter_out_imaginary_acoustic` | If `True`, filters out data points with imaginary acoustic modes before the EOS fit.                                                                                                               | `False`                                         |
| `filter_out_imaginary_optical`  | If `True`, filters out data points with imaginary optical modes before the EOS fit.                                                                                                                | `True`                                          |
| `filter_out_broken_symmetry`    | If `True`, filters out data points where the space group differs from the reference.                                                                                                               | `True`                                          |
| `filter_out_extrapolated_minimum` | If `True`, filters out EOS fits where the free energy minimum is outside the volume sampling interval.                                                                                           | `True`                                          |
| `work_dir`                      | Directory where files are stored at runtime.                                                                                                                                                     | `"./"`                                          |
| `dataset`                       | The main HDF5 file with all data computed for the physical system.                                                                                                                               | `"./properties.hdf5"`                           |
| `root_key`                      | Specifies the root path in the HDF5 dataset where the workflow's output is stored.                                                                                                                       | `"quasi_harmonic"`                              |
| `verbose`                       | Verbosity of the program's output. `0` suppresses warnings.                                                                                                                                      | `0`                                             |
| `save_plots`                    | If `True`, save plots of the simulation results.                                                                                                                                                 | `True`                                          |
| `save_csv`                      | If `True`, save CSV files of the simulation results.                                                                                                                                             | `True`                                          |
| `save_xyz`                      | If `True`, save XYZ files of the simulation results.                                                                                                                                             | `True`                                          |

### `Minimum` Class

**Location:** `mbe_automation.configs.structure.Minimum`

| Parameter                    | Description                                                                                                                                                           | Default Value       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `max_force_on_atom_eV_A`     | Maximum residual force threshold for geometry relaxation (eV/Å).                                                                                                      | `1.0E-4`            |
| `max_n_steps`                | Maximum number of steps in the geometry relaxation.                                                                                                                   | `500`               |
| `cell_relaxation`            | Relaxation of the input structure: "full" (optimizes atomic positions, cell shape, and volume), "constant_volume" (optimizes atomic positions and cell shape at fixed volume), or "only_atoms" (optimizes only atomic positions). | `"constant_volume"` |
| `pressure_GPa`               | External isotropic pressure (in GPa) applied during lattice relaxation.                                                                                               | `0.0`               |
| `symmetrize_final_structure` | If `True`, refines the space group symmetry after each geometry relaxation.                                                                                           | `True`              |
| `symmetry_tolerance_loose`   | Tolerance (in Å) used for symmetry detection for imperfect structures after relaxation.                                                                               | `1.0E-2`            |
| `symmetry_tolerance_strict`  | Tolerance (in Å) used for definite symmetry detection after symmetrization.                                                                                           | `1.0E-5`            |
| `backend`                    | Software used to perform the geometry relaxation: "ase" or "dftb".                                                                                                    | `"ase"`             |
| `algo_primary`               | Primary algorithm for structure relaxation ("PreconLBFGS" or "PreconFIRE"). Referenced only if backend="ase".                                                         | `"PreconLBFGS"`     |
| `algo_fallback`              | Fallback algorithm if the primary relaxation algorithm fails. Referenced only if backend="ase".                                                                       | `"PreconFIRE"`      |

## Function Call Overview

```
+----------------------------------------+
|      workflows.quasi_harmonic          |
|                 run                    |
+----------------------------------------+
                    |
                    |
+----------------------------------------+
|           structure.relax              |   Relaxes the geometry of the
|    isolated_molecule (optional)        |   isolated gas-phase molecule.
+----------------------------------------+
                    |
                    |
+----------------------------------------+
|         dynamics.harmonic.core         |   Computes the vibrational
|          molecular_vibrations          |   frequencies of the molecule.
+----------------------------------------+
                    |
                    |
+----------------------------------------+
|           structure.relax              |   Relaxes the crystal structure
|                crystal                 |   to find the equilibrium volume.
+----------------------------------------+
                    |
                    |
+----------------------------------------+
|          structure.crystal             |   Determines the supercell matrix
|            supercell_matrix            |   for phonon calculations.
+----------------------------------------+
                    |
                    |
+----------------------------------------+
|         dynamics.harmonic.core         |   Computes phonon frequencies and
|                phonons                 |   thermodynamic properties.
+----------------------------------------+
                    |
                    |
+----------------------------------------+
|         dynamics.harmonic.core         |   Determines the equilibrium volume
|           equilibrium_curve            |   at each temperature.
+----------------------------------------+

```

## Computational Bottlenecks

For a detailed discussion of performance considerations, see the [Computational Bottlenecks](./06_bottlenecks.md) section.

## How to read the results

### HDF5 Datasets

The `mbe-automation` program uses the Hierarchical Data Format version 5 (HDF5) for storing large amounts of numerical data. The HDF5 file produced by a workflow contains all the raw and processed data in a hierarchical structure, similar to a file system with folders and files.

### File Structure

You can visualize the structure of the output file using `mbe_automation.storage.tree`.

```python
import mbe_automation

mbe_automation.storage.tree("qha.hdf5")
```

A quasi-harmonic calculation with thermal expansion enabled will produce a file with the following structure:

```
qha.hdf5
└── quasi_harmonic
    ├── eos_interpolated
    ├── eos_sampled
    ├── phonons
    │   ├── crystal[eos:V=1.0000]
    │   ├── crystal[eq:T=300.00]
    │   └── ... (other structures)
    ├── relaxation
    │   ├── crystal[eos:V=1.0000]
    │   ├── crystal[eq:T=300.00]
    │   └── ... (other structures)
    ├── thermodynamics_equilibrium_volume
    └── thermodynamics_fixed_volume
```

- **`eos_sampled`**: Contains the raw data from the equation of state (EOS) calculations at various cell volumes.
- **`eos_interpolated`**: Stores the fitted EOS curves and the calculated free energy minima at each temperature.
- **`phonons`**: Group containing phonon calculations for each structure.
- **`relaxation`**: Group containing the relaxed crystal structures.
- **`thermodynamics_fixed_volume`**: Contains thermodynamic properties calculated at a single, fixed volume.
- **`thermodynamics_equilibrium_volume`**: Contains the final thermodynamic properties calculated at the equilibrium volume for each temperature.

The structures under the `phonons` and `relaxation` groups follow a specific naming scheme:
- **`crystal[opt:...]`**: The relaxed input structure. The keywords after `opt:` indicate which degrees of freedom were included in the minimization of the static electronic energy (e.g., atomic positions, cell shape, cell volume), as determined by the `cell_relaxation` keyword.
- **`crystal[eos:V=...]`**: Structures used to sample the equation of state curve, obtained by relaxing the crystal at a fixed volume.
- **`crystal[eq:T=...]`**: Relaxed structures at the equilibrium volume for a given temperature.

### Reading Thermodynamic Properties

The thermodynamic properties can be read into a `pandas` DataFrame. The final results, including thermal expansion effects, are in the `thermodynamics_equilibrium_volume` group.

```python
import mbe_automation

# Read the thermodynamic data with thermal expansion
df_expansion = mbe_automation.storage.read_data_frame(
    dataset="qha.hdf5",
    key="quasi_harmonic/thermodynamics_equilibrium_volume"
)
print(df_expansion.head())

# Read the thermodynamic data at a fixed volume
df_fixed = mbe_automation.storage.read_data_frame(
    dataset="qha.hdf5",
    key="quasi_harmonic/thermodynamics_fixed_volume"
)
print(df_fixed.head())
```

### Plotting Phonon Band Structure

The phonon band structure for any calculated structure can be plotted using the `band_structure` function.

```python
import mbe_automation

# Plot the phonon band structure for the equilibrium structure at 300 K
mbe_automation.dynamics.harmonic.display.band_structure(
    dataset="qha.hdf5",
    key="quasi_harmonic/phonons/crystal[eq:T=300.00]/brillouin_zone_path",
    save_path="band_structure_300K.png"
)
```

## Complete Input Files

### Python Script (`quasi_harmonic.py`)

```python
import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation.configs.structure import Minimum
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"
work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("path/to/your/model.model"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

# Create custom relaxation settings (optional)
relaxation_config = Minimum(
    cell_relaxation="constant_volume",
    max_force_on_atom_eV_A=1.0E-4
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=from_xyz_file(xyz_solid),
    molecule=from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    work_dir=os.path.join(work_dir, "properties"),
    dataset=os.path.join(work_dir, "properties.hdf5"),
    relaxation=relaxation_config
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

InpScript = "quasi_harmonic.py"
LogFile = "quasi_harmonic.log"

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
