# Quasi-Harmonic Calculation

- [Setup](#setup)
- [Phonon calculation](#phonon-calculation)
- [Empirical Electronic Energy Correction (EEC)](#empirical-electronic-energy-correction-eec)
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
from mbe_automation.calculators import MACE

import mbe_automation.configs
# Import Minimum if you need to customize relaxation parameters
from mbe_automation.configs.structure import Minimum
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/mace.model")
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
    crystal=Structure.from_xyz_file(xyz_solid),
    molecule=Structure.from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    dataset="properties.hdf5",
    relaxation=relaxation_config,
    volume_range=np.array([0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]),
)
```

### Volume Range

The `volume_range` parameter defines the set of scaling factors applied to the equilibrium volume $V_0$ to sample the equation of state (EOS). The code enforces a minimum number of points depending on the selected `equation_of_state`:

*   **Polynomial**: 3 points
*   **Spline, Vinet, Birch-Murnaghan**: 4 points

While these are the minimums required to run the calculation, a denser sampling is recommended for robust numerical computation of second-order thermodynamic derivatives, such as the coefficient of thermal expansion ($\alpha_V$) and the heat capacity at constant pressure ($C_p$).

The workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.run(properties_config)
```

## Empirical Electronic Energy Correction (EEC)

The Empirical Electronic Energy Correction (EEC) option can be applied to enforce a known reference volume ($V_{\text{ref}}$) at a specific reference temperature ($T_{\text{ref}}$). The EEC contribution is added to the crystal's electronic energy and accounted for in all derived thermodynamic functions.

Two types of EEC are supported:
1.  **Linear**: $E_{\text{el,corrected}} = E_{\text{el}} + \text{param} \cdot (V - V_{\text{ref}})$
2.  **Inverse Volume**: $E_{\text{el,corrected}} = E_{\text{el}} + \text{param} / V$

The parameter (`param`) is calculated analytically using a cubic spline fit of the raw Gibbs free energy vs. volume curve to ensure the corrected equilibrium volume matches $V_{\text{ref}}$ at $T_{\text{ref}}$.

To use EEC, add the `electronic_energy_correction` parameter to the `FreeEnergy` configuration object. You must import the [`EEC`](./03_configuration_classes.md#eec-class) dataclass from `mbe_automation.configs.quasi_harmonic`.

The equation of state must be set to `"spline"` (which is the default) to use the EEC option. Additionally, $T_{\text{ref}}$ must be present in the `temperatures_K` array.

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
from mbe_automation.configs.structure import Minimum
from mbe_automation import Structure

# Import EEC
from mbe_automation.configs.quasi_harmonic import EEC

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/mace.model")

relaxation_config = Minimum(
    cell_relaxation="constant_volume",
    max_force_on_atom_eV_A=1.0E-4
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(xyz_solid),
    molecule=Structure.from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 123.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    dataset="properties.hdf5",
    relaxation=relaxation_config,
    volume_range=np.array([0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]),

    # Enable EEC targeting a reference volume of 145.80 A^3 at 123 K
    electronic_energy_correction=EEC(
        type="inverse_volume",
        T_ref=123.0,
        V_ref=145.80
    ),
    equation_of_state="spline" # Required for EEC
)

mbe_automation.run(properties_config)
```

## Adjustable parameters

Detailed descriptions of the configuration classes can be found in the [Configuration Classes](./03_configuration_classes.md) chapter.

*   **[`FreeEnergy`](./03_configuration_classes.md#freeenergy-class)**: Main configuration for the quasi-harmonic workflow.
*   **[`Minimum`](./03_configuration_classes.md#minimum-class)**: Configuration for geometry optimization.
*   **[`EEC`](./03_configuration_classes.md#eec-class)**: Configuration for the Empirical Electronic Energy Correction (EEC).

## Function Call Overview

```
+----------------------------------------+
|            mbe_automation              |
|                 run                    |
+----------------------------------------+
                    |
                    |
+----------------------------------------+
|      workflows.quasi_harmonic          |
|                 run                    |
+----------------------------------------+
                    |
                    |
+----------------------------------------+
|           structure.clusters           |   Extracts and relaxes unique
|    extract_relaxed_unique_molecules    |   molecules from the crystal.
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

For a detailed discussion of performance considerations, see the [Computational Bottlenecks](./05_bottlenecks.md) section.

## How to read the results

### HDF5 Datasets

The `mbe-automation` program uses the Hierarchical Data Format version 5 (HDF5) for storing large amounts of numerical data. The HDF5 file produced by a workflow contains all the raw and processed data in a hierarchical structure, similar to a file system with folders and files.

### File Structure

You can visualize the structure of the output file using `mbe_automation.tree`.

```python
import mbe_automation

mbe_automation.tree("qha.hdf5")
```

A quasi-harmonic calculation with thermal expansion enabled will produce a file with the following structure:

```
qha.hdf5
└── quasi_harmonic
    ├── eos_interpolated
    ├── eos_sampled
    ├── phonons
    │   ├── brillouin_zone_paths
    │   │   ├── crystal[eq:T=300.00,p=0.00010]
    │   │   └── ...
    │   └── force_constants
    │       ├── crystal[eq:T=300.00,p=0.00010]
    │       └── ...
    ├── structures
    │   ├── crystal[eq:T=300.00,p=0.00010]
    │   ├── crystal[eos:V=1.0000]
    │   └── ... (other structures)
    ├── thermodynamics_equilibrium_volume
    └── thermodynamics_fixed_volume
```

- **`eos_sampled`**: Contains the raw data from the equation of state (EOS) calculations at various cell volumes.
- **`eos_interpolated`**: Stores the fitted EOS curves and the calculated free energy minima at each temperature.
- **`phonons`**: Group containing phonon calculations (force constants and Brillouin zone paths).
- **`structures`**: Group containing geometric data of molecular and crystal structures.
- **`thermodynamics_fixed_volume`**: Contains thermodynamic properties calculated at a single, fixed volume.
- **`thermodynamics_equilibrium_volume`**: Contains the final thermodynamic properties calculated at the equilibrium volume for each temperature.

The structures under the `phonons` and `structures` groups follow a specific naming scheme:
- **`crystal[opt:...]`**: The relaxed input structure. The keywords after `opt:` indicate which degrees of freedom were included in the minimization of the static electronic energy (e.g., atomic positions, cell shape, cell volume), as determined by the `cell_relaxation` keyword.
- **`crystal[eos:V=...]`**: Structures used to sample the equation of state curve, obtained by relaxing the crystal at a fixed volume.
- **`crystal[eq:T=...,p=...]`**: Relaxed structures at the equilibrium volume for a given temperature and external isotropic pressure.

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
    key="quasi_harmonic/phonons/brillouin_zone_paths/crystal[eq:T=300.00,p=0.00010]",
    save_path="band_structure_300K.png"
)
```

## Complete Input Files

### Python Script (`quasi_harmonic.py`)

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
from mbe_automation.configs.structure import Minimum
from mbe_automation.configs.quasi_harmonic import EEC
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/model.model")

# Create custom relaxation settings (optional)
relaxation_config = Minimum(
    cell_relaxation="constant_volume",
    max_force_on_atom_eV_A=1.0E-4
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(xyz_solid),
    molecule=Structure.from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 123.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    dataset="properties.hdf5",
    relaxation=relaxation_config,
    volume_range=np.array([0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]),

    # Enable EEC targeting a reference volume of 145.80 A^3 at 123 K
    electronic_energy_correction=EEC(
        type="inverse_volume",
        T_ref=123.0,
        V_ref=145.80
    ),
    equation_of_state="spline" # Required for EEC
)

mbe_automation.run(properties_config)
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

python quasi_harmonic.py > quasi_harmonic.log 2>&1
```
