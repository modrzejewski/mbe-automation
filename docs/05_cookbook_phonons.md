# Cookbook: Reading Phonon Data

This cookbook demonstrates how to extract force constants, phonon frequencies, and eigenvectors from the HDF5 output of a quasi-harmonic calculation.

## Prerequisites

To perform this analysis, you first need to generate a properties dataset using the quasi-harmonic workflow. Below is a minimal example script (`quasi_harmonic.py`) that generates the required `properties.hdf5` file.

```python
import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("path/to/your/model.model"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=from_xyz_file(xyz_solid),
    temperatures_K=np.array([300.0]),
    calculator=mace_calc,
    dataset=os.path.join(work_dir, "properties.hdf5")
)

mbe_automation.workflows.quasi_harmonic.run(properties_config)
```

## Step-by-step Guide

Once you have the `properties.hdf5` file, you can read the computed force constants and analyze the phonon modes.

### Step 1: Identify the Force Constants Key

Use `mbe_automation.storage.tree` to inspect the file structure and locate the force constants key. Look for groups under `phonons` corresponding to your structure of interest (e.g., equilibrium structure at a specific temperature).

```python
import mbe_automation

mbe_automation.storage.tree("properties.hdf5")
```

Example output:
```
properties.hdf5
└── quasi_harmonic
    ├── phonons
    │   ├── crystal[eq:T=300.00]  <-- This group contains force constants
    │   │   ├── force_constants
    │   │   └── ...
    └── ...
```

### Step 2: Read Force Constants and Compute Modes

The following script reads the force constants for the equilibrium crystal structure at 300 K and computes the frequencies and eigenvectors at the Gamma point ($\mathbf{k} = [0, 0, 0]$).

```python
import numpy as np
import mbe_automation

# Define the dataset path and the key for the structure of interest
dataset_path = "properties.hdf5"
# Key pointing to the group containing 'force_constants'
key = "quasi_harmonic/phonons/crystal[eq:T=300.00]"

# Read the force constants from the HDF5 file
fc = mbe_automation.ForceConstants.read(dataset=dataset_path, key=key)

# Compute frequencies and eigenvectors at the Gamma point
freqs_THz, eigenvecs = fc.frequencies_and_eigenvectors(k_point=np.array([0.0, 0.0, 0.0]))

print("Frequencies (THz):")
print(freqs_THz)

print("\nEigenvectors (shape):")
print(eigenvecs.shape)
```

## Output Explanation

*   **`freqs_THz`**: A 1D NumPy array containing the phonon frequencies in Terahertz (THz).
    *   The size is $3N$, where $N$ is the number of atoms in the supercell.
    *   The first 3 modes are typically acoustic modes with frequencies near zero (at the Gamma point).

*   **`eigenvecs`**: A 2D NumPy array containing the phonon eigenvectors.
    *   Shape: `(3N, 3N)`.
    *   Each **column** `j` corresponds to the eigenvector for the frequency `freqs_THz[j]`.
    *   The components represent the displacements of the atoms in Cartesian coordinates.
