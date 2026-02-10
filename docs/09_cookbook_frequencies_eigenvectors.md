# Extracting frequencies and eigenvectors of the dynamical matrix

This cookbook demonstrates how to extract force constants, phonon frequencies, and eigenvectors from the HDF5 output of a quasi-harmonic calculation.

## Prerequisites

To perform this analysis, you first need to generate a properties dataset using the quasi-harmonic workflow. Below is a minimal example script (`quasi_harmonic.py`) that generates the required `properties.hdf5` file using the harmonic approximation (no thermal expansion) and a supercell radius of 24 Å.

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation
from mbe_automation import Structure

cif_path = "experiment.cif"

mace_calc = MACE(model_path="mace.model")

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(cif_path),
    temperatures_K=np.array([300.0]),
    calculator=mace_calc,
    supercell_radius=24.0,
    thermal_expansion=False,
    cell_relaxation="only_atoms",
    dataset="properties.hdf5"
)

mbe_automation.run(properties_config)
```

## Step-by-step Guide

Once you have the `properties.hdf5` file, you can read the computed force constants and analyze the phonon modes.

### Step 1: Identify the Force Constants Key

Use `mbe_automation.tree` to inspect the file structure and locate the force constants key. Look for groups under `phonons` corresponding to your structure of interest.

```python
import mbe_automation

mbe_automation.tree("properties.hdf5")
```

Example output:
```
properties.hdf5
└── quasi_harmonic
    ├── phonons
    │   └── force_constants
    │       └── crystal[opt:atoms,shape]  <-- This group contains the data
    │           ├── force_constants (eV∕Å²)
    │           └── supercell_matrix
    ├── structures
    │   └── crystal[opt:atoms,shape]
    └── thermodynamics_fixed_volume
```

### Step 2: Read Force Constants and Compute Modes

The following script reads the force constants for the relaxed crystal structure and computes the frequencies and eigenvectors at the Gamma point ($\mathbf{k} = [0, 0, 0]$). It also verifies the orthonormality of the eigenvectors.

```python
import numpy as np
from mbe_automation import ForceConstants

dataset_path = "properties.hdf5"
key = "quasi_harmonic/phonons/force_constants/crystal[opt:atoms,shape]"

fc = ForceConstants.read(dataset=dataset_path, key=key)
freqs_THz, eigenvecs = fc.frequencies_and_eigenvectors(k_point=np.array([0.0, 0.0, 0.0]))

print("Frequencies (THz):")
print(freqs_THz)

print("\nEigenvectors (shape):")
print(eigenvecs.shape)

identity_check = np.dot(eigenvecs.conj().T, eigenvecs)
is_orthonormal = np.allclose(identity_check, np.eye(len(freqs_THz)))
print(f"\nEigenvectors are orthonormal: {is_orthonormal}")
```

### Step 3: Normal Mode Refinement

You can refine the calculated phonon frequencies to better match the experimental data using the `refine` method, which utilizes the NoMoRe library.

This is possible if:
1.  You have a CIF file containing the experimental data (anisotropic displacement parameters and structure factors).
2.  The corresponding structure was optimized without changing the lattice vectors and cell volume (i.e., using the `cell_relaxation="only_atoms"` in the [Minimum](03_configuration_classes.md#minimum-configuration) configuration class).

```python
import numpy as np
from mbe_automation import ForceConstants

dataset_path = "properties.hdf5"
key = "quasi_harmonic/phonons/force_constants/crystal[opt:atoms,shape]"
cif_path = "experiment.cif"

fc = ForceConstants.read(dataset=dataset_path, key=key)

# Run refinement
# mesh_size should be a list of 3 odd integers
# temperature is extracted automatically from the CIF file
refinement_result = fc.refine(
    cif_path=cif_path,
    mesh_size=[3, 3, 3],
)

# The refinement_result object contains initial and final frequencies,
# as well as initial and final ADPs.
print("Initial Frequencies (THz):", refinement_result.freqs_initial_THz)
print("Refined Frequencies (THz):", refinement_result.freqs_final_THz)
```

The `refine` method will also print a summary table comparing the initial and refined frequencies, as well as the agreement with experimental ADPs.

## Output Explanation

*   **`freqs_THz`**: A 1D NumPy array containing the phonon frequencies (THz).
    *   The size is `3N`, where `N` is the number of atoms in the primitive cell.
    *   The first 3 modes are acoustic modes with frequencies near zero at the Gamma point. It is normal that the acoustic mode frequencies are slightly negative at the Gamma point due to numerical inaccuracies.

*   **`eigenvecs`**: A 2D NumPy array containing the eigenvectors of the dynamical matrix.
    *   Shape: `(3N, 3N)`.
    *   Each **column** `j` corresponds to the eigenvector for the frequency `freqs_THz[j]`.
