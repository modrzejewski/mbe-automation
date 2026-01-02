# Cookbook: Export to MACE with Atomic Reference Energies

This cookbook demonstrates how to export a dataset of structures to a MACE-compatible training file, including the calculation and inclusion of atomic reference energies (isolated atom energies).

We assume you already have a HDF5 file containing structures with ground truth data (energies and forces) at a specific level of theory (e.g., r2SCAN DFT with def2-tzvpd basis set).

## Prerequisites

*   A HDF5 file (e.g., `structures.hdf5`) containing your structures.
*   Ground truth data stored in the structures at the desired level of theory.
*   The `pyscf` package installed for running DFT calculations.

## Step-by-step Guide

### 1. Read structures

First, we identify the keys for the structures we want to export and read them into a `Dataset` object.

```python
import mbe_automation
from mbe_automation.calculators.pyscf import PySCFCalculator

# Path to your HDF5 file
dataset_path = "structures.hdf5"

# Initialize a Dataset to hold the structures
dataset = mbe_automation.Dataset()

# Filter and read structure keys
# In this example, we select all periodic structures
keys = mbe_automation.DatasetKeys(dataset_path).structures().periodic()

print(f"Found {len(keys)} structures.")

for key in keys:
    # Read the structure
    structure = mbe_automation.Structure.read(dataset_path, key)
    dataset.append(structure)
```

### 2. Generate atomic reference energies

MACE requires the energies of isolated atoms to serve as a baseline for the total energy. We calculate these using the same level of theory as the dataset.

```python
# Define the calculator used for the ground truth
# Ensure the settings match your dataset's level of theory
calculator = PySCFCalculator(
    xc="r2scan",
    basis="def2-tzvpd"
)

# Initialize an AtomicReference object
atomic_reference = mbe_automation.AtomicReference()

# Calculate energies for all unique elements in the dataset
# This runs the calculator for each isolated atom type
print("Calculating atomic reference energies...")
atomic_reference.run(
    atomic_numbers=dataset.unique_elements,
    calculator=calculator
)

print("Atomic energies calculated:")
print(atomic_reference.energies[calculator.level_of_theory])
```

### 3. Export data

Finally, we export the structures and the atomic reference energies to a single XYZ file suitable for training MACE models.

```python
# Define the output path
output_path = "training_data.xyz"

# Export the dataset
# The atomic energies are automatically prepended to the file as "IsolatedAtom" configs
dataset.to_mace_dataset(
    save_path=output_path,
    level_of_theory=calculator.level_of_theory,
    atomic_reference=atomic_reference
)

print(f"Dataset exported to {output_path}")
```

## Output Explanation

The resulting `training_data.xyz` file will start with `IsolatedAtom` configurations for each element (H, C, O, etc.), followed by your structure configurations.

Example structure of the output file:

```
1
config_type=IsolatedAtom REF_energy=-13.6 ...
H 0.0 0.0 0.0

1
config_type=IsolatedAtom REF_energy=-1024.5 ...
C 0.0 0.0 0.0

...

128
config_type=Default REF_energy=-25000.0 ...
C 1.2 0.4 0.0 ...
C 2.4 0.8 0.0 ...
...
```

This file is now ready to be used as input for training a MACE model.
