# Cookbook: Adding Atomic Reference Energies to Existing MACE Training Data

This cookbook demonstrates how to export a dataset of structures to a MACE-compatible training file, including the calculation and inclusion of atomic reference energies (isolated atom energies).

We assume you already have a HDF5 file containing structures with ground truth data (energies and forces) at a specific level of theory (e.g., r2SCAN DFT with def2-tzvpd basis set).

- [Prerequisites](#prerequisites)
- [1. Read structures](#1-read-structures)
- [2. Generate atomic reference energies](#2-generate-atomic-reference-energies)
- [3. Export data](#3-export-data)
- [Output Explanation](#output-explanation)

## Prerequisites

*   A HDF5 file (e.g., `structures.hdf5`) containing your structures.
*   Ground truth data stored in the structures at the desired level of theory.
*   The `pyscf` package installed for running DFT calculations.

## 1. Read structures

First, we identify the keys for the structures we want to export and read them into a `Dataset` object.

```python
from mbe_automation import Dataset, DatasetKeys, FiniteSubsystem
from mbe_automation.calculators import DFT

# Define the calculator used for the ground truth
# Ensure the settings match your dataset's level of theory
calculator = DFT(
    model_name="r2scan-d4",
    basis="def2-tzvpd"
)

# Print the level of theory string
print(f"Level of theory: {calculator.level_of_theory}")

# Path to your HDF5 file
dataset_path = "structures.hdf5"

# Initialize a Dataset to hold the structures
dataset = Dataset()

# Filter and read structure keys
# In this example, we select finite subsystems with data at the specified level of theory
keys = DatasetKeys(dataset_path).finite_subsystems().with_ground_truth(calculator.level_of_theory)

print(f"Found {len(keys)} structures.")

for key in keys:
    # Read the structure
    structure = FiniteSubsystem.read(dataset_path, key)
    dataset.append(structure)
```

## 2. Generate atomic reference energies

MACE requires the energies of isolated atoms to serve as a baseline for the total energy. We calculate these using the same level of theory as the dataset.

```python
# Calculate energies for all unique elements in the dataset
# This runs the calculator for each isolated atom type
print("Calculating atomic reference energies...")
atomic_reference = dataset.atomic_reference(calculator)

print("Atomic energies calculated:")
print(atomic_reference.energies[calculator.level_of_theory])
```

## 3. Export data

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

The resulting `training_data.xyz` file will start with `IsolatedAtom` configurations for each element (H, C, O, etc.), followed by your training configurations.

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
