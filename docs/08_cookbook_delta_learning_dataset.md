# Cookbook: Delta Learning Dataset Creation

This cookbook demonstrates how to create a dataset for **Delta Learning** using the MACE framework. Delta learning aims to learn the difference between a high-level target method (e.g., DFTB, DFT, or CCSD(T)) and a lower-level baseline method (e.g., a semi-empirical method or a simpler ML potential).

## Prerequisites

This cookbook assumes you have already:
1.  Performed MD sampling or generated structures (as described in [Molecular Dynamics](./02_molecular_dynamics.md) or the [MD + DFTB Cookbook](./07_cookbook_mace_md_dftb_energies.md)).
2.  Saved these structures and their properties (energies, forces) to an HDF5 file (referred to here as `md_structures.hdf5`).
3.  Have access to a pre-trained baseline model (e.g., `mace-mh-1.model`) and a target calculator (e.g., DFTB+).

## Workflow Overview

1.  **Reference Calculation**: Compute the energies of an isolated molecule using both the baseline and target calculators. This is essential for referencing energies correctly in delta learning.
2.  **Data Loading**: Iterate through the HDF5 dataset to load periodic crystals and finite clusters.
3.  **Data Splitting**: Split the loaded structures into training, validation, and test sets.
4.  **Export**: Export the datasets to MACE-compatible XYZ files with the appropriate delta learning labels.

## Step-by-Step Guide

### 1. Setup and Reference Calculation

First, we set up the calculators and compute the energies for an isolated reference molecule. This provides the `E_atomic_baseline` and `E_atomic_target` needed for the delta learning scheme.

```python
import os
import itertools
import numpy as np
from mace.calculators import MACECalculator

import mbe_automation
from mbe_automation import Structure, Dataset, FiniteSubsystem
from mbe_automation.calculators.dftb import DFTB3_D4

# Configuration
dataset_path = "md_structures.hdf5"
pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])
cluster_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

# Load reference molecule (extracted from the dataset)
# Ensure this key points to a single, relaxed molecule
ref = Structure.read(dataset=dataset_path, key="training/dftb3_d4/structures/molecule[extracted,0,opt:atoms]")

# Initialize Calculators
# Baseline: MACE model
baseline_calc = MACECalculator(
    model_paths=os.path.expanduser("~/models/mace/mace-mh-1.model"),
    head="omol"
)

# Target: DFTB3-D4 (using chemical symbols from reference)
target_calc = DFTB3_D4(ref.to_ase_atoms().get_chemical_symbols())

# Compute reference energies
# This stores E_baseline and E_target in the structure's delta attribute
ref.run_model(calculator=baseline_calc, level_of_theory="delta/baseline")
ref.run_model(calculator=target_calc, level_of_theory="delta/target")
```

### 2. Loading and Splitting Data

We iterate through the thermodynamic conditions (T, p) and cluster sizes to load structures from the HDF5 file. We then split them into training, validation, and test sets.

```python
# Initialize Dataset containers
train_pbc = Dataset()
train_clusters = Dataset()
validate_pbc = Dataset()
validate_clusters = Dataset()
test_pbc = Dataset()
test_clusters = Dataset()

# Loop over Periodic Crystals
for T, p in itertools.product(temperatures_K, pressures_GPa):
    # Read subsampled frames for this condition
    key = f"training/dftb3_d4/crystal[dyn:T={T:.2f},p={p:.5f}]/subsampled_frames"
    struct = Structure.read(dataset=dataset_path, key=key)

    # Split into Train/Val/Test (90%/5%/5%)
    a, b, c = struct.random_split([0.90, 0.05, 0.05])
    train_pbc.append(a)
    validate_pbc.append(b)
    test_pbc.append(c)

    # Loop over Finite Clusters
    for n_molecules in cluster_sizes:
        key_cluster = f"training/dftb3_d4/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}"
        cluster = FiniteSubsystem.read(dataset=dataset_path, key=key_cluster)

        # Split clusters
        a, b, c = cluster.random_split([0.90, 0.05, 0.05])
        train_clusters.append(a)
        validate_clusters.append(b)
        test_clusters.append(c)
```

### 3. Exporting to MACE Format

Finally, we export the collected datasets to XYZ files. The `learning_strategy="delta"` argument ensures the correct energy differences (Target - Baseline) are written to the file.

For the training sets, we also provide the `reference_molecule` and set `reference_energy_type="reference_molecule"`. This subtracts the isolated molecule energy from the total energies, which is crucial for learning interaction energies in molecular crystals.

```python
output_dir = "./datasets_11.12.2025"

# Export PBC Datasets
train_pbc.to_mace_dataset(
    save_path=f"{output_dir}/train_pbc.xyz",
    learning_strategy="delta",
    reference_energy_type="reference_molecule",
    reference_molecule=ref,
)

validate_pbc.to_mace_dataset(
    save_path=f"{output_dir}/validate_pbc.xyz",
    learning_strategy="delta",
)

test_pbc.to_mace_dataset(
    save_path=f"{output_dir}/test_pbc.xyz",
    learning_strategy="delta",
)

# Export Cluster Datasets
train_clusters.to_mace_dataset(
    save_path=f"{output_dir}/train_clusters.xyz",
    learning_strategy="delta",
    reference_energy_type="reference_molecule",
    reference_molecule=ref,
)

validate_clusters.to_mace_dataset(
    save_path=f"{output_dir}/validate_clusters.xyz",
    learning_strategy="delta",
)

test_clusters.to_mace_dataset(
    save_path=f"{output_dir}/test_clusters.xyz",
    learning_strategy="delta",
)
```

## Complete Script

<details>
<summary>delta_learning_dataset.py</summary>

```python
import os
import itertools
import numpy as np
from mbe_automation import Structure, Dataset, FiniteSubsystem
from mace.calculators import MACECalculator
from mbe_automation.calculators.dftb import DFTB3_D4

dataset = "md_structures.hdf5"
pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])
cluster_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

# Load reference molecule
ref = Structure.read(dataset=dataset, key="training/dftb3_d4/structures/molecule[extracted,0,opt:atoms]")

# Setup Calculators
baseline_calc = MACECalculator(model_paths=os.path.expanduser("~/models/mace/mace-mh-1.model"), head="omol")
target_calc = DFTB3_D4(ref.to_ase_atoms().get_chemical_symbols())

# Compute Reference Energies (Baseline & Target)
ref.run_model(calculator=baseline_calc, level_of_theory="delta/baseline")
ref.run_model(calculator=target_calc, level_of_theory="delta/target")

# Initialize Containers
train_pbc = Dataset()
train_clusters = Dataset()
validate_pbc = Dataset()
validate_clusters = Dataset()
test_pbc = Dataset()
test_clusters = Dataset()

# Iterate and Split
for T, p in itertools.product(temperatures_K, pressures_GPa):
    # Process PBC
    struct = Structure.read(
        dataset=dataset,
        key = f"training/dftb3_d4/crystal[dyn:T={T:.2f},p={p:.5f}]/subsampled_frames"
    )
    a, b, c = struct.random_split([0.90, 0.05, 0.05])
    train_pbc.append(a)
    validate_pbc.append(b)
    test_pbc.append(c)

    # Process Clusters
    for n_molecules in cluster_sizes:
        cluster = FiniteSubsystem.read(
            dataset=dataset,
            key=f"training/dftb3_d4/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}"
        )
        a, b, c = cluster.random_split([0.90, 0.05, 0.05])
        train_clusters.append(a)
        validate_clusters.append(b)
        test_clusters.append(c)

# Export Datasets
output_dir = "./datasets_11.12.2025"

train_pbc.to_mace_dataset(
    save_path=f"{output_dir}/train_pbc.xyz",
    learning_strategy="delta",
    reference_energy_type="reference_molecule",
    reference_molecule=ref,
)

validate_pbc.to_mace_dataset(
    save_path=f"{output_dir}/validate_pbc.xyz",
    learning_strategy="delta",
)

test_pbc.to_mace_dataset(
    save_path=f"{output_dir}/test_pbc.xyz",
    learning_strategy="delta",
)

train_clusters.to_mace_dataset(
    save_path=f"{output_dir}/train_clusters.xyz",
    learning_strategy="delta",
    reference_energy_type="reference_molecule",
    reference_molecule=ref,
)

validate_clusters.to_mace_dataset(
    save_path=f"{output_dir}/validate_clusters.xyz",
    learning_strategy="delta",
)

test_clusters.to_mace_dataset(
    save_path=f"{output_dir}/test_clusters.xyz",
    learning_strategy="delta",
)
```

</details>

## Output Explanation

Running this script will generate six XYZ files in the `datasets_11.12.2025` directory (or whichever directory you specified):

*   `train_pbc.xyz`, `validate_pbc.xyz`, `test_pbc.xyz`: Contain periodic crystal structures.
*   `train_clusters.xyz`, `validate_clusters.xyz`, `test_clusters.xyz`: Contain finite molecular clusters.

The `train_*.xyz` files will have their energies adjusted by the reference molecule energy (if `reference_energy_type="reference_molecule"` was used), making them suitable for training MACE to learn interaction energies. All files will contain the delta energies (Target - Baseline) and forces required for training the delta model.
