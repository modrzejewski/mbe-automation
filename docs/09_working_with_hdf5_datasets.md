# Working with HDF5 Datasets

The `mbe_automation.storage` module handles data persistence using hierarchical dataset files. It provides tools to inspect the file structure, query specific data groups using filters, and perform maintenance operations like copying or deleting data.

*   [Inspecting Datasets](#inspecting-datasets)
*   [Querying Data with DatasetKeys](#querying-data-with-datasetkeys)
    *   [Initialization](#initialization)
    *   [Available Filters](#available-filters)
*   [Examples](#examples)
    *   [1. Iterating Over Specific Data Types](#1-iterating-over-specific-data-types)
    *   [2. Processing Finite Subsystems](#2-processing-finite-subsystems)
    *   [3. Dataset Maintenance](#3-dataset-maintenance)
    *   [4. Selecting Data for Training](#4-selecting-data-for-training)

## Inspecting Datasets

The structure of a dataset file can be visualized using the `tree` function. This prints the hierarchy of groups, datasets, and their attributes.

```python
import mbe_automation

# Print the file hierarchy
mbe_automation.tree("properties.hdf5")
```

## Querying Data with `DatasetKeys`

The `DatasetKeys` class provides a programmatic way to iterate over keys in a dataset file. It supports method chaining to filter keys based on data types, physical properties, or naming conventions.

### Initialization

```python
from mbe_automation import DatasetKeys

# Load keys from the file
keys = DatasetKeys("properties.hdf5")

# Inspect the loaded keys
print(keys)
```

### Available Filters

The `DatasetKeys` class provides the following methods for filtering keys. These methods return a new `DatasetKeys` instance, allowing them to be chained together.

**Type-Based Filters**

| Filter | Description |
| --- | --- |
| `DatasetKeys.structures()` | Selects `Structure` and `Trajectory` objects. |
| `DatasetKeys.trajectories()` | Selects `Trajectory` objects (e.g., from MD simulations). |
| `DatasetKeys.molecular_crystals()` | Selects `MolecularCrystal` objects (analyzed periodic systems). |
| `DatasetKeys.finite_subsystems(n)` | Selects `FiniteSubsystem` objects (extracted clusters). Optional `n` filters by cluster size (number of molecules). |
| `DatasetKeys.force_constants()` | Selects `ForceConstants` objects (phonons). |
| `DatasetKeys.brillouin_zone_paths()` | Selects `BrillouinZonePath` objects (band structures). |
| `DatasetKeys.eos_curves()` | Selects `EOSCurves` objects (equations of state). |

**Property-Based Filters**

| Filter | Description |
| --- | --- |
| `DatasetKeys.periodic()` | Selects systems with periodic boundary conditions. |
| `DatasetKeys.finite()` | Selects non-periodic systems (e.g., isolated molecules or clusters). |
| `DatasetKeys.with_feature_vectors()` | Selects entries that contain computed MLIP feature vectors. |
| `DatasetKeys.with_delta_learning_data()` | Selects entries that contain reference data for delta learning. |

**Path-Based Filters**

| Filter | Description |
| --- | --- |
| `DatasetKeys.starts_with(prefix)` | Selects keys that start with the given string. |
| `DatasetKeys.excludes(prefix)` | Selects keys that do **not** start with the given string. |

## Examples

### 1. Iterating Over Specific Data Types

This example iterates over all periodic trajectories in the dataset.

```python
from mbe_automation import DatasetKeys

dataset = "properties.hdf5"

# Select only periodic trajectories
keys = DatasetKeys(dataset).trajectories().periodic()

for key in keys:
    print(f"Found trajectory: {key}")
```

### 2. Processing Finite Subsystems

This example selects finite subsystems with exactly 2 molecules that belong to a specific MD run (identified by a key prefix). It then loads each cluster and performs an operation.

```python
from mbe_automation import FiniteSubsystem, DatasetKeys

dataset = "training_set.hdf5"
prefix = "training/md_sampling"

# Chain filters: Subsystems -> Size 2 -> Start with prefix
for key in DatasetKeys(dataset).finite_subsystems(n=2).starts_with(prefix):
    cluster = FiniteSubsystem.read(dataset=dataset, key=key)
    # ... perform analysis or calculation ...
```

### 3. Dataset Maintenance

This example demonstrates how to delete specific groups of data using filters.

```python
import mbe_automation
from mbe_automation import DatasetKeys

dataset = "old_data.hdf5"

# Delete all molecular crystal analysis data
for key in DatasetKeys(dataset).molecular_crystals():
    print(f"Deleting: {key}")
    mbe_automation.storage.delete(dataset=dataset, key=key)
```

### 4. Selecting Data for Training

This example selects finite clusters that have feature vectors, subsamples them to reduce the dataset size, and exports the result to a MACE-compatible training file.

```python
from mbe_automation import DatasetKeys, FiniteSubsystem, Dataset

dataset_file = "training_set.hdf5"

# 1. Select keys: Finite subsystems with precomputed feature vectors
keys = DatasetKeys(dataset_file).finite_subsystems().with_feature_vectors()

training_dataset = Dataset()

for key in keys:
    # 2. Read the subsystem
    subsystem = FiniteSubsystem.read(dataset=dataset_file, key=key)

    # 3. Subsample: Select 10 diverse frames using farthest point sampling
    subsampled_subsystem = subsystem.subsample(n=10)

    # 4. Add to the dataset collection
    training_dataset.append(subsampled_subsystem)

# 5. Export the aggregated dataset to MACE XYZ format
training_dataset.to_mace_dataset(
    save_path="training_data.xyz",
    learning_strategy="direct"
)

print(f"Exported {len(training_dataset.structures)} subsampled structures to 'training_data.xyz'.")
```
