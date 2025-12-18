# Working with HDF5 Datasets

The `mbe_automation.storage` module handles data persistence using hierarchical dataset files. It provides tools to inspect the file structure, query specific data groups using filters, and perform maintenance operations like copying or deleting data.

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

# Load all keys from the file
keys = DatasetKeys("properties.hdf5")

# Iterate over all keys
for key in keys:
    print(key)

```

### Available Filters

Filters reduce the list of keys to those matching specific criteria.

**Type-Based Filters**

| Filter | Description |
| --- | --- |
| `structures()` | Selects `Structure` and `Trajectory` objects. |
| `trajectories()` | Selects `Trajectory` objects (e.g., from MD simulations). |
| `molecular_crystals()` | Selects `MolecularCrystal` objects (analyzed periodic systems). |
| `finite_subsystems(n)` | Selects `FiniteSubsystem` objects (extracted clusters). Optional `n` filters by cluster size (number of molecules). |
| `force_constants()` | Selects `ForceConstants` objects (phonons). |
| `brillouin_zone_paths()` | Selects `BrillouinZonePath` objects (band structures). |
| `eos_curves()` | Selects `EOSCurves` objects (equations of state). |

**Property-Based Filters**

| Filter | Description |
| --- | --- |
| `periodic()` | Selects systems with periodic boundary conditions. |
| `finite()` | Selects non-periodic systems (e.g., isolated molecules or clusters). |
| `with_feature_vectors()` | Selects entries that contain computed MLIP feature vectors. |
| `with_delta_learning_data()` | Selects entries that contain reference data for delta learning. |

**Path-Based Filters**

| Filter | Description |
| --- | --- |
| `starts_with(prefix)` | Selects keys that start with the given string. |
| `excludes(prefix)` | Selects keys that do **not** start with the given string. |

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

This example selects structures that already possess feature vectors, which is useful when preparing data for machine learning tasks like subsampling.

```python
from mbe_automation import DatasetKeys

dataset = "training_set.hdf5"

# Find finite clusters that have feature vectors
ready_keys = DatasetKeys(dataset).finite_subsystems().with_feature_vectors()

print(f"Found {len(ready_keys)} clusters with precomputed features.")

```

