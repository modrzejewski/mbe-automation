# Design Proposal: `AnySystemType`

## 1. Overview

The goal is to simplify iterating over datasets containing mixed system types (Structures, Trajectories, FiniteSubsystems, etc.) and performing common operations like `run_model` or `subsample` without prior knowledge of the specific type stored in the HDF5 file.

We propose adding a class `AnySystemType` to `mbe_automation.api.classes`. This class will act primarily as a **factory**, using the metadata already stored in HDF5 (specifically the `dataclass` attribute) to instantiate the correct class.

## 2. Proposed Implementation

The `AnySystemType` class will define a static method `read`. It will inspect the HDF5 group attributes to determine the type of object stored and dispatch the read call to the appropriate class (`Structure`, `Trajectory`, `FiniteSubsystem`, or `MolecularCrystal`).

### Code Snippet

This class should be added to `mbe_automation/api/classes.py`.

```python
class AnySystemType:
    """
    A factory class to read any supported system type from a dataset
    based on the stored 'dataclass' attribute.
    """

    @staticmethod
    def read(dataset: str, key: str):
        """
        Reads the object at the given key, automatically determining its type.
        """
        # 1. Peek at the 'dataclass' attribute
        from mbe_automation.storage.file_lock import dataset_file

        # We assume the file exists and is readable.
        with dataset_file(dataset, "r") as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in dataset '{dataset}'")

            group = f[key]
            dataclass_name = group.attrs.get("dataclass")

        # 2. Dispatch to the appropriate class
        # Note: These classes are defined in the same module (mbe_automation.api.classes)
        # If AnySystemType is defined at the end of the file, these are available.

        if dataclass_name == "Structure":
            return Structure.read(dataset, key)
        elif dataclass_name == "Trajectory":
            return Trajectory.read(dataset, key)
        elif dataclass_name == "FiniteSubsystem":
            return FiniteSubsystem.read(dataset, key)
        elif dataclass_name == "MolecularCrystal":
            return MolecularCrystal.read(dataset, key)
        elif dataclass_name == "ForceConstants":
            return ForceConstants.read(dataset, key)
        else:
            raise ValueError(
                f"Unknown or missing 'dataclass' attribute '{dataclass_name}' "
                f"at key '{key}' in dataset '{dataset}'."
            )
```

## 3. Usage Example

This implementation directly supports the workflow requested:

```python
from mbe_automation.api.classes import AnySystemType, Dataset

# Iterate over keys (assuming the user has a way to get keys, e.g. from a Dataset object or list)
dataset_path = "urea.hdf5"

# Example: iterating over keys provided by a Dataset helper or externally
for key in Dataset(dataset_path).with_ground_truth():
    # 1. Polymorphic Read
    x = AnySystemType.read(dataset_path, key)

    # 2. Polymorphic Subsample
    # Works for Structure, Trajectory, FiniteSubsystem, MolecularCrystal
    x_sub = x.subsample(10)

    # 3. Polymorphic Run Model
    # Works for Structure, Trajectory, FiniteSubsystem.
    # Note: Will raise AttributeError if x is a MolecularCrystal (which doesn't support run_model).
    x_sub.run_model(calc)

    # 4. Polymorphic Save
    # All these classes support .save(dataset, key)
    x_sub.save(dataset_path, f"{key}_processed")
```

## 4. Interface Consistency Considerations

For `AnySystemType` to be truly effective, the returned objects must share a common interface (Duck Typing).

*   **`read(dataset, key)`**: Implemented by all classes.
*   **`subsample(n, ...)`**: Implemented by `Structure`, `Trajectory`, `MolecularCrystal`, and `FiniteSubsystem`.
*   **`save(dataset, key, ...)`**:
    *   Implemented by `FiniteSubsystem` and `MolecularCrystal` in `api/classes.py`.
    *   `Structure` and `Trajectory` inherit `save` from their storage parents, but strict consistency in `api/classes.py` is good. The current `Structure.save` in `api/classes.py` is well defined.
*   **`run_model(calc)`**:
    *   Implemented by `Structure`, `Trajectory`, `FiniteSubsystem`.
    *   **Not** implemented by `MolecularCrystal`.

### Recommendation for `MolecularCrystal`
If the dataset contains `MolecularCrystal` objects and the user intends to call `run_model`, they should first convert it, e.g., via `extract_finite_subsystems` or `to_molecular_crystal` (if that made sense, but here it's already one).

If the user's dataset only contains `Structure`, `Trajectory`, and `FiniteSubsystem`, the loop will work seamlessly.

## 5. Type Hinting Support

To help IDEs and static analysis tools understand what `x` is, we can define a Type Alias or Protocol.

```python
from typing import Union

# Union Type for static analysis
AnySystem = Union[Structure, Trajectory, FiniteSubsystem, MolecularCrystal]

# In the AnySystemType class:
class AnySystemType:
    @staticmethod
    def read(dataset: str, key: str) -> AnySystem:
        ...
```

This allows the user to see available methods like `run_model` (with warnings that it might not exist on `MolecularCrystal` if strict checking is enabled).

## 6. Summary

The `AnySystemType` class acts as a lightweight factory that solves the problem by leveraging the existing HDF5 metadata scheme. It requires no changes to the underlying storage format, as the `dataclass` attribute is already being written by `mbe_automation.storage`.
