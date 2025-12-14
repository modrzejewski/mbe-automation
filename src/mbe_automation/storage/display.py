import os
import h5py
from typing import Set

# Dataclasses that contain structures which should be ignored
IGNORED_PARENTS = {"FiniteSubsystem", "MolecularCrystal", "ForceConstants"}

def tree(dataset: str):
    """
    Visualize the tree structure of a dataset file.
    """
    if not os.path.exists(dataset):
        print(f"Error: File '{dataset}' not found.")
        return

    def print_recursive(name, obj, indent="", is_last=True):
        """Recursively print the contents of a group or dataset."""
        
        connector = "└── " if is_last else "├── "
        display_name = name

        if isinstance(obj, h5py.Dataset):
            info = f"shape={obj.shape}, dtype={obj.dtype}"
            print(f"{indent}{connector}{display_name} [{info}]")
        else:
            print(f"{indent}{connector}{display_name}")
            
            child_indent = indent + ("    " if is_last else "│   ")
            
            items = list(obj.items())
            
            if obj.attrs:
                is_last_attr_block = not items
                attr_connector = "└── " if is_last_attr_block else "├── "
                print(f"{child_indent}{attr_connector}@attrs")
                
                attr_indent = child_indent + ("    " if is_last_attr_block else "│   ")
                attr_items = list(obj.attrs.items())
                for i, (key, value) in enumerate(attr_items):
                    is_last_attr = (i == len(attr_items) - 1)
                    connector_attr = "└── " if is_last_attr else "├── "
                    print(f"{attr_indent}{connector_attr}{key}: {value}")
            
            for i, (key, item) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                print_recursive(key, item, child_indent, is_last_item)

    try:
        with h5py.File(dataset, "r") as f:
            print(f"{os.path.basename(dataset)}")
            items = list(f.items())
            for i, (name, obj) in enumerate(items):
                is_last = (i == len(items) - 1)
                print_recursive(name, obj, "", is_last)
    except Exception as e:
        print(f"Error reading dataset file: {e}")


def _list_objects_by_dataclass(
        dataset: str,
        target_dataclasses: Set[str],
        check_ignored_parents: bool = True
) -> list[str]:
    """
    Helper to list keys with specific dataclass attributes.
    """
    if not os.path.exists(dataset):
        print(f"Error: File '{dataset}' not found.")
        return []

    found_keys = []

    def visit_func(name, obj):
        dataclass_attr = obj.attrs.get("dataclass")
        if dataclass_attr in target_dataclasses:
            if check_ignored_parents:
                parent = obj.parent
                parent_dataclass = parent.attrs.get("dataclass")
                if parent_dataclass in IGNORED_PARENTS:
                    return

            found_keys.append(name)

    try:
        with h5py.File(dataset, "r") as f:
            f.visititems(visit_func)
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return []

    return found_keys


def list_structures(dataset: str) -> list[str]:
    """
    List all keys in a given dataset file which correspond to
    dataclass="Structure" or dataclass="Trajectory|Structure".

    Structures contained within composite objects (FiniteSubsystem,
    MolecularCrystal, ForceConstants) are excluded.
    """
    return _list_objects_by_dataclass(
        dataset,
        {"Structure", "Trajectory|Structure"},
        check_ignored_parents=True
    )


def list_trajectories(dataset: str) -> list[str]:
    """
    List all keys in a given dataset file which correspond to
    dataclass="Trajectory|Structure".

    Trajectories contained within composite objects are excluded.
    """
    return _list_objects_by_dataclass(
        dataset,
        {"Trajectory|Structure"},
        check_ignored_parents=True
    )


def list_finite_subsystems(dataset: str) -> list[str]:
    """
    List all keys in a given dataset file which correspond to
    dataclass="FiniteSubsystem".
    """
    # FiniteSubsystems themselves are not usually nested in ignored parents in a way that needs hiding.
    # But for consistency with the pattern (listing top-level or relevant objects),
    # we can default to False or True. Given they are "composite objects" themselves,
    # checking for ignored parents might not be strictly necessary unless we have FS inside FS?
    # Let's disable parent checking for FS unless we know they shouldn't be listed.
    # The requirement for parent checking was specifically for "Structure" objects.
    return _list_objects_by_dataclass(
        dataset,
        {"FiniteSubsystem"},
        check_ignored_parents=False
    )
