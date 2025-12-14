import os
import h5py
from typing import Set, List, Literal

# Dataclasses that contain structures which should be ignored
IGNORED_PARENTS = {"FiniteSubsystem", "MolecularCrystal", "ForceConstants"}

FilterCriterion = Literal[
    "periodic",
    "finite",
    "with_delta_learning_data",
    "with_feature_vectors"
]

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
        check_ignored_parents: bool = True,
        filters: List[FilterCriterion] | None = None
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

            if filters:
                for criterion in filters:
                    if criterion == "periodic":
                        if not obj.attrs.get("periodic"):
                            return
                    elif criterion == "finite":
                        if obj.attrs.get("periodic"):
                            return
                    elif criterion == "with_delta_learning_data":
                        if "delta" not in obj:
                            return
                    elif criterion == "with_feature_vectors":
                        if "feature_vectors" not in obj:
                            return

            found_keys.append(name)

    try:
        with h5py.File(dataset, "r") as f:
            f.visititems(visit_func)
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return []

    return found_keys


def list_structures(
        dataset: str,
        filters: List[FilterCriterion] | None = None
) -> list[str]:
    """
    List all keys in a given dataset file which correspond to
    dataclass="Structure" or dataclass="Trajectory|Structure".

    Structures contained within composite objects (FiniteSubsystem,
    MolecularCrystal, ForceConstants) are excluded.
    """
    return _list_objects_by_dataclass(
        dataset,
        {"Structure", "Trajectory|Structure"},
        check_ignored_parents=True,
        filters=filters
    )


def list_trajectories(
        dataset: str,
        filters: List[FilterCriterion] | None = None
) -> list[str]:
    """
    List all keys in a given dataset file which correspond to
    dataclass="Trajectory|Structure".

    Trajectories contained within composite objects are excluded.
    """
    return _list_objects_by_dataclass(
        dataset,
        {"Trajectory|Structure"},
        check_ignored_parents=True,
        filters=filters
    )


def list_finite_subsystems(dataset: str) -> list[str]:
    """
    List all keys in a given dataset file which correspond to
    dataclass="FiniteSubsystem".
    """
    return _list_objects_by_dataclass(
        dataset,
        {"FiniteSubsystem"},
        check_ignored_parents=False
    )
