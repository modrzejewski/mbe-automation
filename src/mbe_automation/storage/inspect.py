from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import os
import h5py
from typing import Set, List, Literal, Iterator

@dataclass(kw_only=True)
class DatasetKey:
    key: str
    is_periodic: bool
    level_of_theory: str | None
    has_feature_vectors: bool | None
    has_ground_truth: bool | None
    ground_truth_levels: List[str] | None
    contains_exactly_n_molecules: int | None
    dataclass: Literal[
        "Structure",
        "Trajectory",
        "FiniteSubsystem",
        "MolecularCrystal",
        "ForceConstants",
        "BrillouinZonePath",
        "EOSCurves",
    ]

@dataclass
class DatasetKeys:
    """
    Dataclass intended for filtering of dataset keys according to physical properties.
    """
    _items: List[DatasetKey] = field(default_factory=list)

    def __init__(self, dataset: str | Path | None = None, *, _items: List[DatasetKey] = None):
        """
        Load keys from a dataset file.
        
        Args:
            dataset: Path to the dataset file.
            _items: Internal use only. Pre-populated list of DatasetKey objects.
        """
        if _items is not None:
            self._items = _items
        elif dataset:
            self._items = []
            if Path(dataset).exists():
                self._load_from_file(Path(dataset).resolve())
            else:
                raise FileNotFoundError(f"Error: File '{dataset}' not found.")
        else:
            self._items = []

    def _load_from_file(self, filename: str | Path):
        def _get_levels(group):
            if "levels_of_theory" in group.attrs:
                return list(group.attrs["levels_of_theory"])
            return []

        def visit_func(name, obj):
            dataclass_attr = obj.attrs.get("dataclass")
            if not dataclass_attr:
                return

            if obj.parent and obj.parent.name != '/':
                parent_dataclass = obj.parent.attrs.get("dataclass")
                if parent_dataclass in ["FiniteSubsystem", "MolecularCrystal", "ForceConstants"]:
                    return
            
            ground_truth_levels = None

            if dataclass_attr == "Structure" or dataclass_attr == "Trajectory":
                has_feature_vectors = "feature_vectors" in obj
                has_ground_truth = "ground_truth" in obj
                if has_ground_truth:
                    ground_truth_levels = _get_levels(obj["ground_truth"])
                is_periodic = bool(obj.attrs.get("periodic", False))
                contains_exactly_n_molecules = None
                level_of_theory = obj.attrs.get("level_of_theory")

            elif dataclass_attr == "MolecularCrystal":
                has_feature_vectors = "feature_vectors" in obj["supercell"]
                has_ground_truth = "ground_truth" in obj["supercell"]
                if has_ground_truth:
                    ground_truth_levels = _get_levels(obj["supercell"]["ground_truth"])
                is_periodic = True
                contains_exactly_n_molecules = obj.attrs.get("n_molecules")
                level_of_theory = None

            elif dataclass_attr == "FiniteSubsystem":
                has_feature_vectors = "feature_vectors" in obj["cluster_of_molecules"]
                has_ground_truth = "ground_truth" in obj["cluster_of_molecules"]
                if has_ground_truth:
                    ground_truth_levels = _get_levels(obj["cluster_of_molecules"]["ground_truth"])
                is_periodic = False
                contains_exactly_n_molecules = obj.attrs.get("n_molecules")
                level_of_theory = None

            elif dataclass_attr in ["ForceConstants", "BrillouinZonePath", "EOSCurves"]:
                has_feature_vectors = None
                has_ground_truth = None
                is_periodic = True
                contains_exactly_n_molecules = None
                level_of_theory = None
                
            self._items.append(DatasetKey(
                key=name,
                is_periodic=is_periodic,
                has_feature_vectors=has_feature_vectors,
                has_ground_truth=has_ground_truth,
                ground_truth_levels=ground_truth_levels,
                contains_exactly_n_molecules=contains_exactly_n_molecules,
                dataclass=dataclass_attr,
                level_of_theory=level_of_theory,
            ))

        with h5py.File(filename, "r") as f:
            f.visititems(visit_func)

        return

    def __getitem__(self, index: int | slice) -> str | List[str]:
        """
        Returns the 'key' string instead of the DatasetKey object.
        """
        if isinstance(index, slice):
            return [item.key for item in self._items[index]]
        return self._items[index].key

    def __iter__(self) -> Iterator[str]:
        return (item.key for item in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        if not self._items:
            return "DatasetKeys(count=0, keys=[])"
        keys_formatted = "\n".join(f"    {i.key}" for i in self._items)
        return f"DatasetKeys(count={len(self)}):\n{keys_formatted}"
    
    def _filter(self, func) -> DatasetKeys:
        filtered_items = [item for item in self._items if func(item)]
        return DatasetKeys(_items=filtered_items)

    def structures(self, level_of_theory: str | None = None) -> DatasetKeys:
        if level_of_theory is None:
            return self._filter(lambda x: x.dataclass in ["Structure", "Trajectory"])
        else:
            return self._filter(
                lambda x: (
                    x.dataclass in ["Structure", "Trajectory"] and
                    x.level_of_theory is not None and
                    x.level_of_theory == level_of_theory
                )
            )

    def trajectories(self) -> DatasetKeys:
        return self._filter(lambda x: x.dataclass == "Trajectory")

    def molecular_crystals(self) -> DatasetKeys:
        return self._filter(lambda x: x.dataclass == "MolecularCrystal")

    def finite_subsystems(self, n: int | None = None) -> DatasetKeys:
        return self._filter(lambda x: (
            x.dataclass == "FiniteSubsystem" and
            (n is None or x.contains_exactly_n_molecules == n)
        ))

    def brillouin_zone_paths(self) -> DatasetKeys:
        return self._filter(lambda x: x.dataclass == "BrillouinZonePath")

    def eos_curves(self) -> DatasetKeys:
        return self._filter(lambda x: x.dataclass == "EOSCurves")

    def force_constants(self) -> DatasetKeys:
        return self._filter(lambda x: x.dataclass == "ForceConstants")

    def periodic(self) -> DatasetKeys:
        return self._filter(lambda x: x.is_periodic)

    def finite(self) -> DatasetKeys:
        return self._filter(lambda x: not x.is_periodic)

    def with_feature_vectors(self) -> DatasetKeys:
        return self._filter(lambda x: x.has_feature_vectors)

    def with_ground_truth(self, level_of_theory: str | None = None) -> DatasetKeys:
        if level_of_theory is None:
            return self._filter(lambda x: x.has_ground_truth)
        else:
            return self._filter(lambda x: x.has_ground_truth and x.ground_truth_levels and level_of_theory in x.ground_truth_levels)

    def starts_with(self, root_key: str) -> DatasetKeys:
        """Select keys under a specific root group."""
        prefix = root_key.strip("/") + "/"
        return self._filter(lambda x: x.key.startswith(prefix))

    def excludes(self, root_key: str) -> DatasetKeys:
        """Select keys not under a specific root group."""
        prefix = root_key.strip("/") + "/"
        return self._filter(lambda x: not x.key.startswith(prefix))

def tree(dataset: str):
    """
    Visualize the tree structure of a dataset file.
    """
    if not Path(dataset).exists():
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
            print(Path(dataset).name)
            items = list(f.items())
            for i, (name, obj) in enumerate(items):
                is_last = (i == len(items) - 1)
                print_recursive(name, obj, "", is_last)
    except Exception as e:
        print(f"Error reading dataset file: {e}")
