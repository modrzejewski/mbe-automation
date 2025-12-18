from pathlib import Path
import h5py

def copy(
    dataset_source: Path | str,
    key_source: str,
    dataset_target: Path | str,
    key_target: str,
    overwrite: bool = False
):
    """Copy group or dataset from source storage to target."""
    path_source = Path(dataset_source).resolve()
    path_target = Path(dataset_target).resolve()
    is_same_file = path_source == path_target

    with h5py.File(path_source, "a" if is_same_file else "r") as source:
        if is_same_file:
            if overwrite and key_target in source:
                del source[key_target]
            source.copy(key_source, key_target)
        else:
            with h5py.File(path_target, "a") as target:
                if overwrite and key_target in target:
                    del target[key_target]
                source.copy(key_source, target, name=key_target)

def rename(
    dataset: Path | str,
    key_old: str,
    key_new: str,
    overwrite: bool = False
):
    """Rename a data group in a dataset storage file."""
    path = Path(dataset).resolve()
    with h5py.File(path, "a") as f:
        if overwrite and key_new in f:
            del f[key_new]
        f.move(key_old, key_new)

def delete(
    dataset: Path | str,
    key: str
):
    """Delete a data group from storage."""
    path = Path(dataset).resolve()
    with h5py.File(path, "a") as f:
        del f[key]
