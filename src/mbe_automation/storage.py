import pandas as pd
from phonopy import Phonopy
import h5py
import numpy as np
import numpy.typing as npt
from collections import namedtuple
import os

FBZPath = namedtuple(
    "FBZPath", 
    [
        "kpoints",
        "frequencies",
        "eigenvectors",
        "path_connections",
        "labels",
        "distances"
    ]
)

EOSCurves = namedtuple(
    "EOSCurves",
    [
        "temperatures",
        "V_sampled",
        "F_sampled",
        "V_interp",
        "F_interp",
        "V_min",
        "F_min"
    ]
)

Structure = namedtuple(
    "Structure",
    [
        "positions",
        "atomic_numbers",
        "masses",
        "cell",
        "n_frames",
        "n_atoms",
        "periodic"
    ]
)


def save_data(
        df,
        dataset,
        key="quasi_harmonic/quasi_harmonic_equilibrium_properties",
        mode="a"
):
    """
    Save a Pandas DataFrame to an HDF5 file, handling various column data types.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - dataset (str): Path to the HDF5 file.
    - key (str): Path to the group in the HDF5 file (e.g., "group/subgroup").
    - mode (str): File mode for h5py.File ('a' for append, 'w' for write/overwrite).

    Returns:
    - None
    """

    df.to_hdf(
        dataset,
        key=key,
        mode=mode,
        format="fixed"
    )


def read_data(
        dataset,
        key="quasi_harmonic/quasi_harmonic_equilibrium_properties"
):
    """
    Read a Pandas DataFrame from an HDF5 file.

    Parameters:
    - dataset (str): Path to the HDF5 file.
    - key (str): Path to the group in the HDF5 file (e.g., "group/subgroup").

    Returns:
    - pd.DataFrame: The reconstructed DataFrame.
    """

    df = pd.read_hdf(
        dataset,
        key=key
        )

    return df


def save_fbz_path(
        phonons: Phonopy,
        dataset: str,
        key: str
):

    band_structure = phonons.band_structure    
    n_segments = len(band_structure.frequencies)
    n_kpoints, n_bands = band_structure.frequencies[0].shape
    
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
            
        group = f.create_group(key)
        group.attrs["n_segments"] = n_segments
        group.attrs["n_kpoints"] = n_kpoints
        group.attrs["n_bands"] = n_bands

        group.create_dataset(
            name="path_connections",
            data=np.array(band_structure.path_connections)
        )
        group.create_dataset(
            name="labels",
            data=np.array([s.encode("utf-8") for s in band_structure.labels]),
            dtype=h5py.string_dtype(encoding="utf-8")
        )
        for i in range(n_segments):
            group.create_dataset(
                name=f"frequencies_{i} (THz)",
                data=band_structure.frequencies[i]                
            )
            group.create_dataset(
                name=f"distances_{i}",
                data=band_structure.distances[i]
            )
            group.create_dataset(
                name=f"kpoints_{i}",
                data=band_structure.qpoints[i]
            )
            group.create_dataset(
                name=f"eigenvectors_{i}",
                data=band_structure.eigenvectors[i]
            )
            

def read_fbz_path(
        dataset: str,
        key: str
):

    with h5py.File(dataset, "r") as f:
        group = f[key]
        n_segments = group.attrs["n_segments"]
        
        path_connections = group["path_connections"][...]
        labels = group["labels"][...].astype(str)

        frequencies, distances, kpoints, eigenvectors = [], [], [], []
        for i in range(n_segments):
            frequencies.append(group[f"frequencies_{i} (THz)"][...])
            distances.append(group[f"distances_{i}"][...])
            kpoints.append(group[f"kpoints_{i}"][...])
            eigenvectors.append(group[f"eigenvectors_{i}"][...])

    return FBZPath(
        kpoints,
        frequencies,
        eigenvectors,
        path_connections,
        labels,
        distances
    )


def display(dataset):
    """
    Print ASCII tree visualization of HDF5 dataset structure.
    
    Parameters:
    -----------
    dataset : str
        Path to HDF5 file
    """
    if not os.path.exists(dataset):
        print(f"Error: File {dataset} not found")
        return
    
    def print_attrs(obj, indent=""):
        """Print attributes of HDF5 object"""
        if obj.attrs:
            print(f"{indent}└── @attributes:")
            for i, (key, value) in enumerate(obj.attrs.items()):
                connector = "├──" if i < len(obj.attrs) - 1 else "└──"
                print(f"{indent}    {connector} {key}: {value}")
    
    def print_tree(name, obj, indent="", is_last=True):
        """Recursively print HDF5 tree structure"""
        connector = "└──" if is_last else "├──"
        
        if isinstance(obj, h5py.Dataset):
            # Print dataset with shape
            print(f"{indent}{connector} {name}{'/':<25} # {obj.shape} - {obj.dtype}")
        else:
            # Print group
            print(f"{indent}{connector} {name}/")
            
            # Get all items in this group
            items = list(obj.items())
            
            # Print datasets first, then attributes
            for i, (key, item) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                next_indent = indent + ("    " if is_last else "│   ")
                print_tree(key, item, next_indent, is_last_item)
            
            # Print attributes after datasets
            if obj.attrs:
                next_indent = indent + ("    " if is_last else "│   ")
                print_attrs(obj, next_indent)
    
    try:
        with h5py.File(dataset, 'r') as f:
            print(f"{os.path.basename(dataset)}")
            
            # Get root level items
            items = list(f.items())
            
            for i, (name, obj) in enumerate(items):
                is_last = (i == len(items) - 1)
                print_tree(name, obj, "", is_last)
            
            # Print root attributes if any
            if f.attrs:
                print_attrs(f, "")
                
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")


def save_eos_curves(
        F_tot_curves,
        temperatures,
        dataset,
        key
):
    """
    Save Helmholtz free energy vs. volume data for multiple temperatures.
    """

    n_temperatures = len(temperatures)
    n_volumes = len(F_tot_curves[0].V_sampled)
    n_interp = 200

    V_sampled = np.zeros((n_temperatures, n_volumes))
    F_sampled = np.zeros((n_temperatures, n_volumes))
    V_min = np.zeros(n_temperatures) 
    F_min = np.zeros(n_temperatures)
    for i, fit in enumerate(F_tot_curves):
        V_sampled[i, :] = fit.V_sampled[:]
        F_sampled[i, :] = fit.F_sampled[:]
        if fit.min_found:
            V_min[i] = fit.V_min
            F_min[i] = fit.F_min
        else:
            V_min[i] = np.nan
            F_min[i] = np.nan

    F_interp = np.zeros((n_temperatures, n_interp))
    V_interp = np.linspace(np.min(V_sampled), np.max(V_sampled), 200)
    for i, fit in enumerate(F_tot_curves):
        F_interp[i, :] = fit.F_interp(V_interp)
        
    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
            
        group = f.create_group(key)
        group.attrs["n_temperatures"] = n_temperatures
        group.attrs["n_volumes"] = n_volumes
        group.attrs["n_interp"] = n_interp

        group.create_dataset(
            name="T (K)",
            data=temperatures
        )
        group.create_dataset(
            name="V_sampled (Å³/unit cell)",
            data=V_sampled
        )
        group.create_dataset(
            name="F_sampled (kJ/mol/unit cell)",
            data=F_sampled
        )
        group.create_dataset(
            name="V_interp (Å³/unit cell)",
            data=V_interp
        )
        group.create_dataset(
            name="F_interp (kJ/mol/unit cell)",
            data=F_interp
        )
        group.create_dataset(
            name="V_min (Å³/unit cell)",
            data=V_min
        )
        group.create_dataset(
            name="F_min (kJ/mol/unit cell)",
            data=F_min
        )

        
def read_eos_curves(
        dataset,
        key
):

    with h5py.File(dataset, "r") as f:
        group = f[key]
        eos_curves = EOSCurves(
            temperatures=group["T (K)"][...],
            V_sampled=group["V_sampled (Å³/unit cell)"][...],
            F_sampled=group["F_sampled (kJ/mol/unit cell)"][...],
            V_interp=group["V_interp (Å³/unit cell)"][...],
            F_interp=group["F_interp (kJ/mol/unit cell)"][...],
            V_min=group["V_min (Å³/unit cell)"][...],
            F_min=group["F_min (kJ/mol/unit cell)"][...]
        )

    return eos_curves


def save_structure(
        dataset: str,
        key: str,
        positions: npt.NDArray[np.floating],
        atomic_numbers: npt.NDArray[np.integer],
        masses: npt.NDArray[np.floating],
        cell: npt.NDArray[np.floating] | None=None):

    if positions.ndim == 2:
        n_frames = 1
        n_atoms = positions.shape[0]
    elif positions.ndim == 3:
        n_frames = positions.shape[0]
        n_atoms = positions.shape[1]
    else:
        raise ValueError(
            f"positions array must have rank 2 or 3, but has rank {positions.ndim}"
        )

    with h5py.File(dataset, "a") as f:
        if key in f:
            del f[key]
        group = f.require_group(key)

        group.create_dataset(
            name="positions (Å)",
            data=positions
        )
        group.create_dataset(
            name="atomic_numbers",
            data=atomic_numbers
        )
        group.create_dataset(
            name="masses (u)",
            data=masses
        )
        is_periodic = (cell is not None)
        if is_periodic:
            group.create_dataset(
                name="cell (Å)",
                data=cell
            )

        group.attrs["n_frames"] = n_frames
        group.attrs["n_atoms"] = n_atoms
        group.attrs["periodic"] = is_periodic


def read_structure(dataset, key):
    
    with h5py.File(dataset, "r") as f:
        group = f[key]
        is_periodic = group.attrs["periodic"]
        structure = Structure(
            positions=group["positions (Å)"][...],
            atomic_numbers=group["atomic_numbers"][...],
            masses=group["masses (u)"][...],
            cell=(group["cell (Å)"][...] if is_periodic else None),
            n_frames=group.attrs["n_frames"],
            n_atoms=group.attrs["n_atoms"],
            periodic=is_periodic
        )
        
    return structure
