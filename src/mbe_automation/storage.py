import pandas as pd
from phonopy import Phonopy
import h5py
import numpy as np
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
            data=np.array([s.encode("utf-8") for s in band_structure.labels])
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


# def save_eos_curves(
#         F_tot_curves,
#         temperatures
# ):
#     """
#     Save Helmholtz free energy vs. volume data for multiple temperatures.
#     """

#     n_temperatures = len(temperatures)
#     n_volumes = len(F_tot_curves[0].V_sampled)
#     n_interp = 200

#     F_min_global = min(np.min(fit.F_exact) for fit in F_tot_curves)
#     for fit in F_tot_curves:
#         if fit.min_found:
#             F_min_global = min(F_min_global, fit.F_min)

#     V_sampled = np.zeros((n_temperatures, n_volumes))
#     F_exact = np.zeros((n_temperatures, n_volumes))
#     F_interp = 
#     for i, fit in enumerate(F_tot_curves):
#         V_sampled[i] = fit.V_sampled
#         F_exact[i] = fit.F_exact
        

#     V_min = np.min(V_sampled)
#     V_max = np.max(V_sampled)

#     V_smooth = np.linspace(V_min_range, V_max_range, 200)
#     F_smooth = fit_result.F_interp(V_smooth)
    

    
#     fig, ax = plt.subplots(figsize=(8, 6))

#     cmap = plt.get_cmap('plasma')
#     min_temp = np.min(temperatures)
#     max_temp = np.max(temperatures)
#     norm = mcolors.Normalize(vmin=min_temp, vmax=max_temp)



#     for fit_result, T in zip(F_tot_curves, temperatures):
#         if not fit_result.min_found:
#             print(f"Skipping plot for T={T} K as no minimum was found.")
#             continue

#         color = cmap(norm(T))

#         ax.scatter(
#             fit_result.V_sampled,
#             fit_result.F_exact - F_min_global,
#             color=color,
#             marker='o',
#             facecolors='none' 
#         )

#         if fit_result.F_interp is not None:
#             V_min_range = np.min(fit_result.V_sampled)
#             V_max_range = np.max(fit_result.V_sampled)
#             V_smooth = np.linspace(V_min_range, V_max_range, 200)
#             F_smooth = fit_result.F_interp(V_smooth)

#             ax.plot(
#                 V_smooth,
#                 F_smooth - F_min_global,
#                 color=color,
#                 linestyle='-'
#             )
            
#     equilibria = np.array([
#         (fit.V_min, fit.F_min) for fit in F_tot_curves if fit.min_found
#     ])
#     if equilibria.size > 0:
#         V_eq = equilibria[:, 0]
#         F_eq = equilibria[:, 1]
    
#         ax.plot(
#             V_eq,
#             F_eq - F_min_global,
#             color='black',
#             linestyle='--',
#             marker='x',
#             label='Equilibrium Path'
#         )
#         ax.legend()


#     ax.set_xlabel("Volume (Å³/unit cell)", fontsize=14)
#     ax.set_ylabel("$F_{tot} - F_{min}$ (kJ/mol/unit cell)", fontsize=14)
#     ax.set_title("Equation of State Curves", fontsize=16)
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.tick_params(labelsize=12)
#     ax.set_ylim(bottom=0)

#     if len(temperatures) > 1:
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#         cbar = fig.colorbar(sm, ax=ax)
#         cbar.set_label('Temperature (K)', fontsize=14)

#     plt.tight_layout()
#     output_path = os.path.join(work_dir, "eos_curves.png")
#     plt.savefig(output_path, dpi=300)
#     plt.close(fig)
