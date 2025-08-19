import pandas as pd

def save_dataframe(
        df,
        hdf5_dataset,
        group_path="quasi_harmonic/quasi_harmonic_equilibrium_properties",
        mode="a"
):
    """
    Save a Pandas DataFrame to an HDF5 file, handling various column data types.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - hdf5_dataset (str): Path to the HDF5 file.
    - group_path (str): Path to the group in the HDF5 file (e.g., "group/subgroup").
    - mode (str): File mode for h5py.File ('a' for append, 'w' for write/overwrite).

    Returns:
    - None
    """

    df.to_hdf(
        hdf5_dataset,
        key=group_path,
        mode=mode,
        format="fixed"
    )


def read_dataframe(
        hdf5_dataset,
        group_path="quasi_harmonic/quasi_harmonic_equilibrium_properties"
):
    """
    Read a Pandas DataFrame from an HDF5 file.

    Parameters:
    - hdf5_dataset (str): Path to the HDF5 file.
    - group_path (str): Path to the group in the HDF5 file (e.g., "group/subgroup").

    Returns:
    - pd.DataFrame: The reconstructed DataFrame.
    """

    df = pd.read_hdf(
        hdf5_dataset,
        key=group_path
        )

    return df

    
