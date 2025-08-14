import pandas as pd
import h5py
import numpy as np
from pandas.api.types import is_string_dtype

def save_dataframe(df, hdf5_dataset, group_path="data", mode="a"):
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
    try:
        with h5py.File(hdf5_dataset, mode) as f:
            group = f.require_group(group_path)
            for col in df.columns:
                data = df[col].values
                if is_string_dtype(df[col]):
                    data = np.array(df[col].astype(str)).astype(np.str_)
                    dtype = data.dtype
                elif df[col].dtype == "bool":
                    dtype = np.bool_
                elif np.issubdtype(df[col].dtype, np.integer):
                    dtype = np.int64
                elif np.issubdtype(df[col].dtype, np.floating):
                    dtype = np.float64
                else:
                    print(f"Warning: Skipping column '{col}' with unsupported dtype '{df[col].dtype}'")
                    continue
                if col in group:
                    del group[col]
                group.create_dataset(col, data=data, dtype=dtype)
    except Exception as e:
        print(f"Error saving DataFrame to HDF5: {str(e)}")

        
def read_dataframe(hdf5_dataset, group_path="data"):
    """
    Read a Pandas DataFrame from an HDF5 file.

    Parameters:
    - hdf5_dataset (str): Path to the HDF5 file.
    - group_path (str): Path to the group in the HDF5 file (e.g., "group/subgroup").

    Returns:
    - pd.DataFrame: The reconstructed DataFrame, or None if an error occurs.
    """
    try:
        with h5py.File(hdf5_dataset, "r") as f:
            if group_path not in f:
                raise KeyError(f"Group '{group_path}' not found in {hdf5_dataset}")
            group = f[group_path]
            data = {}
            row_count = None
            for col in group:
                dataset = group[col]
                if row_count is None:
                    row_count = dataset.shape[0]
                elif dataset.shape[0] != row_count:
                    raise ValueError(f"Inconsistent row count for column '{col}': expected {row_count}, got {dataset.shape[0]}")
                if np.issubdtype(dataset.dtype, np.str_):
                    data[col] = dataset[:].astype(str)
                elif np.issubdtype(dataset.dtype, np.bool_):
                    data[col] = dataset[:].astype(bool)
                elif np.issubdtype(dataset.dtype, np.integer):
                    data[col] = dataset[:].astype(np.int64)
                elif np.issubdtype(dataset.dtype, np.floating):
                    data[col] = dataset[:].astype(np.float64)
                else:
                    print(f"Warning: Skipping column '{col}' with unsupported dtype '{dataset.dtype}'")
                    continue
                data[col] = data[col]
            if not data:
                raise ValueError(f"No valid columns found in group '{group_path}'")
            df = pd.DataFrame(data)
            return df
    except Exception as e:
        print(f"Error reading DataFrame from HDF5: {str(e)}")
        return None
