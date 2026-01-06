import numpy as np
import sys
import h5py
from mbe_automation.storage.file_lock import dataset_file

def normalized(feature_vectors, atomic_numbers, feature_vector_mean, feature_vector_sigma):
    """
    Normalize feature vectors using per-element statistics (vectorized implementation).
    
    Parameters:
    -----------
    feature_vectors : np.ndarray
        Shape (n_frames, n_atoms, n_features)
    atomic_numbers : np.ndarray
        Shape (n_frames, n_atoms)
    feature_vector_mean : np.ndarray
        Shape (max_atomic_number+1, n_features)
    feature_vector_sigma : np.ndarray
        Shape (max_atomic_number+1, n_features)
        
    Returns:
    --------
    normalized_feature_vectors : np.ndarray
        Shape (n_frames, n_atoms, n_features)
    """
    # Safe minimum sigma to avoid division by zero
    eps = 1e-8
    
    # Use atomic numbers as indices to broadcast statistics
    means = feature_vector_mean[atomic_numbers]  # shape of the means matrix: (n_frames, n_atoms, n_features)
    sigmas = np.maximum(feature_vector_sigma[atomic_numbers], eps)
    
    return (feature_vectors - means) / sigmas


def normalized_hdf5(hdf5_dataset, system_types, reference_system_type="crystals"):
    """
    Normalize feature vectors in HDF5 dataset using reference system statistics.
    
    Parameters:
    -----------
    hdf5_dataset : str
        Path to the HDF5 data file
    system_types : list
        System types to normalize
    reference_system_type : str, default="crystals"
        System type providing normalization statistics. 
    """
    with dataset_file(hdf5_dataset, 'r+') as f:
        ref_group = f[reference_system_type]
        feature_vector_mean = ref_group['feature_vector_mean'][:]
        feature_vector_sigma = ref_group['feature_vector_sigma'][:]
        
        for system_type in system_types:
            group = f[system_type]
            feature_vectors = group['feature_vectors'][:]
            atomic_numbers = group['atomic_numbers'][:]
            
            normalized_vectors = normalized(
                feature_vectors, atomic_numbers, 
                feature_vector_mean, feature_vector_sigma
            )
            
            if 'normalized_feature_vectors' in group:
                del group['normalized_feature_vectors']
            group.create_dataset('normalized_feature_vectors', data=normalized_vectors)
            
    print(f"All normalized descriptors saved to {hdf5_dataset}")


def molecular(feature_matrix, atomic_numbers):
    """
    Group atoms by the nuclear charge Z and sum the feature vectors
    corresponding to the same Z. The resulting feature vector represents
    the whole molecule or a periodic cell and is useful as a similarity
    measure applied to compare structures from different MD frames.
    
    Parameters:
    -----------
    feature_matrix : np.ndarray, shape (n_frames, n_atoms, n_features)
        Feature matrix where first dim is frames, second dim is atoms, third dim is features
    atomic_numbers : np.ndarray, shape (n_frames, n_atoms)
        Atomic number for each atom in each frame. Since frames represent time evolution
        of the same system, atomic numbers are identical across all frames.
    
    Returns:
    --------
    grouped_features : np.ndarray, shape (n_frames, n_unique_elements*n_features)
        Feature matrix grouped by atomic number
    """
    
    frame0_atomic_numbers = atomic_numbers[0]
    unique_atomic_numbers = np.unique(frame0_atomic_numbers)
    
    n_frames, n_atoms, n_features = feature_matrix.shape
    n_unique = len(unique_atomic_numbers)
    
    grouped_features = np.zeros((n_frames, n_unique, n_features))
    for frame_idx in range(n_frames):
        for i, atomic_num in enumerate(unique_atomic_numbers):
            mask = atomic_numbers[frame_idx] == atomic_num
            grouped_features[frame_idx, i, :] = feature_matrix[frame_idx, mask, :].sum(axis=0)
    
    return grouped_features.reshape(n_frames, n_unique * n_features)


def molecular_hdf5(hdf5_dataset, system_types=["molecules", "crystals"]):
    """
    Compute molecular feature vectors in HDF5 dataset. Molecular feature-vectors
    enable comparisons between structures using permutionally invariant descriptors.
    
    Parameters:
    -----------
    hdf5_dataset : str
        Path to the HDF5 data file
    system_types : list
        System types for which to compute molecular descriptors
    """
    with dataset_file(hdf5_dataset, 'r+') as f:
        for system_type in system_types:
            group = f[system_type]
            normalized_feature_vectors = group['normalized_feature_vectors'][:]
            atomic_numbers = group['atomic_numbers'][:]
            
            molecular_feature_vectors = molecular(
                normalized_feature_vectors, atomic_numbers)
            
            if 'molecular_feature_vectors' in group:
                del group['molecular_feature_vectors']
            group.create_dataset('molecular_feature_vectors', data=molecular_feature_vectors)
            
    print(f"All molecular descriptors saved to {hdf5_dataset}")
