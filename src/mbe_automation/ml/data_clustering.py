import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import h5py
import time
import os
import matplotlib.pyplot as plt

def compute_linkage_matrix_hdf5(hdf5_dataset, system_types=["crystals", "molecules"]):
    """
    Compute and store linkage matrices for specified system types in HDF5 file
    
    Parameters:
    -----------
    hdf5_dataset : str
        Path to HDF5 file containing feature data
    system_types : list, default ["crystals", "molecules"]
        List of system types for which to compute linkage matrices
        
    Returns:
    --------
    None
        Linkage matrices are stored directly in the HDF5 file
    """
    
    print(f"Computing linkage matrices for {len(system_types)} system types")
    print(f"HDF5 file: {hdf5_dataset}")

    with h5py.File(hdf5_dataset, 'r+') as f:
        for system_type in system_types:
            print(f"\nProcessing system type: {system_type}")
            hdf5_path = system_type            
            feature_data = f[hdf5_path]['molecular_feature_vectors'][:]            
            print(f"Loaded molecular feature vectors: {feature_data.shape}")
            
            if 'linkage_matrix' in f[hdf5_path]:
                print(f"Linkage matrix already exists for '{system_type}'. Overwriting.")
                del f[hdf5_path]['linkage_matrix']
            
            print(f"Computing single-linkage clustering...")

            linkage_start_time = time.time()
            Z = linkage(feature_data, method='single')
            linkage_time = time.time() - linkage_start_time
            
            print(f"Linkage matrix shape: {Z.shape}")
            print(f"Linkage computation time: {linkage_time/60:.1e} minutes")
            
            f[hdf5_path].create_dataset('linkage_matrix', data=Z, compression='gzip')
            
            print(f"Successfully stored linkage matrix for '{system_type}'")
            
    print(f"\nCompleted linkage matrix computation for all system types.")


def find_representative_frames_hdf5(hdf5_dataset,
                                    target_n_frames={
                                        "crystals": 10,
                                        "molecules": 10
                                    }
                                    ):
    """
    Create data clusters by hierarchical agglomerative clustering
    of feature vectors.

    For each data cluster, compute the representative frame,
    which has a feature vector located closest the average
    feature vector of that data cluster.

    Uses the linkage matrix stored in the HDF5 data file.
    
    Parameters:
    -----------
    hdf5_dataset: str
        Path to HDF5 file containing feature data and linkage matrix
    target_n_frames: dict
        Number of frames to select for each system type
        
    Returns:
    --------
    None
        All results are stored in the HDF5 file
    """

    system_types = target_n_frames.keys()
    with h5py.File(hdf5_dataset, 'r+') as f:
        for system_type in system_types:
            print(f"\nProcessing system type: {system_type}")            
            hdf5_path = system_type

            X = f[hdf5_path]['molecular_feature_vectors'][:]
            n_frames = f[hdf5_path].attrs["n_frames"]
            Z = f[hdf5_path]['linkage_matrix'][:]
            
            print(f"Hierarchical agglomerative clustering")
            print(f"Target number of representative data frames: {min(n_frames, target_n_frames[system_type])}")
            
            frame_to_cluster_map = fcluster(
                Z,
                min(n_frames, target_n_frames[system_type]),
                criterion='maxclust') - 1
            n_representative_frames = len(np.unique(frame_to_cluster_map))
            
            n_similar_frames = np.zeros(n_representative_frames, dtype=int)
            for cluster_id in range(n_representative_frames):
                n_similar_frames[cluster_id] = np.sum(frame_to_cluster_map == cluster_id)            
            sorted_indices = np.argsort(-n_similar_frames)
            
            representative_frames = np.zeros(n_representative_frames, dtype=int)            
            for cluster_id in range(n_representative_frames):
                cluster_indices = np.where(frame_to_cluster_map == sorted_indices[cluster_id])[0]
                cluster_center = np.mean(X[cluster_indices], axis=0)                
                distances_to_center = np.linalg.norm(X[cluster_indices] - cluster_center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances_to_center)]                
                representative_frames[cluster_id] = closest_idx
                n_similar_frames[cluster_id] = np.sum(frame_to_cluster_map == sorted_indices[cluster_id])
            
            if 'n_similar_frames' in f[hdf5_path]:
                del f[hdf5_path]['n_similar_frames']
            if 'representative_frames' in f[hdf5_path]:
                del f[hdf5_path]['representative_frames']
            
            f[hdf5_path].create_dataset('n_similar_frames', data=n_similar_frames, compression='gzip')
            f[hdf5_path].create_dataset('representative_frames', data=representative_frames, compression='gzip')
            f[hdf5_path].attrs['n_representative_frames'] = n_representative_frames
            
            print(f"Selected {n_representative_frames} data frames for '{system_type}'")
    
    print(f"\nCompleted frame selection for all system types.")


def plot_cluster_sizes(hdf5_dataset, training_dir, system_types=["crystals", "molecules"]):
    """
    Create plots showing data cluster index vs number of similar data frames
    for each system type in stacked panels.
    
    Parameters:
    -----------
    hdf5_dataset : str
        Path to HDF5 file containing clustering results
    training_dir : str
        Directory where plots will be saved
    system_types : list, default ["crystals", "molecules"]
        List of system types to plot
        
    Returns:
    --------
    None
        Saves plot to training_dir
    """
    
    n_types = len(system_types)
    fig, axes = plt.subplots(n_types, 1, figsize=(10, 4*n_types), sharex=True)
    
    # Make axes iterable even if there's only one subplot
    if n_types == 1:
        axes = [axes]
    
    with h5py.File(hdf5_dataset, 'r') as f:
        for idx, system_type in enumerate(system_types):
            hdf5_path = system_type            
            if 'n_similar_frames' not in f[hdf5_path]:
                print(f"Warning: No clustering data found for {system_type}")
                continue
            
            n_similar_frames = f[hdf5_path]['n_similar_frames'][:]
            n_representative_frames = len(n_similar_frames)            
            cluster_indices = np.arange(n_representative_frames)
            
            ax = axes[idx]
            ax.plot(cluster_indices, n_similar_frames, 'bo-', linewidth=2, markersize=6)
            
            ax.set_ylabel('Number of Similar Data Frames', fontsize=12)
            ax.set_title(f'{system_type.capitalize()}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            total_frames = np.sum(n_similar_frames)
            ax.text(0.02, 0.95, f'Total frames: {total_frames}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Representative Frame Index', fontsize=12)
    fig.suptitle('Cluster Sizes by System Type', fontsize=16, y=0.995)
    plt.tight_layout()
    filename = os.path.join(training_dir, 'cluster_sizes.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved as {filename}")

