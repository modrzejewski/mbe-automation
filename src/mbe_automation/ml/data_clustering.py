import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter

def Hierarchical(descriptors, threshold=None):
    """
    Analyze molecular diversity using single-linkage hierarchical clustering
    
    Parameters:
    -----------
    descriptors: numpy array or list
        Feature vectors for each molecule
    threshold: float or None
        Distance threshold to cut the dendrogram
        
    Returns:
    --------
    results: dict
        Clustering results including labels and hierarchy
    """
    X = np.array(descriptors)
    print(f"Analyzing {X.shape[0]} molecules with {X.shape[1]}-dimensional feature vectors")
    print(f"Algorithm: hierarchical agglomerative clustering")
    print(f"Max distance between feature vectors in the same cluster: {threshold:.6f}")
    
    # Perform single-linkage clustering
    Z = linkage(X, method='single')    
    labels = fcluster(Z, threshold, criterion='distance') - 1  # 0-based indexing
    n_clusters = len(np.unique(labels))
    
    # Analyze cluster distribution
    cluster_counts = Counter(labels)

    # Find representative molecules (closest to cluster centroid)
    representatives = {}
    for cluster_id in range(n_clusters):
        # Get molecules in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
            
        # Get cluster center (mean of all points in the cluster)
        cluster_center = np.mean(X[cluster_indices], axis=0)
        
        # Find closest molecule to center
        distances_to_center = np.linalg.norm(X[cluster_indices] - cluster_center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances_to_center)]
        
        representatives[cluster_id] = closest_idx
    
    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'hierarchy': Z,
        'threshold': threshold,
        'cluster_counts': cluster_counts,
        'representatives': representatives
    }

