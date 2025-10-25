from __future__ import annotations
from typing import Literal, Tuple
import numpy as np
import numpy.typing as npt
import scipy
import sklearn

SUBSAMPLING_ALGOS = [
    "farthest_point_sampling",
    "kmeans"
]

def _average_over_atoms(
    feature_vectors: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Average feature vectors over the atomic dimension
    to get one feature vector per configuration
    of a physical system.
    
    """
    return np.mean(feature_vectors, axis=1)


def _farthest_point_sampling(
    feature_vectors: npt.NDArray[np.floating],
    n_samples: int
) -> npt.NDArray[np.integer]:
    """Select points using farthest point sampling."""
    
    n_frames = feature_vectors.shape[0]
    selected_indices = np.zeros(n_samples, dtype=int)
    min_distances = np.full(n_frames, np.inf)

    first_index = np.random.randint(n_frames)
    selected_indices[0] = first_index

    for i in range(1, n_samples):
        last_selected_point = feature_vectors[selected_indices[i-1], :].reshape(1, -1)
        distances_to_last = scipy.spatial.distance.cdist(last_selected_point, feature_vectors)[0]
        min_distances = np.minimum(min_distances, distances_to_last)
        next_index = np.argmax(min_distances)
        selected_indices[i] = next_index
        
    return np.sort(selected_indices)


def _kmeans_sampling(
    feature_vectors: npt.NDArray[np.floating],
    n_samples: int
) -> npt.NDArray[np.integer]:
    """Select points using K-means medoid sampling."""

    kmeans = sklearn.cluster.KMeans(
        n_clusters=n_samples,
        n_init="auto",
        random_state=42
    ).fit(feature_vectors)    
    centroids = kmeans.cluster_centers_
    closest_indices, _ = sklearn.metrics.pairwise_distances_argmin_min(
        centroids,
        feature_vectors
    )
    
    return np.sort(closest_indices)


def subsample(
        feature_vectors: npt.NDArray[np.floating],
        n_samples: int,
        algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling"
) -> npt.NDArray[np.integer]:
    """
    Subsample feature vectors using farthest point sampling or
    K-means clustering (kmeans).
    """
    
    n_frames = feature_vectors.shape[0]
    n_atoms = feature_vectors.shape[1]
    n_features = feature_vectors.reshape(n_frames, n_atoms, -1).shape[2]
    features_averaged = _average_over_atoms(feature_vectors.reshape(n_frames, n_atoms, n_features))
    
    assert n_frames >= n_samples
    assert algorithm in SUBSAMPLING_ALGOS

    if n_samples == n_frames:
        return np.arange(n_frames, dtype=int)
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaled_features = scaler.fit_transform(features_averaged)

    if algorithm == "farthest_point_sampling":
        selected_frames = _farthest_point_sampling(
            feature_vectors=scaled_features,
            n_samples=n_samples
        )
    elif algorithm == "kmeans":
       selected_frames = _kmeans_sampling(
            feature_vectors=scaled_features,
            n_samples=n_samples
        )

    return selected_frames


def pca(
        feature_vectors: npt.NDArray[np.floating],
        n_components
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Compute principal components for the feature vectors of a structure.
    
    Returns a tuple containing:
    - The principal vectors stored in columns.
    - The explained variance ratio for each component.
    """

    n_frames = feature_vectors.shape[0]
    n_atoms = feature_vectors.shape[1]
    n_features = feature_vectors.reshape(n_frames, n_atoms, -1).shape[2]
    features_averaged = _average_over_atoms(feature_vectors.reshape(n_frames, n_atoms, n_features))

    assert n_components > 0
    assert n_components <= min(n_features, n_frames)

    scaler = sklearn.preprocessing.StandardScaler()
    scaled_features = scaler.fit_transform(features_averaged)

    pca_decomposition = sklearn.decomposition.PCA(n_components=n_components)
    principal_vecs = pca_decomposition.fit_transform(scaled_features)
    
    return principal_vecs, pca_decomposition.explained_variance_ratio_
