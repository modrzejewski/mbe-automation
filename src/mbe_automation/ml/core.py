from __future__ import annotations
from typing import Literal, Tuple
import numpy as np
import numpy.typing as npt
import scipy
import sklearn
import sklearn.cluster
import sklearn.metrics
import sklearn.preprocessing
import sklearn.decomposition

SUBSAMPLING_ALGOS = [
    "farthest_point_sampling",
    "kmeans"
]
#
# Type of feature vectors to be saved as a part
# of the Trajectory and Structure objects:
#
# (1) none: no feature vectors are computed and saved
# (2) atomic_environments: feature vectors for every atom
# (3) averaged_environments: feature vectors
#     averaged over all atoms
#
# Options (2) or (3) enable subsampling of frames
# based on the distances in the feature space.
#
# Limitation: options (2) and (3) work only with
# MACE models.
#
FEATURE_VECTOR_TYPES = [
    "none",
    "averaged_environments",
    "atomic_environments"
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
    n_samples: int,
    rng: np.random.Generator
) -> npt.NDArray[np.integer]:
    """Select points using farthest point sampling."""
    
    n_frames = feature_vectors.shape[0]
    selected_indices = np.zeros(n_samples, dtype=int)
    min_distances = np.full(n_frames, np.inf)

    first_index = rng.integers(n_frames)
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
    n_samples: int,
    rng: np.random.Generator
) -> npt.NDArray[np.integer]:
    """Select points using K-means medoid sampling."""

    # Generate a seed for sklearn from the rng
    seed = rng.integers(0, 2**32 - 1)

    kmeans = sklearn.cluster.KMeans(
        n_clusters=n_samples,
        n_init="auto",
        random_state=seed
    ).fit(feature_vectors)    
    centroids = kmeans.cluster_centers_
    closest_indices, _ = sklearn.metrics.pairwise_distances_argmin_min(
        centroids,
        feature_vectors
    )
    
    return np.sort(closest_indices)


def subsample(
        feature_vectors: npt.NDArray[np.floating],
        feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES],
        n_samples: int,
        algorithm: Literal[*SUBSAMPLING_ALGOS] = "farthest_point_sampling",
        rng: np.random.Generator | None = None
) -> npt.NDArray[np.integer]:
    """
    Subsample feature vectors using farthest point sampling or
    K-means clustering (kmeans).
    """
    assert feature_vectors_type != "none"
    
    if rng is None:
        #
        # Random number sequence initialized without seed means
        # that the entropy is taken from the operating system
        # an very function call will result in a different sequence.
        #
        rng = np.random.default_rng()

    n_frames = feature_vectors.shape[0]
    
    if feature_vectors_type == "atomic_environments":
        n_atoms = feature_vectors.shape[1]
        n_features = feature_vectors.shape[2]
        features_averaged = _average_over_atoms(feature_vectors)
        
    elif feature_vectors_type == "averaged_environments":
        features_averaged = feature_vectors

    assert n_frames >= n_samples
    assert algorithm in SUBSAMPLING_ALGOS

    if n_samples == n_frames:
        return np.arange(n_frames, dtype=int)
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaled_features = scaler.fit_transform(features_averaged)

    if algorithm == "farthest_point_sampling":
        selected_frames = _farthest_point_sampling(
            feature_vectors=scaled_features,
            n_samples=n_samples,
            rng=rng
        )
    elif algorithm == "kmeans":
       selected_frames = _kmeans_sampling(
            feature_vectors=scaled_features,
            n_samples=n_samples,
            rng=rng
        )

    return selected_frames


def pca(
        feature_vectors: npt.NDArray[np.floating],
        feature_vectors_type: Literal[*FEATURE_VECTOR_TYPES],
        n_components
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Compute principal components for the feature vectors of a structure.
    
    Returns a tuple containing:
    - The principal vectors stored in columns.
    - The explained variance ratio for each component.
    """

    assert feature_vectors_type != "none"
    
    n_frames = feature_vectors.shape[0]

    if feature_vectors_type == "atomic_environments":
        n_atoms = feature_vectors.shape[1]
        n_features = feature_vectors.shape[2]
        features_averaged = _average_over_atoms(feature_vectors)
        
    elif feature_vectors_type == "averaged_environments":
        n_features = feature_vectors.shape[1]
        features_averaged = feature_vectors
    
    assert n_components > 0
    assert n_components <= min(n_features, n_frames)

    scaler = sklearn.preprocessing.StandardScaler()
    scaled_features = scaler.fit_transform(features_averaged)

    pca_decomposition = sklearn.decomposition.PCA(n_components=n_components)
    principal_vecs = pca_decomposition.fit_transform(scaled_features)
    
    return principal_vecs, pca_decomposition.explained_variance_ratio_
