from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np
import chemiscope

import mbe_automation.ml.core
import mbe_automation.storage

def to_chemiscope(
    structure: mbe_automation.storage.Structure,
) -> chemiscope.jupyter.Chemiscope:
    """
    Create an interactive Chemiscope widget from a Structure object,
    including its first two principal components, for display in a notebook.
    """
    if structure.feature_vectors_type == "none":
        raise ValueError("Your Structure does not contain feature vectors. "
                         "Execute run_neural_network.")

    n_components_chemiscope = 2
    pca_vecs, explained_variance_ratio = mbe_automation.ml.core.pca(
        feature_vectors=structure.feature_vectors,
        feature_vectors_type=structure.feature_vectors_type,
        n_components=n_components_chemiscope
    )

    properties = {
        f"principal component {i+1}": {
            "target": "structure",
            "values": pca_vecs[:, i],
            "description": f"Explained variance: {explained_variance_ratio[i]:.2%}"
        }
        for i in range(n_components_chemiscope)
    }

    if structure.E_pot is not None:
        properties["energy"] = {
            "target": "structure",
            "values": structure.E_pot,
            "units": "eV/atom",
            "description": "Potential energy per atom"
        }

    frames = list(mbe_automation.storage.ASETrajectory(structure))

    return chemiscope.show(
        frames=frames,
        properties=properties
    )


def pca(
    structure: mbe_automation.storage.Structure,
    n_vectors: int = 50,
    plot_type: Literal["2d", "3d"] = "2d",
    save_path: str | None = None,
    subset_size: int | None = None, # New argument
    subsample_algorithm: Literal["farthest_point_sampling", "kmeans"] = "farthest_point_sampling"
) -> None:
    """
    Create plots summarizing the principal component analysis.

    Generates two plots:
    1. A plot of the cumulative percentage of variance explained by the
       first `n_vectors` components.
    2. A scatter plot of frames in the space of the first two or three
       principal components, depending on `plot_type`.
       
    Optionally performs subsampling and highlights the selected points
    on the scatter plot.
    """

    if structure.feature_vectors_type == "none":
        raise ValueError("Your Structure does not contain feature vectors. "
                         "Execute run_neural_network.")

    min_components_for_plot = 3 if plot_type == "3d" else 2
    if n_vectors < min_components_for_plot:
        raise ValueError(
            f"n_vectors must be at least {min_components_for_plot} for a '{plot_type}' plot."
        )

    n_features = structure.feature_vectors.shape[-1]
    max_components = min(
        structure.n_frames,
        n_features
    )
    if n_vectors > max_components:
        n_vectors = max_components
    
    pca_vecs, explained_variance_ratio = mbe_automation.ml.core.pca(
        feature_vectors=structure.feature_vectors,
        feature_vectors_type=structure.feature_vectors_type,
        n_components=n_vectors
    )

    # --- Subsampling logic ---
    subsample_indices = None
    if subset_size is not None:
        if not isinstance(subset_size, int) or subset_size <= 0:
            raise ValueError("subset_size must be a positive integer.")
        if subset_size >= structure.n_frames:
            print("Warning: subset_size >= total frames. No subsampling performed.")
            subset_size = None # Disable subsampling if redundant
        else:
            subsample_indices = mbe_automation.ml.core.subsample(
                feature_vectors=structure.feature_vectors,
                feature_vectors_type=structure.feature_vectors_type,
                n_samples=subset_size,
                algorithm=subsample_algorithm
            )
            print(f"Subsampled {subset_size} frames using '{subsample_algorithm}'.")

    fig = plt.figure(figsize=(8, 12))

    ax1 = fig.add_subplot(2, 1, 1)
    cumulative_variance = np.cumsum(explained_variance_ratio) * 100
    component_numbers = np.arange(1, n_vectors + 1)

    ax1.plot(component_numbers, cumulative_variance, "o-", color="C1")
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Explained Variance (%)")
    ax1.grid(True, linestyle=":", alpha=0.7)
    locator = plt.MaxNLocator(integer=True, nbins="auto")
    ax1.xaxis.set_major_locator(locator)
    ax1.set_xlim(left=0, right=n_vectors + 1)
    ax1.set_ylim(0, 105)

    # --- Scatter Plot Logic Modified ---
    if plot_type == "3d":
        ax2 = fig.add_subplot(2, 1, 2, projection="3d")
        # Plot all points
        ax2.scatter(
            pca_vecs[:, 0], pca_vecs[:, 1], pca_vecs[:, 2],
            alpha=0.3, # Lighter alpha for background points
            edgecolors="k",
            linewidth=0.5,
            s=20, # Smaller size for background points
            label="All Frames"
        )
        # Highlight subsampled points if available
        if subsample_indices is not None:
            ax2.scatter(
                pca_vecs[subsample_indices, 0],
                pca_vecs[subsample_indices, 1],
                pca_vecs[subsample_indices, 2],
                alpha=0.8,
                color="C3", # Use a distinct color (e.g., red)
                edgecolors="k",
                linewidth=0.5,
                s=40, # Larger size for highlighted points
                label=f"Subsampled ({subsample_algorithm})"
            )
            ax2.legend()

        total_explained_variance = np.sum(explained_variance_ratio[:3]) * 100
        ax2.set_xlabel(f"PC 1 ({explained_variance_ratio[0]:.1%})")
        ax2.set_ylabel(f"PC 2 ({explained_variance_ratio[1]:.1%})")
        ax2.set_zlabel(f"PC 3 ({explained_variance_ratio[2]:.1%})")

    else: # plot_type == "2d"
        ax2 = fig.add_subplot(2, 1, 2)
        # Plot all points
        ax2.scatter(
            pca_vecs[:, 0],
            pca_vecs[:, 1],
            alpha=0.3, # Lighter alpha for background points
            edgecolors="k",
            linewidth=0.5,
            s=20, # Smaller size for background points
            label="All Frames"
        )
        # Highlight subsampled points if available
        if subsample_indices is not None:
             ax2.scatter(
                pca_vecs[subsample_indices, 0],
                pca_vecs[subsample_indices, 1],
                alpha=0.8,
                color="C3", # Use a distinct color (e.g., red)
                edgecolors="k",
                linewidth=0.5,
                s=40, # Larger size for highlighted points
                label=f"Subsampled ({subsample_algorithm})"
             )
             ax2.legend()

        total_explained_variance_2d = np.sum(explained_variance_ratio[:2]) * 100
        ax2.set_xlabel(f"Principal Component 1 ({explained_variance_ratio[0]:.1%})")
        ax2.set_ylabel(f"Principal Component 2 ({explained_variance_ratio[1]:.1%})")
        ax2.set_aspect("equal", "box")

    ax2.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
