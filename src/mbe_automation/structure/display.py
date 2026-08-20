from __future__ import annotations
import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import os
import pymatviz
from pathlib import Path

import mbe_automation.storage

def to_pymatviz(
        structure: mbe_automation.storage.Structure
) -> pymatviz.TrajectoryWidget:

    trajectory_data = []
    has_energy = structure.E_pot is not None

    for i in range(structure.n_frames):
        frame_dict = {
            "structure": mbe_automation.storage.to_pymatgen(structure, frame_index=i),
            "step": i,
        }
        if has_energy:
            frame_dict["energy"] = structure.E_pot[i]
        trajectory_data.append(frame_dict)

    widget = pymatviz.TrajectoryWidget(
        trajectory=trajectory_data,
        display_mode="structure+scatter",
        show_controls=True,
        style="height: 600px;",
        show_force_vectors=False,
        show_bonds=True,
        bonding_strategy="nearest_neighbor",
    )

    return widget


def to_chemiscope(
    structure: mbe_automation.storage.Structure
):
    """
    Create an interactive Chemiscope widget from a Structure object
    for display in a Jupyter Notebook.

    Visualizes the structure or trajectory and its potential energy,
    if available.
    
    Args:
        structure: The Structure object to visualize.
        properties: Optional dictionary of additional custom properties.
    
    Returns:
        A chemiscope.jupyter.Chemiscope widget.
    """
    frames = list(mbe_automation.storage.ASETrajectory(structure))
    if structure.E_pot is not None:
        properties = {
            "index": np.arange(len(frames)),
            "E_pot": {
                'target': 'structure',
                'values': structure.E_pot,
                'description': 'Potential energy per atom',
                'units': 'eV/atom'
            }
        }
    else:
        raise ValueError("Visualization with chemiscope requires properties")
        
    return chemiscope.show(
        frames=frames,
        properties=properties
    )

def plot_cumulative_cluster_count(
    unique_clusters: dict[str, "mbe_automation.structure.clusters.UniqueClusters"],
    save_path: str | Path | None = None
) -> plt.Figure | None:
    """
    Generate and save a multi-panel cumulative cluster count plot vs distance.
    Groups clusters elegantly by len(composition) and plots on a shared Y-axis.
    """
    # Find all unique cluster sizes (excluding monomers and empty clusters)
    sizes = sorted(
        set(
            c.n_molecules
            for c in unique_clusters.values()
            if c.n_molecules > 1 and c.n_clusters_unique > 0
        )
    )
    n_panels = len(sizes)
    
    if n_panels == 0:
        return
        
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]
        
    color_idx = 0
    for ax, size in zip(axes, sizes):
        clusters_of_size = [
            c
            for c in unique_clusters.values()
            if c.n_molecules == size and c.n_clusters_unique > 0
        ]
        base_type_name = clusters_of_size[0].type_string
        
        for clusters in clusters_of_size:
            distances = clusters.characteristic_distances
            
            if size == 2:
                xlabel_dist = r"$\min\; r_{ij}$ (Å)"
            else:
                xlabel_dist = r"$\max\; \min\; r_{ij}$ (Å)"
                
            counts = np.arange(1, len(distances) + 1)
            
            # For a single-component crystal, label the line as 'Dimers' (or base_type).
            # For multi-component, use 'AA', 'AB' etc.
            is_single_component = clusters.n_molecules_unique == 1
            line_label = base_type_name.capitalize() if is_single_component else clusters.composition_string
            
            ax.plot(
                distances,
                counts,
                label=line_label,
                drawstyle="steps-post",
                linewidth=2,
                color=plt.cm.tab10(color_idx % 10)
            )
            color_idx += 1

        ax.set_xlabel(xlabel_dist)
        
        if not (clusters_of_size[0].n_molecules_unique == 1):
            ax.legend(title=base_type_name.capitalize())
        else:
            ax.legend()
            
        ax.grid(True, linestyle="--", alpha=0.7)
        
    axes[0].set_ylabel("Symmetry-unique clusters")
    plt.tight_layout()
    
    if save_path is not None:
        save_dir = Path(save_path).parent
        if str(save_dir) != ".":
            save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        return None
    else:
        return fig
