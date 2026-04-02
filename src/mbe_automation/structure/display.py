from __future__ import annotations
import chemiscope
import pymatviz
import numpy as np

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

