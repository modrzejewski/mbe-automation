"""
This package provides data structures and I/O operations for handling
simulation and analysis data.
"""

# --- Core data structures and I/O functions ---
from mbe_automation.storage.core import (
    # Data structures
    FBZPath,
    EOSCurves,
    Structure,
    Trajectory,

    # I/O for pandas DataFrames
    save_data_frame,
    read_data_frame,

    # I/O for phonons
    save_fbz_path,
    read_fbz_path,
    save_force_constants,
    read_force_constants,

    # I/O for EOSCurves
    save_eos_curves,
    read_eos_curves,

    # I/O for Structure
    save_structure,
    read_structure,

    # I/O for Trajectory
    save_trajectory,
    read_trajectory,
    
    # Specific readers
    read_gamma_point_eigenvecs,
    
)

# --- Visual summary of dataset files ---
from mbe_automation.storage.display import tree

# --- Views for external libraries ---
from mbe_automation.storage.views import ASETrajectory
from mbe_automation.storage.views import to_ase_atoms
from mbe_automation.storage.views import to_dynasor_mode_projector

# --- Define the public API of the package ---
__all__ = [
    # Core classes
    "FBZPath",
    "EOSCurves",
    "Structure",
    "Trajectory",
    
    # Core I/O functions
    "save_data_frame",
    "read_data_frame",
    "save_fbz_path",
    "read_fbz_path",
    "save_eos_curves",
    "read_eos_curves",
    "save_structure",
    "read_structure",
    "save_trajectory",
    "read_trajectory",
    "read_gamma_point_eigenvecs",
    
    # Visualization of the dataset tree structure
    "tree",
    
    # Views
    "ASETrajectory",
    "to_ase_atoms",
    "to_dynasor_mode_projector"
]


