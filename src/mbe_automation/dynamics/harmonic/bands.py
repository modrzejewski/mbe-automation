"""
Band assignment support module.

This module provides the interface between mbe_automation's Phonopy objects
and nomore_ase's band assignment algorithms. It encapsulates strict interface
requirements using adapters.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING, Any
import phonopy
from mbe_automation.storage.core import ForceConstants
from nomore_ase.optimization.band_assignment import assign_bands
from mbe_automation.storage import views

class PhonopyASEAdapter:
    """
    Adapter to expose Phonopy internal objects as an ASE Phonons-like interface.
    
    This satisfies the duck-typing requirements of nomore_ase.optimization.assign_bands:
    1. get_force_constant() -> np.ndarray
    2. compute_dynamical_matrix(q, D_N) -> np.ndarray
    3. atoms -> object with .cell.reciprocal() and length (n_atoms)
    
    Compatible with units:
    - Force constants: eV/Å²
    - Eigenvalues: eV/(Å²·AMU) -> assign_bands interprets this correctly as energy in eV.
    """
    
    def __init__(self, ph: phonopy.Phonopy):
        self.ph = ph
        # Pre-compute ASE atoms wrapper for accessing cell/reciprocal cell
        self.atoms = views.to_ase(ph.primitive)
        
    def get_force_constant(self) -> npt.NDArray[np.float64]:
        """Return force constants matrix."""
        return self.ph.force_constants
        
    def compute_dynamical_matrix(self, q: npt.ArrayLike, D_N: Any = None) -> npt.NDArray[np.complex128]:
        """
        Compute dynamical matrix at q.
        
        Args:
            q: q-point in fractional coordinates
            D_N: Ignored (Phonopy object already holds force constants)
            
        Returns:
            Dynamical matrix (n_modes, n_modes)
        """
        self.ph.dynamical_matrix.run(q)
        return self.ph.dynamical_matrix.dynamical_matrix

def compute_band_indices(
    phonopy_object: phonopy.Phonopy,
    q_points: npt.NDArray[np.float64],
    q_spacing: float = 0.05,
    k_neighbors: int = 6
) -> npt.NDArray[np.int64]:
    """
    Compute band indices for a set of q-points using path tracing.
    
    Args:
        phonopy_object: Consistently initialized Phonopy object
        q_points: (N_q, 3) list of q-points to assign
        q_spacing: Spacing for path interpolation in Å⁻¹
        k_neighbors: Number of neighbors for graph matching
        
    Returns:
        band_indices: (N_q, N_modes) integer array of band IDs.
        
        The array is shaped as (Number of q-points, Number of modes per q-point).
        The value band_indices[i, j] is the unique ID of the band that mode j at q-point i belongs to.
        
        Example:
            For a system with 3 modes and 2 q-points (q0, q1):
            Input q_points: [q0, q1]
            Returns: [[0, 1, 2], [2, 0, 1]]
            
            This means:
            - At q0: mode 0 -> band 0, mode 1 -> band 1, mode 2 -> band 2
            - At q1: mode 0 -> band 2, mode 1 -> band 0, mode 2 -> band 1
            
            (i.e., band 0 connects mode 0 at q0 to mode 1 at q1)
    """
    adapter = PhonopyASEAdapter(phonopy_object)
    
    flat_indices = assign_bands(
        phonons=adapter,
        q_points=q_points,
        q_spacing=q_spacing,
        k_neighbors=k_neighbors
    )
    
    n_q = len(q_points)
    n_modes = flat_indices.size // n_q
    
    n_atoms = len(adapter.atoms)
    if n_modes != 3 * n_atoms:
        raise ValueError(
            f"Number of modes ({n_modes}) does not match 3 * n_atoms ({3 * n_atoms})."
        )
    
    return flat_indices.reshape(n_q, n_modes)


def reorder_frequencies(
    frequencies: npt.NDArray[np.float64],
    band_indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """
    Reorder frequencies so that column j contains frequencies of band j.
    
    Args:
        frequencies: (n_q, n_bands) array of frequencies.
        band_indices: (n_q, n_bands) array of band indices from compute_band_indices.
        
    Returns:
        (n_q, n_bands) array where reordered[k, b] is the frequency of band b at q-point k.
    """
    n_q, n_bands = frequencies.shape
    
    reordered = np.empty((n_q, n_bands), dtype=np.float64)
    for k in range(n_q):
        reordered[k, band_indices[k]] = frequencies[k]
    
    return reordered
