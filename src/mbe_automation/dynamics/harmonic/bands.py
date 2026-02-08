"""
Band assignment support module.

This module provides the interface between mbe_automation's Phonopy objects
and nomore_ase's band assignment algorithms. It encapsulates strict interface
requirements using adapters.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING, Any, List
import phonopy
from mbe_automation.storage.core import ForceConstants
from nomore_ase.optimization.band_assignment import assign_bands
from mbe_automation.storage import views
from nomore_ase.core.symmetric_phonons import SymmetricPhonons
from mbe_automation.dynamics.harmonic.modes import at_k_points
from mbe_automation.configs.structure import SYMMETRY_TOLERANCE_STRICT

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


def find_degenerate_frequencies(freqs: npt.NDArray[np.float64], tolerance: float = 1e-4) -> List[List[int]]:
    """
    Find index groups of degenerate modes.

    Args:
        freqs: Array of frequencies (should be 1D).
        tolerance: Tolerance for degeneracy (same units as freqs).

    Returns:
        List of lists, where each inner list contains indices of degenerate modes.
        Indices refer to the position in the original `freqs` array.

    Example:
        >>> freqs = np.array([10.0, 10.05, 20.0, 30.0, 30.01, 30.02])
        >>> groups = find_degenerate_frequencies(freqs, tolerance=0.1)
        >>> print(groups)
        [[0, 1], [2], [3, 4, 5]]
        
        In this example:
        - Modes 0 and 1 are degenerate (diff < 0.1).
        - Mode 2 is non-degenerate.
        - Modes 3, 4, and 5 are degenerate.
    """
    n_modes = len(freqs)
    if n_modes == 0:
        return []

    visited = np.zeros(n_modes, dtype=bool)
    groups = []
    
    argsort = np.argsort(freqs)
    
    current_group = [argsort[0]]
    visited[argsort[0]] = True
    
    for i in range(1, n_modes):
        idx = argsort[i]
        prev_idx = argsort[i-1]
        
        if np.abs(freqs[idx] - freqs[prev_idx]) < tolerance:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
        visited[idx] = True
    groups.append(current_group)
    
    return groups


def determine_degenerate_bands(
    phonopy_object: phonopy.Phonopy,
    band_indices: npt.NDArray[np.int64],
    gamma_index: int
) -> List[List[int]]:
    """
    Compute degeneracy groups using nomore_ase's rigorous symmetry analysis.
    
    This replaces the native Phonopy Irreps approach to correctly handle
    acoustic mode splitting at Gamma (and other subtle symmetry cases).
    
    1. Extracts Eigenvectors at Gamma ([0, 0, 0]).
    2. Uses nomore_ase.SymmetricPhonons to find degenerate groups by checking
       eigenvector mixing under symmetry operations.
    3. Maps these Gamma-point groups to all q-points via band indices.
    
    Args:
        phonopy_object: Phonopy object with initialized structure and supercell.
        band_indices: (n_q * n_bands,) or (n_q, n_bands) array of band indices.
                      Values must be in range [0, n_bands-1].
        gamma_index: Index of the Gamma point in the q-point list corresponding to band_indices.
        
    Returns:
        List of lists, where each inner list contains mode indices (into the flattened frequency array)
        that form a degenerate group.
        
        Example:
            Consider a system with 3 bands and 2 q-points (q0, q1), where q0 is Gamma.
            Modes at q0 are indices 0, 1, 2. Modes at q1 are indices 3, 4, 5.
            
            Suppose modes 0 and 1 are degenerate at Gamma (e.g. formed by bands 0 and 1).
            All modes belonging to bands 0 and 1 across the Brillouin zone will be grouped together.
            
            Result:
            [[0, 1, 3, 4], [2, 5]]
            
            - Group 0: Modes from bands 0 and 1 (indices 0, 1 at q0; 3, 4 at q1)
            - Group 1: Modes from band 2 (index 2 at q0; 5 at q1)
    """
    sym_calc = SymmetricPhonons(
        atoms=views.to_ase(phonopy_object.primitive), 
        calculator=None, 
        supercell=None, 
        symprec=SYMMETRY_TOLERANCE_STRICT,
    )

    _, eigenvecs_all = at_k_points(
        phonopy_object.dynamical_matrix, 
        [[0, 0, 0]], 
        compute_eigenvecs=True, 
        eigenvectors_storage="rows"
    )
    
    eigenvecs_gamma = eigenvecs_all[0]
    n_atoms = len(phonopy_object.primitive)
    n_modes = n_atoms * 3
    
    eigenvectors = eigenvecs_gamma.reshape(n_modes, n_atoms, 3)
    gamma_mode_groups = sym_calc.find_degeneracy_groups_by_symmetry(eigenvectors, q_point=(0,0,0))
    
    bands_at_gamma = band_indices[gamma_index]    
    flat_band_indices = band_indices.flatten()
    full_groups = []
    
    for group in gamma_mode_groups:
        degenerate_band_ids = bands_at_gamma[group]
        mask = np.isin(flat_band_indices, degenerate_band_ids)
        full_groups.append(np.flatnonzero(mask).tolist())
        
    return full_groups
