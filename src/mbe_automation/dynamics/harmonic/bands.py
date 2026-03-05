"""
Band assignment support module.

This module provides the interface between mbe_automation's Phonopy objects
and nomore_ase's band assignment algorithms.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any, List, Literal
import phonopy
from mbe_automation.storage import views
from mbe_automation.configs.structure import SYMMETRY_TOLERANCE_STRICT
from mbe_automation.dynamics.harmonic.modes import at_k_points

DEFAULT_Q_SPACING = 0.05  # Å⁻¹
DEFAULT_DEGENERATE_FREQS_TOL = 0.5  # cm⁻¹

try:
    from nomore_ase.optimization.band_assignment import assign_bands
    from nomore_ase.core.symmetric_phonons import SymmetricPhonons
    _NOMORE_AVAILABLE = True
except ImportError:
    _NOMORE_AVAILABLE = False

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
    
    def __init__(self, ph: phonopy.Phonopy, symmetrize_Dq: bool = False, symprec: float = 1e-5):
        self.ph = ph
        self.atoms = views.to_ase(ph.primitive)
        self.symmetrize_Dq = symmetrize_Dq
        self.symprec = symprec
        
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
        if self.symmetrize_Dq:
            from mbe_automation.dynamics.harmonic.symmetry import symmetrized_dynamical_matrix
            return symmetrized_dynamical_matrix(self.ph, q, tolerance=self.symprec)
        else:
            self.ph.dynamical_matrix.run(q)
            return self.ph.dynamical_matrix.dynamical_matrix

def track_from_gamma(
    phonopy_object: phonopy.Phonopy,
    q_points: npt.NDArray[np.float64],
    q_spacing: float = DEFAULT_Q_SPACING,
    degenerate_freqs_tol_cm1: float = DEFAULT_DEGENERATE_FREQS_TOL,
    delta_q: float = 0.05,
    symmetrize_Dq: bool = False,
    symprec: float = 1e-5,
) -> npt.NDArray[np.int64]:
    """
    Compute band indices for a set of q-points using path tracking from Gamma.
    
    Args:
        phonopy_object: Initialized Phonopy object
        q_points: (N_q, 3) list of q-points to assign
        q_spacing: Spacing for path interpolation in Å⁻¹
        degenerate_freqs_tol_cm1: Tolerance for detecting degenerate frequencies in cm⁻¹.
        delta_q: Displacement distance for perturbation theory in Å⁻¹
        
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
    if not _NOMORE_AVAILABLE:
        raise ImportError(
            "The `track_from_gamma` function requires the `nomore_ase` package. "
            "Install it in your environment to use this functionality."
        )

    adapter = PhonopyASEAdapter(phonopy_object, symmetrize_Dq=symmetrize_Dq, symprec=symprec)
    
    flat_indices = assign_bands(
        phonons=adapter,
        q_points=q_points,
        q_spacing=q_spacing,
        use_degenerate_pt=True,
        degenerate_freqs_tol_cm1=degenerate_freqs_tol_cm1,
        delta_q=delta_q,
    )
    
    n_q = len(q_points)
    n_modes = flat_indices.size // n_q
    
    n_atoms = len(adapter.atoms)
    if n_modes != 3 * n_atoms:
        raise ValueError(
            f"Number of modes ({n_modes}) does not match 3 * n_atoms ({3 * n_atoms})."
        )
    
    return flat_indices.reshape(n_q, n_modes)


def reorder(
    band_indices: npt.NDArray[np.int64],
    frequencies: npt.NDArray[np.float64],
    eigenvectors: npt.NDArray[np.complex128] | None = None,
    eigenvectors_storage: Literal["columns", "rows"] = "rows"
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """
    Reorder frequencies (and optionally eigenvectors) so that column j contains data of band j.
    
    Args:
        band_indices: (n_q, n_bands) array of band indices from track_from_gamma.
        frequencies: (n_q, n_bands) array of frequencies.
        eigenvectors: Optional array of eigenvectors.
                      Shape must be consistent with `dynamics.harmonic.modes.at_k_points`.
                      - If storage="rows": (n_q, n_bands, n_dim)
                      - If storage="columns": (n_q, n_dim, n_bands)
                      Where n_dim must be equal to n_bands.
                      n_dim is the number of spatial dimensions, i.e. 3 * n_atoms,
                      where n_atoms is the number of atoms in the unit cell.
        eigenvectors_storage: Convention for storing eigenvectors, "columns" or "rows".
            This convention applies to both the input `eigenvectors` array and the output `reordered_vecs` array.
            - "rows": Eigenvectors are stored in rows (n_kpoints, n_bands, n_dim).
              v[k, i, :] is the eigenvector for the i-th band at k-point k.
            - "columns": Eigenvectors are stored in columns (n_kpoints, n_dim, n_bands).
              v[k, :, i] is the eigenvector for the i-th band at k-point k.
            Default is "rows".
        
    Returns:
        If eigenvectors is None:
            (n_q, n_bands) array `reordered_freqs` where `reordered_freqs[k, b]` is the frequency of band b at q-point k.
        If eigenvectors is provided:
            Tuple of (reordered_freqs, reordered_vecs).
            `reordered_vecs` has the same shape and storage convention as `eigenvectors`.
    """
    if eigenvectors_storage not in ["rows", "columns"]:
        raise ValueError(f"Invalid eigenvectors_storage: {eigenvectors_storage}")

    n_q, n_bands = frequencies.shape
    
    reordered_freqs = np.empty((n_q, n_bands), dtype=np.float64)
    if eigenvectors is not None:
        reordered_vecs = np.empty_like(eigenvectors)

    for k in range(n_q):
        reordered_freqs[k, band_indices[k]] = frequencies[k]
        
        if eigenvectors is not None:
            if eigenvectors_storage == "rows":
                reordered_vecs[k, band_indices[k], :] = eigenvectors[k, :, :]
            else: # columns
                reordered_vecs[k, :, band_indices[k]] = eigenvectors[k, :, :]
    
    if eigenvectors is not None:
        return reordered_freqs, reordered_vecs
    
    return reordered_freqs


def find_degenerate_frequencies(
    freqs: npt.NDArray[np.float64], 
    tolerance: float = 1e-4
) -> List[List[int]]:
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
    gamma_index: int,
    symmetry_tolerance: float | None = None,
    symmetrize_Dq: bool = False,
    symprec: float = 1e-5,
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
        symmetry_tolerance: Tolerance used by nomore_ase to determine if two 
            normal modes are degenerate. It defines the minimum required inner product 
            (overlap) between their eigenvectors after applying a lattice 
            symmetry operation. Defaults to 0.01 if None.
        
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
        phonopy_object=phonopy_object, 
        k_points=[[0, 0, 0]], 
        compute_eigenvecs=True, 
        eigenvectors_storage="rows",
        symmetrize_Dq=symmetrize_Dq,
        symprec=symprec,
    )
    
    eigenvecs_gamma = eigenvecs_all[0]
    n_atoms = len(phonopy_object.primitive)
    n_modes = n_atoms * 3
    
    eigenvectors = eigenvecs_gamma.reshape(n_modes, n_atoms, 3)
    
    tol = symmetry_tolerance if symmetry_tolerance is not None else 0.01
    gamma_mode_groups = sym_calc.find_degeneracy_groups_by_symmetry(
        eigenvectors, 
        q_point=(0,0,0),
        tol=tol
    )
    
    bands_at_gamma = band_indices[gamma_index]    
    flat_band_indices = band_indices.flatten()
    full_groups = []
    
    for group in gamma_mode_groups:
        degenerate_band_ids = bands_at_gamma[group]
        mask = np.isin(flat_band_indices, degenerate_band_ids)
        full_groups.append(np.flatnonzero(mask).tolist())
        
    return full_groups
