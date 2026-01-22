"""
Integration module for NoMoRe (Normal Mode Refinement).

This module provides adapters and functions to bridge mbe_automation's ForceConstants
with the nomore_ase refinement library.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional

import spglib
from ase import units as ase_units
from nomore_ase.core.phonon_data import PhononData

import logging
import mbe_automation.api.classes

logger = logging.getLogger(__name__)

# Constants
EV_TO_CM1 = 8065.54429

class NomoreAdapter:
    """
    Adapter to convert mbe_automation ForceConstants to nomore_ase PhononData.
    """
    def __init__(self, force_constants: 'mbe_automation.api.classes.ForceConstants'):
        self.fc = force_constants
        self.structure = self.fc.primitive

    def get_phonon_data(
        self,
        q_mesh: Tuple[int, int, int],
        symmetrize: bool = True,
        symprec: float = 1e-2,
        compute_bands: bool = True
    ) -> PhononData:
        """
        Generate PhononData object required by nomore_ase.
        
        Args:
            q_mesh: (Nx, Ny, Nz) grid dimensions.
            symmetrize: Whether to symmetrize frequencies/eigenvectors (currently unused logic but kept for interface compatibility).
            symprec: Symmetry precision for finding IBZ.
            compute_bands: Whether to compute band indices (currently unused).
            
        Returns:
            PhononData object with frequencies in cm⁻¹.
        """
        # 1. Get Geometry from Primitive Structure
        atoms_ase = self.structure.to_ase_atoms()
        cell = (
            atoms_ase.get_cell(),
            atoms_ase.get_scaled_positions(),
            atoms_ase.get_atomic_numbers()
        )
        
        # 2. Get Irreducible Brillouin Zone (IBZ) from spglib
        mapping, grid = spglib.get_ir_reciprocal_mesh(q_mesh, cell, is_shift=[0, 0, 0], symprec=symprec)
        unique_indices = np.unique(mapping)
        n_irr = len(unique_indices)
        
        # Weights (multiplicity)
        q_weights = np.array([np.sum(mapping == idx) for idx in unique_indices])
        
        # Fractional q-points
        mesh_array = np.array(q_mesh)
        irr_q_frac = grid[unique_indices] / mesh_array
        
        # 3. Compute Frequencies and Eigenvectors for each IBZ q-point
        # ForceConstants.frequencies_and_eigenvectors(q) returns (freqs_THz, eigenvectors)
        all_freqs_cm1 = []
        all_eigenvectors = [] # will be list of (n_modes, n_atoms, 3)
        mode_weights = []
        
        # Pre-calculate factor for unit conversion if needed.
        # FC returns THz (usually). nomore_ase expects cm⁻¹.
        # 1 THz = 33.35641 cm⁻¹
        THZ_TO_CM1 = 33.35641
        
        for q_idx, q_point in enumerate(irr_q_frac):
            # Evaluate at q-point
            freqs_thz, eigenvectors = self.fc.frequencies_and_eigenvectors(q_point)
            
            # Convert frequencies to cm⁻¹
            freqs_cm1 = freqs_thz * THZ_TO_CM1
            
            # Ensure eigenvectors are (n_modes, n_atoms, 3)
            # The API returns eigenvectors.
            # We need to verify the shape. Assuming standard phonopy-like return: (n_modes, n_atoms, 3)
            
            all_freqs_cm1.append(freqs_cm1)
            all_eigenvectors.append(eigenvectors)
            
            # Weights: each mode at this q-point shares the q-point weight
            n_modes = len(freqs_cm1)
            mode_weights.extend([q_weights[q_idx]] * n_modes)
            
        # Flatten everything
        flat_freqs_cm1 = np.concatenate(all_freqs_cm1)
        flat_eigenvectors = np.concatenate(all_eigenvectors, axis=0) # Shape: (TotalModes, n_atoms, 3)
        flat_weights = np.array(mode_weights)
        
        # 4. Construct Degeneracy Groups
        # This logic is adapted from ase_adapter.py
        degeneracy_groups = self._find_degeneracy_groups(flat_freqs_cm1, tol=1e-4) # 1e-4 cm-1 tolerance
        
        # Average degenerate frequencies
        flat_freqs_cm1 = self._average_degenerate_frequencies(flat_freqs_cm1, degeneracy_groups)
        
        # 5. Build required metadata for PhononData
        n_q = len(irr_q_frac)
        # Actually a safer way for mode_q_indices:
        mode_q_indices = []
        for i, q_modes in enumerate(all_freqs_cm1):
             mode_q_indices.extend([i] * len(q_modes))
        mode_q_indices = np.array(mode_q_indices)

        # Supercell size
        # We can infer it from the ForceConstants supercell matrix diagonal if simpler
        # or just pass it through if known. Using definition from FC structure.
        # ForceConstants usually has supercell matrix.
        supercell_matrix = self.fc.supercell_matrix
        if supercell_matrix is not None:
             supercell = (int(supercell_matrix[0,0]), int(supercell_matrix[1,1]), int(supercell_matrix[2,2]))
        else:
             supercell = (1, 1, 1)

        return PhononData(
            frequencies_cm1=flat_freqs_cm1,
            eigenvectors=flat_eigenvectors,
            q_points=irr_q_frac,
            mode_q_indices=mode_q_indices,
            weights=flat_weights,
            degeneracy_groups=degeneracy_groups,
            positions_frac=atoms_ase.get_scaled_positions(),
            cell=atoms_ase.get_cell()[:],
            symbols=list(atoms_ase.get_chemical_symbols()),
            masses=atoms_ase.get_masses(),
            supercell=supercell,
            n_atoms=len(atoms_ase),
            band_indices=None  # Can implement band assignment later if needed
        )

    def _find_degeneracy_groups(self, freqs: np.ndarray, tol: float = 1e-4) -> list[list[int]]:
        """Find index groups of degenerate modes."""
        n_modes = len(freqs)
        visited = np.zeros(n_modes, dtype=bool)
        groups = []
        
        argsort = np.argsort(freqs)
        sorted_freqs = freqs[argsort]
        
        # Efficient sorted grouping
        current_group = [argsort[0]]
        visited[argsort[0]] = True
        
        for i in range(1, n_modes):
            idx = argsort[i]
            prev_idx = argsort[i-1]
            
            if np.abs(freqs[idx] - freqs[prev_idx]) < tol:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
            visited[idx] = True
        groups.append(current_group)
        
        return groups

    def _average_degenerate_frequencies(self, freqs: np.ndarray, groups: list[list[int]]) -> np.ndarray:
        """Enforce exact degeneracy by averaging."""
        new_freqs = freqs.copy()
        for group in groups:
            if len(group) > 1:
                avg = np.mean(freqs[group])
                new_freqs[group] = avg
        return new_freqs


def to_phonon_data(
    force_constants: 'mbe_automation.api.classes.ForceConstants',
    q_mesh: Tuple[int, int, int]
) -> PhononData:
    """
    Convert ForceConstants to nomore_ase PhononData using a q-point mesh.
    
    Args:
        force_constants: The force constants model.
        q_mesh: Mesh dimensions (Nx, Ny, Nz).
        
    Returns:
        PhononData object ready for refinement.
    """
    adapter = NomoreAdapter(force_constants)
    return adapter.get_phonon_data(q_mesh=q_mesh)


def run_nomore_refinement(
    force_constants: 'mbe_automation.api.classes.ForceConstants',
    cif_path: str,
    output_dir: str,
    q_mesh: Tuple[int, int, int] = (1, 1, 1),
    supercell: Optional[Tuple[int, int, int]] = None,
    restraint_weight: float = 0.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Run NoMoRe refinement on the force constants against an experimental CIF.
    
    Args:
        force_constants: The input force constants model.
        cif_path: Path to experimental CIF with ADPs.
        output_dir: Directory to save results.
        q_mesh: Q-point mesh for sampling.
        supercell: Supercell dimensions (optional, inferred from FC if None).
        restraint_weight: Weight for restraining to initial frequencies.
        **kwargs: Additional arguments passed to NoMoReRefinement.run()
        
    Returns:
        Dictionary containing refinement results (refined frequencies, ADPs, etc).
    """
    from nomore_ase.workflows.refinement import NoMoReRefinement
    
    # 1. Convert to PhononData
    phonon_data = to_phonon_data(force_constants, q_mesh)
    
    # Ensure supercell is consistent if provided, otherwise use what's in PhononData (from FC)
    if supercell is None:
        supercell = phonon_data.supercell
    
    # 2. Setup Refinement Workflow
    # We pass the CIF path. NoMoReRefinement loads the CIF for structure/ADPs.
    # We allow it to use its default/mock calculator because we provide phonon_data directly.
    workflow = NoMoReRefinement(cif_path=cif_path)
    
    # 3. Run Refinement
    result = workflow.run(
        supercell=supercell,
        q_mesh=q_mesh,
        restraint_weight=restraint_weight,
        phonon_data=phonon_data,
        output_dir=output_dir,
        **kwargs
    )
    
    return result
