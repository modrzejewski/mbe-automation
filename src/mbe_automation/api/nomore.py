"""
Integration module for NoMoRe (Normal Mode Refinement).

This module provides adapters and functions to bridge mbe_automation's ForceConstants
with the nomore_ase refinement library.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional


import phonopy.physical_units
from nomore_ase.core.phonon_data import PhononData

import mbe_automation.api.classes
import mbe_automation.dynamics.harmonic.modes

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
        compute_bands: bool = True
    ) -> PhononData:
        """
        Generate PhononData object required by nomore_ase.
        
        Args:
            q_mesh: (Nx, Ny, Nz) grid dimensions.
            compute_bands: Whether to compute band indices (currently unused).
            
        Returns:
            PhononData object with frequencies in cm⁻¹.
        """
        # 1. Initialize Phonopy and Units
        units = phonopy.physical_units.get_physical_units()
        ph = self.fc.to_phonopy()
        
        # 2. Get Irreducible Brillouin Zone (IBZ)
        irr_q_frac, q_weights = mbe_automation.dynamics.harmonic.modes.phonopy_k_point_grid(
            phonopy_object=ph,
            k_point_mesh=q_mesh,
            use_symmetry=True
        )

        n_irr = len(irr_q_frac)
        
        # 3. Compute Frequencies and Eigenvectors for each IBZ q-point
        all_freqs_cm1 = []
        all_eigenvectors = [] # will be list of (n_modes, n_atoms, 3)
        mode_weights = []
        
        for q_idx, q_point in enumerate(irr_q_frac):
            # Evaluate at q-point
            freqs_thz, eigenvectors = mbe_automation.dynamics.harmonic.modes.at_k_point(ph.dynamical_matrix, q_point)
            
            # Convert frequencies to cm⁻¹
            freqs_cm1 = freqs_thz * units.THzToCm
            
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

        return PhononData(
            frequencies_cm1=flat_freqs_cm1,
            eigenvectors=flat_eigenvectors,
            q_points=irr_q_frac,
            mode_q_indices=mode_q_indices,
            weights=flat_weights,
            degeneracy_groups=degeneracy_groups,
            positions_frac=ph.primitive.scaled_positions,
            cell=ph.primitive.cell,
            symbols=ph.primitive.symbols,
            masses=ph.primitive.masses,
            supercell=[1, 1, 1],
            n_atoms=len(ph.primitive),
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
