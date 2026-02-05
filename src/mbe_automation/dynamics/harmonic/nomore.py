"""
Integration module for normal mode refinement using
the nomore_ase library (https://github.com/ase/nomore_ase) by Paul Niklas Ruth.

This module provides adapters and functions to bridge mbe_automation's ForceConstants
with the nomore_ase refinement library.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional, Literal, TYPE_CHECKING
import numpy.typing as npt


import phonopy.physical_units
try:
    from nomore_ase.core.phonon_data import PhononData
except ImportError:
    raise ImportError(
        "The `nomore` module requires the `nomore_ase` package. "
        "Install it in your environment to use this functionality."
    )

if TYPE_CHECKING:
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
        mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float,
        compute_bands: bool = True
    ) -> PhononData:
        """
        Generate PhononData object required by nomore_ase.
        
        Args:
            mesh_size: The k-points for sampling the Brillouin zone. Can be:
                - "gamma": Use only the [0, 0, 0] k-point.
                - A floating point number: Defines a supercell of radius R.
                - array of 3 integers: Defines an explicit Monkhorst-Pack mesh.
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
            mesh_size=mesh_size,
            use_symmetry=True
        )
        
        # 3. Compute Frequencies and Eigenvectors for each IBZ q-point using vectorized at_k_points
        # Note: at_k_points returns frequencies in the requested units (cm⁻¹ here)
        # and eigenvectors as (n_q, n_modes, n_modes)
        # We request "rows" storage so that v[q, i, :] is the eigenvector for mode i.
        freqs_cm1_grid, eigenvectors_grid = mbe_automation.dynamics.harmonic.modes.at_k_points(
            dynamical_matrix=ph.dynamical_matrix,
            k_points=irr_q_frac,
            compute_eigenvecs=True,
            freq_units="invcm",
            eigenvectors_storage="rows"
        )
        
        # Flatten everything to match PhononData requirements
        # freqs_cm1_grid is (n_q, n_modes) -> flatten to 1D
        flat_freqs_cm1 = freqs_cm1_grid.flatten()
        
        # eigenvectors: (n_q, n_modes, n_modes) -> need (TotalModes, n_atoms, 3) 
        # TotalModes = n_q * n_modes
        # n_modes = n_atoms * 3
        # So we reshape to (-1, n_atoms, 3) because PhononData expects (n_modes_total, n_atoms, 3)
        # Verify shape first:
        n_modes = eigenvectors_grid.shape[-1]
        n_atoms = len(ph.primitive)
        # Reshape to (n_q * n_modes, n_atoms, 3)
        flat_eigenvectors = eigenvectors_grid.reshape(-1, n_atoms, 3)
        
        # Weights: (n_q, ) -> repeat n_modes times for each q
        flat_weights = np.repeat(q_weights, n_modes)
        
        # 4. Construct Degeneracy Groups
        # This logic is adapted from ase_adapter.py
        degeneracy_groups = self._find_degeneracy_groups(flat_freqs_cm1, tol=1e-4) # 1e-4 cm-1 tolerance
        
        # Average degenerate frequencies
        flat_freqs_cm1 = self._average_degenerate_frequencies(flat_freqs_cm1, degeneracy_groups)
        
        # 5. Frequency -> k-point mapping
        # Create mapping of which q-point index each flat mode belongs to
        mode_q_indices = np.repeat(np.arange(len(irr_q_frac)), n_modes)

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
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float
) -> PhononData:
    """
    Convert ForceConstants to nomore_ase PhononData using a q-point mesh.
    
    Args:
        force_constants: The force constants model.
        mesh_size: The k-points for sampling the Brillouin zone.
        
    Returns:
        PhononData object ready for refinement.
    """
    adapter = NomoreAdapter(force_constants)
    return adapter.get_phonon_data(mesh_size=mesh_size)


def run(
    force_constants: 'mbe_automation.api.classes.ForceConstants',
    cif_path: str,
    output_dir: str,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
    restraint_weight: float | None = None,
    strategy = None,
    fix_positions: bool = True,
    weighting_scheme: Literal["sigma", "unit"] = "sigma",
) -> Dict[str, Any]:
    """
    Run NoMoRe refinement on the force constants against an experimental CIF.
    
    Args:
        force_constants: The input force constants model.
        cif_path: Path to experimental CIF with ADPs.
        output_dir: Directory to save results.
        mesh_size: The k-points for sampling the Brillouin zone.
        restraint_weight: Weight for restraining to initial frequencies.
        **kwargs: Additional arguments passed to NoMoReRefinement.run()
        
    Returns:
        Dictionary containing refinement results (refined frequencies, ADPs, etc).
    """
    from nomore_ase.workflows.refinement import NoMoReRefinement
    from nomore_ase.core.frequency_partition import SensitivityBasedStrategy

    if strategy is None:
        strategy = SensitivityBasedStrategy(low_threshold=0.60, high_threshold=0.90)

    if restraint_weight is None:
        restraint_weight = 0.0
    
    # 1. Convert to PhononData
    phonon_data = to_phonon_data(force_constants, mesh_size)
    
    # 2. Setup Refinement Workflow
    # We pass the CIF path. NoMoReRefinement loads the CIF for structure/ADPs.
    # We allow it to use its default/mock calculator because we provide phonon_data directly.
    workflow = NoMoReRefinement(cif_path=cif_path)
    
    # 3. Run Refinement
    # nomore_ase expects q_mesh to be a tuple/array of integers for np.prod
    if isinstance(mesh_size, str) and mesh_size == "gamma":
        q_mesh = (1, 1, 1)
    else:
        q_mesh = mesh_size
        
    result = workflow.run_joint(
        supercell=phonon_data.supercell,
        q_mesh=q_mesh,
        restraint_weight=restraint_weight,
        phonon_data=phonon_data,
        output_dir=output_dir,
        strategy=strategy,
        fix_positions=fix_positions,
        weighting_scheme=weighting_scheme,
    )
    
    return result
