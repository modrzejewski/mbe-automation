from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, List, Optional
import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.geometry import find_mic
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pathlib import Path

import mbe_automation.structure.relax
from mbe_automation.configs.structure import Minimum, SYMMETRY_TOLERANCE_STRICT

@dataclass
class MFDThermalDisplacements:
    """
    Thermal displacements derived from Molecular Dynamics trajectories.
    """
    mean_square_displacements_matrix_diagonal: npt.NDArray[np.float64] # Symmetrized ADPs (N, 3, 3)
    raw_covariance_matrix: npt.NDArray[np.float64] # Raw MD Covariance (N, 3, 3)
    mean_square_displacements_matrix_diagonal_cif: Optional[npt.NDArray[np.float64]] = None 
    temperature: Optional[float] = None
    structure: Optional[Structure] = None

    @property
    def as_tensor_list(self) -> List[list]:
        """
        Returns ADPs in the format:
        [
            [label, [U11, U22, U33], [U12, U13, U23]],
            ...
        ]
        Only for symmetry-unique atoms.
        """
        if self.structure is None:
            raise ValueError("Structure is required to generate tensor list with labels.")
            
        sga = SpacegroupAnalyzer(self.structure, symprec=SYMMETRY_TOLERANCE_STRICT)
        symmetrized_structure = sga.get_symmetrized_structure()
        
        results = []
        # Iterate over groups of equivalent atoms
        for group in symmetrized_structure.equivalent_indices:
            # unique atom is the first one in the group
            idx = group[0]
            site = self.structure[idx]
            
            # Label format: "Element Index" (e.g. "H 0")
            label = f"{site.specie.symbol} {idx}"
            
            # Symmetrized ADP for this atom (already symmetrized in the diagonal matrix)
            u = self.mean_square_displacements_matrix_diagonal[idx]
            
            # Diagonal elements
            diag = [u[0, 0], u[1, 1], u[2, 2]]
            
            # Off-diagonal elements (symmetric matrix, so we store xy, xz, yz)
            off_diag = [u[0, 1], u[0, 2], u[1, 2]]
            
            results.append([label, diag, off_diag])
            
        return results


def symmetrize_adps(
        structure: Structure,
        adps: npt.NDArray[np.float64],
        symprec: float = SYMMETRY_TOLERANCE_STRICT
) -> npt.NDArray[np.float64]:
    """
    Averages ADP tensors (U_cart) according to crystal symmetry.
    Matches logic in harmonic/modes.py.
    
    Args:
        structure: A pymatgen.core.Structure object
        adps: Numpy array (N_atoms, 3, 3) containing raw ADPs
        symprec: Symmetry precision
        
    Returns:
        symmetrized_adps: Matrix (N_atoms, 3, 3) with correctly averaged tensors.
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    symmetrized_structure = sga.get_symmetrized_structure()
    
    adps_final = np.zeros_like(adps)
    
    # Iterate over groups of equivalent atoms (Wyckoff positions)
    for group_indices in symmetrized_structure.equivalent_indices:
        
        # 1. Select a representative (first atom in the group)
        ref_index = group_indices[0]
        ref_site = structure[ref_index]
        
        # Container for tensors transformed to the representative's frame
        rotated_tensors = []
        
        # 2. Transform all tensors to the representative's reference frame
        for idx in group_indices:
            target_site = structure[idx]
            original_tensor = adps[idx]
            
            # Find the symmetry operation R that maps ref_site -> target_site
            op = None
            for symm_op in sga.get_symmetry_operations():
                transformed_coords = symm_op.operate(ref_site.frac_coords)
                distance, _ = structure.lattice.get_distance_and_image(
                        transformed_coords,
                        target_site.frac_coords
                )
                if np.allclose(distance, 0, atol=symprec):
                    op = symm_op
                    break
            
            if op is None:
                # Fallback or stricter check needed? 
                # Ideally every equivalent atom IS mapped by a symmetry op.
                raise ValueError(f"No symmetry operation found between atom {ref_index} and {idx}")
            
            # R is the rotation matrix (Cartesian part of the operation)
            R = op.rotation_matrix
            
            # Rotate the tensor to ref frame: U_ref = R.T @ U_target @ R
            # (R.T is the inverse for orthogonal matrices)
            U_rotated_to_ref = R.T @ original_tensor @ R
            rotated_tensors.append(U_rotated_to_ref)
            
        # 3. Compute the average in the representative's frame
        U_avg_ref = np.mean(rotated_tensors, axis=0)
        
        # 4. Propagate the average back to all atoms in the group
        for idx in group_indices:
            # Re-find operation (optimisation: could cache)
            op = None
            target_site = structure[idx]
            for symm_op in sga.get_symmetry_operations():
                transformed_coords = symm_op.operate(ref_site.frac_coords)
                distance, _ = structure.lattice.get_distance_and_image(
                        transformed_coords,
                        target_site.frac_coords
                )
                if np.allclose(distance, 0, atol=symprec):
                    op = symm_op
                    break

            if op is None:
                 raise ValueError(f"No symmetry operation found between atom {ref_index} and {idx}")

            R = op.rotation_matrix

            # Rotate average to target: U_target = R @ U_avg_ref @ R.T
            adps_final[idx] = R @ U_avg_ref @ R.T
            
    return adps_final


def get_relaxed_reference(
        atoms: Atoms, 
        calculator: Calculator, 
        work_dir: Path | str = Path("./relaxation_ref")
) -> Atoms:
    """
    Relax a single frame to use as the reference structure.
    Uses constant volume relaxation by default.
    """
    
    # Use default configuration
    relax_config = Minimum()
    relax_config.cell_relaxation = "constant_volume"
    
    relaxed_atoms, _ = mbe_automation.structure.relax.crystal(
        unit_cell=atoms,
        calculator=calculator,
        config=relax_config,
        work_dir=work_dir,
        key="relax_ref"
    )
    
    return relaxed_atoms


def calculate_adp_covariance_matrix(
        traj_objects: List[Atoms],
        calculator: Optional[Calculator] = None,
        burn_in: int = 500,
        temperature: Optional[float] = None,
        work_dir: Path | str = Path("./")
) -> MFDThermalDisplacements:
    """
    Directly derives anisotropic ADPs (U^C) from MD trajectories.
    
    Args:
        traj_objects: List of ASE Atoms objects representing the trajectory.
        calculator: Calculator for relaxing the reference frame.
        burn_in: Number of initial frames to discard.
        temperature: Temperature of the MD simulation (meta-data only).
        work_dir: Directory for relaxation outputs.

    Returns:
        MFDThermalDisplacements object containing raw and symmetrized ADPs.
    """
    # 1. Discard non-equilibrated frames
    if burn_in >= len(traj_objects):
         raise ValueError(f"Burn-in ({burn_in}) is larger than or equal to trajectory length ({len(traj_objects)}).")

    work_traj = traj_objects[burn_in:]
    n_frames = len(work_traj)
    
    # Get atoms info from the first frame in the working trajectory
    ref_atoms_frame0 = work_traj[0]
    n_atoms = len(ref_atoms_frame0)
    cell = ref_atoms_frame0.get_cell()
    pbc = ref_atoms_frame0.get_pbc()

    # 2. Get all positions and find the mean position for each individual atom
    # Shape: (n_frames, n_atoms, 3)
    all_pos = np.array([atoms.get_positions() for atoms in work_traj])
    mean_positions = np.mean(all_pos, axis=0)
    
    # Initialize accumulator for the covariance matrices
    adp_tensor_sum = np.zeros((n_atoms, 3, 3))

    for atoms in work_traj:
        current_pos = atoms.get_positions()
        
        # 3. Calculate displacement vector from MEAN position
        # find_mic handles periodic boundary conditions safely
        raw_diff = current_pos - mean_positions
        diff, _ = find_mic(raw_diff, cell, pbc)
        
        # 4. Remove Global Translation per frame (Center of Mass drift correction)
        # Note: Ideally this should be mass-weighted, but simple mean is common if masses are similar
        # or if just removing geometric drift. 
        # Using simple mean to match previous logic.
        frame_drift = np.mean(diff, axis=0)
        diff -= frame_drift 
        
        # 5. Compute outer product for the covariance matrix
        # Einsum 'ki,kj->kij' computes the outer product for each atom k
        outer = np.einsum('ki,kj->kij', diff, diff)
        adp_tensor_sum += outer
        
    # Average over frames to get the final tensors (Variance/Covariance)
    raw_cov_matrix = adp_tensor_sum / n_frames
    
    # 6. Symmetrize
    # We need a reference structure to determine symmetry. 
    # If calculator is provided, we relax the first frame (of work_traj) to get a clean high-symmetry structure.
    # If not, we fall back to the mean positions (which might be 'smeared' but usually have the average symmetry)
    # or just the first frame.
    
    if calculator is not None:
        try:
             # We relax the first frame of the *working* trajectory (post burn-in)
             # to avoid initial noise, or maybe pre burn-in frame 0? 
             # Usually frame 0 of the production run is a good start guess.
             # Let's use work_traj[0]
             print("Relaxing reference structure for symmetry analysis...")
             reference_ase = get_relaxed_reference(ref_atoms_frame0, calculator, work_dir=work_dir)
        except Exception as e:
             print(f"Relaxation failed: {e}. Falling back to unrelaxed first frame.")
             reference_ase = ref_atoms_frame0
    else:
        # Fallback to mean positions? 
        # Constructing atoms from mean positions might be safer for average symmetry key
        reference_ase = ref_atoms_frame0.copy()
        reference_ase.set_positions(mean_positions)
        
    pmg_structure = AseAtomsAdaptor.get_structure(reference_ase)
    
    symmetrized_adps_matrix = symmetrize_adps(pmg_structure, raw_cov_matrix)
    
    return MFDThermalDisplacements(
        mean_square_displacements_matrix_diagonal=symmetrized_adps_matrix,
        raw_covariance_matrix=raw_cov_matrix,
        temperature=temperature,
        structure=pmg_structure
    )
