from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Literal

import mbe_automation.storage.core
import mbe_automation.storage
from nomore_ase.core.calculator import NoMoReCalculator
from nomore_ase.optimization.engine import RefinementEngine
from nomore_ase.crystallography.cctbx_adapter import CctbxAdapter
from nomore_ase.analysis.atom_matching import get_atom_mapping_by_position
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core import Structure as PymatgenStructure, Lattice
from mbe_automation.storage.views import to_pymatgen
from mbe_automation.structure.crystal import match as match_structures
from scipy.spatial.transform import Rotation
from cctbx import adptbx

DEFAULT_RESTRAINT_WEIGHT = 0.1

import phonopy.physical_units
import mbe_automation.dynamics.harmonic.modes

def _find_degeneracy_groups(freqs: np.ndarray, tol: float = 1e-4) -> list[list[int]]:
    """Find index groups of degenerate modes."""
    n_modes = len(freqs)
    visited = np.zeros(n_modes, dtype=bool)
    groups = []
    
    argsort = np.argsort(freqs)
    # sorted_freqs = freqs[argsort]
    
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

def _average_degenerate_frequencies(freqs: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Enforce exact degeneracy by averaging."""
    new_freqs = freqs.copy()
    for group in groups:
        if len(group) > 1:
            avg = np.mean(freqs[group])
            new_freqs[group] = avg
    return new_freqs

def _get_mapping_and_transform(
    struct_target: PymatgenStructure,
    struct_source: PymatgenStructure
) -> tuple[list[int], npt.NDArray]:
    """
    Get indices and rotation matrix to align Source to Target.
    
    Args:
        struct_target: The reference structure (defining order).
        struct_source: The structure to reorder (providing data).
        
    Returns:
        tuple: (mapping_indices, rotation_matrix)
            - mapping_indices[i] is index in Source for atom i in Target.
            - rotation_matrix is (3,3) array R such that x_target = R @ x_source.
    """
    # Enable primitive_cell and attempt_supercell to handle cases where 
    # CIF is a supercell or conventional cell of the primitive reference.
    # Note: get_transformation fails if primitive_cell=True, so we set it to False
    # and rely on attempt_supercell (and explicit P1 expansion) to handle size diffs.
    matcher = StructureMatcher(
        primitive_cell=False, 
        scale=False, 
        attempt_supercell=True, 
        comparator=ElementComparator()
    )
    
    # 1. Get transformations and mapping
    # Returns: supercell matrix, fractional translation vector, mapping
    result = matcher.get_transformation(struct_target, struct_source)
    
    if result is None:
        raise ValueError("Could not match CIF structure to Reference structure.")
        
    _, translation_frac, match_result = result
        
    if any(idx is None for idx in match_result):
        n_matched = sum(1 for x in match_result if x is not None)
        raise ValueError(f"Incomplete atom mapping. Only {n_matched} atoms matched.")

    # Calculate Rotation using Lattice Matrices
    # struct_source_aligned has the lattice orientation of struct_target (roughly)
    struct_source_aligned = matcher.get_s2_like_s1(struct_target, struct_source)
    
    # A = Source Lattice Matrix (rows are vectors)
    # A_prime = Aligned Lattice Matrix (rows are vectors)
    # We want R acting on column vectors: v' = R v
    # This implies A_prime.T = R @ A.T
    # So R = A_prime.T @ inv(A.T) = (inv(A) @ A_prime).T
    
    A = struct_source.lattice.matrix
    A_prime = struct_source_aligned.lattice.matrix
    
    R = (np.linalg.inv(A) @ A_prime).T
    
    # Verification (Determinant should be +1 for pure rotation)
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=0.01):
        print(f"Warning: Transformation determinant is {det:.4f} (expected 1.0 for rotation).")

    # RMSD Check (StructureMatcher)
    rms_info = matcher.get_rms_dist(struct_target, struct_source)
    if rms_info is not None:
        rms_dist, max_dist = rms_info
        print(f"StructureMatcher RMSD: {rms_dist:.4f} Å")
        
    # Manual Sanity Check
    _check_transformation_quality(struct_target, struct_source, match_result, R, translation_frac)
    
    return match_result, R

def _check_transformation_quality(
    target: PymatgenStructure,
    source: PymatgenStructure,
    mapping: list[int],
    rot_matrix: npt.NDArray,
    translation_frac: npt.NDArray
) -> None:
    """
    Sanity check: Calculate RMSD and Max Dist using the computed rotation and mapping.
    Ensures the rotation matrix R aligns the atoms correctly.
    """
    # 1. Rotate Source positions using Cartesian Rotation R: x' = R x
    # struct.cart_coords is (N, 3). So we need (R @ x.T).T = x @ R.T
    pos_source_rotated = source.cart_coords @ rot_matrix.T
    
    # 2. Apply Translation
    # Translation is fractional in Target lattice
    lat = target.lattice.matrix
    translation_cart = translation_frac @ lat
    
    pos_source_transformed = pos_source_rotated + translation_cart
    
    # 3. Reorder Source to match Target
    # mapping is list of source indices for each target index
    pos_source_ordered = pos_source_transformed[mapping]
    pos_target = target.cart_coords
    
    # 4. Compute differences
    diff_cart = pos_target - pos_source_ordered
    
    # 5. Handle Periodic Boundary Conditions
    # Convert cartesian diff to fractional diff in Target lattice
    # diff_frac = diff_cart @ inv_matrix
    inv_lat = target.lattice.inv_matrix
    
    diff_frac = diff_cart @ inv_lat
    # Wrap to [-0.5, 0.5]
    diff_frac -= np.round(diff_frac)
    # Convert back to Cartesian
    diff_cart_wrapped = diff_frac @ lat
    
    # 6. Calculate Stats
    dists_sq = np.sum(diff_cart_wrapped**2, axis=1)
    dists = np.sqrt(dists_sq)
    
    rmsd = np.sqrt(np.mean(dists_sq))
    max_dist = np.max(dists)
    
    print(f"Manual Sanity Check RMSD: {rmsd:.4f} Å (Max Dist: {max_dist:.4f} Å)")

def _extract_u_cart_exp(
    cif_path: str, 
    reference_structure: mbe_automation.storage.core.Structure | None = None,
    matching_algo: Literal["nomore_ase", "robust"] = "nomore_ase",
) -> npt.NDArray:
    """
    Extract experimental Cartesian ADPs from a CIF file.
    
    Uses nomore_ase's CctbxAdapter to parse the CIF.
    
    If 'reference_structure' is provided:
    - "robust": Matches CIF atoms to Reference atoms (handling PBC/Rotation/Translation)
      and rotates ADPs to match Reference frame.
    - "nomore_ase": Matches atoms by position (assuming already aligned) and reorders ADPs.
      Does NOT rotate ADPs.
    
    Args:
        cif_path: Path to the CIF file.
        reference_structure: Optional structure to match atom ordering against.
        matching_algo: Algorithm to use for atom matching ("nomore_ase" or "robust").
        
    Returns:
        npt.NDArray: Array of Cartesian ADPs with shape (N_atoms, 3, 3) in Å².
    """
    adapter = CctbxAdapter(cif_path)
    
    # Expand to P1 to get full unit cell content (not just asymmetric unit)
    # This solves atom count mismatch issues.
    xs_p1 = adapter.xray_structure.expand_to_p1(sites_mod_positive=True)
    
    # Extract data from P1 structure
    cif_pos_cart = xs_p1.sites_cart().as_numpy_array()
    cif_symbols = [s.element_symbol() for s in xs_p1.scatterers()]
    cif_cell_par = xs_p1.unit_cell().parameters()
    
    # Extract ADPs from P1 structure
    unit_cell = xs_p1.unit_cell()
    u_cart_list = []
    
    for sc in xs_p1.scatterers():
        if sc.u_iso != -1.0 and sc.u_star == (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0):
            # Isotropic atom: Convert U_iso to U_cart (diagonal)
            u_cart_val = adptbx.u_iso_as_u_cart(sc.u_iso)
        else:
            # Anisotropic atom: Convert U_star to U_cart
            u_cart_val = adptbx.u_star_as_u_cart(unit_cell, sc.u_star)
        
        # cctbx u_cart tuple order: u11, u22, u33, u12, u13, u23
        # Convert to 3x3 symmetric matrix
        u_tensor = np.array([
            [u_cart_val[0], u_cart_val[3], u_cart_val[4]],
            [u_cart_val[3], u_cart_val[1], u_cart_val[5]],
            [u_cart_val[4], u_cart_val[5], u_cart_val[2]]
        ])
        u_cart_list.append(u_tensor)
        
    u_cart = np.array(u_cart_list)
    
    if reference_structure is not None:
        # Pre-check RMSD using independent match function (for debugging/info)
        try:
            # cif_pos_cart already extracted from P1
            cif_cell = np.array(cif_cell_par)
            # Convert CIF params to matrix for match function?
            # match function expects cell_vectors as (3,3) matrix.
            # CIF usually gives params (a,b,c,alpha,beta,gamma).
            # Need to convert params to matrix.
            # Pymatgen Lattice does this.
            cif_lat = Lattice.from_parameters(*cif_cell_par)
            cif_matrix = cif_lat.matrix
            
            # Map symbols to atomic numbers
            from ase.data import atomic_numbers
            # cif_symbols already extracted
            cif_z = np.array([atomic_numbers[s] for s in cif_symbols])
            
            # Reference data
            # Assuming reference_structure has these attributes as per storage.core.Structure
            ref_z = reference_structure.atomic_numbers
            ref_matrix = reference_structure.cell_vectors
            
            # Reference positions: use ASE adapter for safety (consistent with later usage)
            ref_positions = reference_structure.to_ase_atoms().get_positions()
            
            rmsd_check = match_structures(
                positions_a=ref_positions,
                atomic_numbers_a=ref_z,
                cell_vectors_a=ref_matrix,
                positions_b=cif_pos_cart,
                atomic_numbers_b=cif_z,
                cell_vectors_b=cif_matrix
            )
            print(f"Initial RMSD Check (mbe_automation.structure.crystal.match): {rmsd_check}")
        except Exception as e:
            print(f"Initial RMSD Check failed: {e}")

        if matching_algo == "robust":
            # 1. Convert CIF to Pymatgen Structure (already have data)
            # cif_pos_cart, cif_symbols, cif_cell_par
            
            struct_cif = PymatgenStructure(
                lattice=Lattice.from_parameters(*cif_cell_par),
                species=cif_symbols,
                coords=cif_pos_cart,
                coords_are_cartesian=True
            )

            # 2. Convert Reference to Pymatgen Structure
            struct_ref = to_pymatgen(structure=reference_structure)
            
            # Check for atom count mismatch
            if len(struct_ref) != len(struct_cif):
                print(f"Warning: Atom counts differ (Ref: {len(struct_ref)}, CIF: {len(struct_cif)}). Attempting simple robust match...")
            
            # 3. Get mapping and rotation
            cif_indices, rot_matrix = _get_mapping_and_transform(struct_target=struct_ref, struct_source=struct_cif)
                
            # 4. Reorder ADPs
            u_cart_reordered = u_cart[cif_indices]
            
            # 5. Rotate ADPs: U' = R U R^T
            # Einstein summation: R_ia U_ab R_jb -> U_ij
            u_cart = np.einsum('ia,kab,jb->kij', rot_matrix, u_cart_reordered, rot_matrix)
            
        elif matching_algo == "nomore_ase":
            # Use simple position-based matching (legacy behavior)
            # cif_pos_cart already extracted from P1
            # ref_pos_cart = reference_structure.get_positions() # Error
            ref_pos_cart = reference_structure.to_ase_atoms().get_positions()
            
            if len(cif_pos_cart) != len(ref_pos_cart):
                 raise ValueError(
                    f"Atom count mismatch (CIF: {len(cif_pos_cart)}, Ref: {len(ref_pos_cart)}). "
                    "Cannot use 'nomore_ase' matching. Try matching_algo='robust'."
                )
            
            # Mapping: indices in CIF corresponding to Reference atoms
            cif_indices = get_atom_mapping_by_position(
                target_positions=ref_pos_cart,
                source_positions=cif_pos_cart
            )
            
            # Reorder ADPs only
            u_cart = u_cart[cif_indices]
            
        else:
            raise ValueError(f"Unknown matching_algo: {matching_algo}")
        
    return u_cart

def fit_to_adps(
    fc: mbe_automation.storage.core.ForceConstants,
    cif_path: str,
    temperature: float,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
    restraint_weight: float = DEFAULT_RESTRAINT_WEIGHT,
    bounds: tuple[float, float] = (10.0, 1e4), # Default bounds in cm-1
    matching_algo: Literal["nomore_ase", "robust"] = "nomore_ase",
) -> npt.NDArray:
    """
    Refine phonon frequencies by fitting calculated ADPs to experimental ADPs.

    Args:
        fc: The force constants model. # storage class
        cif_path: Path to the CIF file containing experimental ADPs.
        temperature: Temperature in Kelvin.
        mesh_size: k-point mesh for sampling the Brillouin zone.
        restraint_weight: Weight for restraining refined frequencies to initial values.
        bounds: (min, max) frequency bounds in cm⁻¹.
        matching_algo: Algorithm to use for atom matching ("nomore_ase" or "robust").

    Returns:
        npt.NDArray: Refined frequencies in cm⁻¹.
    """
    # 0. Extract Experimental ADPs
    u_cart_exp = _extract_u_cart_exp(
        cif_path, 
        reference_structure=fc.primitive,
        matching_algo=matching_algo
    )

    # 1. Initialize Phonopy and Units
    ph = mbe_automation.storage.to_phonopy(fc)
    
    # 2. Get Irreducible Brillouin Zone (IBZ)
    irr_q_frac, q_weights = mbe_automation.dynamics.harmonic.modes.phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=mesh_size,
        use_symmetry=True
    )
    
    # 3. Compute Frequencies and Eigenvectors for each IBZ q-point using vectorized at_k_points
    freqs_cm1_grid, eigenvectors_grid = mbe_automation.dynamics.harmonic.modes.at_k_points(
        dynamical_matrix=ph.dynamical_matrix,
        k_points=irr_q_frac,
        compute_eigenvecs=True,
        freq_units="invcm",
        eigenvectors_storage="rows"
    )
    
    # Flatten everything
    flat_freqs_cm1 = freqs_cm1_grid.flatten()
    
    n_modes = eigenvectors_grid.shape[-1]
    n_atoms = len(ph.primitive)
    
    # Reshape to (TotalModes, n_atoms, 3)
    flat_eigenvectors = eigenvectors_grid.reshape(-1, n_atoms, 3)
    
    # Weights: (n_q, ) -> repeat n_modes times for each q
    flat_weights = np.repeat(q_weights, n_modes)
    
    # 4. Construct Degeneracy Groups
    degeneracy_groups = _find_degeneracy_groups(flat_freqs_cm1, tol=1e-4) # 1e-4 cm-1 tolerance
    
    # Average degenerate frequencies
    flat_freqs_cm1 = _average_degenerate_frequencies(flat_freqs_cm1, degeneracy_groups)
    
    # 5. Initialize NoMoRe Calculator
    total_weight = np.sum(q_weights)
    
    calculator = NoMoReCalculator(
        eigenvectors=flat_eigenvectors,
        masses=ph.primitive.masses,
        temperature=temperature,
        normalization_factor=total_weight, 
        weights=flat_weights,
        degeneracy_groups=degeneracy_groups
    )

    # 6. Initialize Engine
    engine = RefinementEngine(calculator)

    # 7. Run ADP fitting
    result = engine.fit_to_adps(
        initial_frequencies=flat_freqs_cm1,
        u_exp=u_cart_exp,
        bounds=bounds,
        use_degeneracy_groups=True,
        restraint_weight=restraint_weight
    )

    if not result['success']:
        print(f"Warning: ADP fitting result reported failure: {result.get('message', 'Unknown error')}")

    return result['frequencies']

