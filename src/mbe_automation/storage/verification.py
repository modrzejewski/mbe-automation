from __future__ import annotations
import gemmi
import numpy as np
import numpy.typing as npt
import pymatgen.core
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import mbe_automation.dynamics.harmonic.modes
from mbe_automation.structure.crystal import SYMMETRY_TOLERANCE_STRICT

def verify_adps_roundtrip(
    cif_path: str,
    original_structure: pymatgen.core.Structure,
    thermal_displacements: mbe_automation.dynamics.harmonic.modes.ThermalDisplacements,
    temperature_idx: int = 0,
    symprec: float = SYMMETRY_TOLERANCE_STRICT,
    atol_adp: float = 1e-4,
) -> bool:
    """
    Verifies that ADPs saved to a CIF file match the input ADPs after accounting for 
    coordinate transformations and symmetry expansion.

    Args:
        cif_path: Path to the saved CIF file.
        original_structure: The original Pymatgen structure used for calculation.
        thermal_displacements: The original computed thermal displacements object.
        temperature_idx: Index of the temperature to compare.
        symprec: Symmetry precision used for symmetry analysis.
        atol_adp: Absolute tolerance for ADP comparison.

    Returns:
        True if verification passes, False otherwise (prints details).
    """
    # 1. Load CIF Data via gemmi
    try:
        doc = gemmi.cif.read_file(cif_path)
        block = doc.sole_block()
        small_structure = gemmi.make_small_structure_from_block(block)
    except Exception as e:
        print(f"CIF roundtrip failed: could not read CIF - {e}")
        return False

    # 2. Extract Representative Atom Data and ADPs
    rep_sites = {}
    u_cif_map = {}
    for site in small_structure.sites:
        rep_sites[site.label] = np.array([site.fract.x, site.fract.y, site.fract.z])
        if site.aniso.nonzero():
            u_cif_map[site.label] = np.array(site.aniso.as_mat33().tolist())

    if not u_cif_map:
        print(f"CIF roundtrip failed: missing ADP data in CIF - {cif_path}")
        return False

    # 3. Construct Transformation Matrix (CIF to Cartesian)
    # Reconstruct lattice from original structure (assuming it matches CIF)
    # The convention in modes.py _to_cif is:
    # N = diag(norm(vrec)) where vrec are cols of inv(lattice_vectors)
    # lattice_vectors = dynamical_matrix.primitive.cell.T (columns)
    # We need A = lattice vectors in columns
    
    lattice = original_structure.lattice
    A = lattice.matrix.T # Pymatgen lattice.matrix is row vectors, take transpose for columns
    A_inv = np.linalg.inv(A)
    
    # Reciprocal lattice vectors (crystallographic definition without 2pi)
    # B = inv(A).T. For pymatgen, lattice.reciprocal_lattice.matrix is also row vectors.
    # We need columns of B for N calculation as per modes.py convention?
    # modes.py: vrec in np.linalg.inv(lattice_vectors) (which is inv(A))
    lattice_vectors = A
    N_diag = [np.linalg.norm(vrec) for vrec in np.linalg.inv(lattice_vectors)]
    N = np.diag(N_diag)
    
    # Forward transform used in modes.py: U_cif = ANinv @ U_cart @ ANinv.T
    # where ANinv = inv(lattice_vectors @ N)
    # So U_cart = (lattice_vectors @ N) @ U_cif @ (lattice_vectors @ N).T
    
    M_transform = lattice_vectors @ N

    # 4. Expand Symmetry and Compare
    sga = SpacegroupAnalyzer(original_structure, symprec=symprec)
    # Getting symmetrized structure helps identify which input atom maps to which Wyckoff position
    # But strictly we need to map input atoms to the Representative atoms from CIF.
    
    # Input Reference ADPs
    input_adps_raw = thermal_displacements.mean_square_displacements_matrix_diagonal[temperature_idx]
    
    # We should calculate the expected symmetrized ADPs from input
    if symprec is not None:
        expected_adps = mbe_automation.dynamics.harmonic.modes.symmetrize_adps(
            original_structure, input_adps_raw, symprec=symprec
        )
    else:
        expected_adps = input_adps_raw

    # Reconstructed ADPs array
    reconstructed_adps = np.zeros_like(expected_adps)
    
    # Iterate over all atoms in the full structure
    errors = []
    
    for i, site in enumerate(original_structure):
        # We need to find which representative atom corresponds to this site `i`
        found = False
        
        # Search through representative sites
        for label, rep_frac in rep_sites.items():
            # Check if symmetry operation maps rep_frac -> site.frac_coords
            for op in sga.get_symmetry_operations():
                transformed_coords = op.operate(rep_frac)
                dist, _ = lattice.get_distance_and_image(transformed_coords, site.frac_coords)
                
                if dist < symprec:
                    # Found the mapping!
                    # 1. Get U_cif for rep
                    if label not in u_cif_map:
                         # This might happen if 'label' is in loops for positions but not in ADPs loops
                         continue 
                         
                    U_cif_rep = u_cif_map[label]
                    
                    # 2. Convert rep to Cartesian
                    U_cart_rep = M_transform @ U_cif_rep @ M_transform.T
                    
                    # 3. Rotate to current atom frame
                    # U_site = R @ U_rep @ R.T
                    R = A @ op.rotation_matrix @ A_inv
                    U_cart_site = R @ U_cart_rep @ R.T
                    
                    reconstructed_adps[i] = U_cart_site
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"CIF roundtrip failed: could not find symmetry mapping for atom {i} ({site.species})")
            return False

        diff = np.abs(reconstructed_adps[i] - expected_adps[i])
        max_diff = np.max(diff)
        errors.append(max_diff)

    max_error_total = np.max(errors)

    if max_error_total > atol_adp:
        print(f"CIF roundtrip failed (max deviation {max_error_total:.1e})")
        return False

    print(f"CIF roundtrip passed (max deviation {max_error_total:.1e})")
    return True
