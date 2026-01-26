from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Literal

import mbe_automation.storage.core
import mbe_automation.storage
from nomore_ase.core.calculator import NoMoReCalculator
from nomore_ase.optimization.engine import RefinementEngine
from nomore_ase.crystallography.cctbx_adapter import CctbxAdapter
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core import Structure as PymatgenStructure, Lattice
from mbe_automation.storage.views import to_pymatgen
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

def _map_adps(
    struct_ref: PymatgenStructure,
    struct_src: PymatgenStructure,
    adps_src_cart: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, list[int]]:
    """
    Map ADPs and Positions from source to reference using lattice transformation.

    Args:
        struct_ref: Reference structure.
        struct_src: Source structure.
        adps_src_cart: (N, 3, 3) Cartesian ADPs of the source.

    Returns:
        tuple: (mapped_adps, mapped_positions, indices) where mapped_adps/positions are in 
        the Reference Cartesian frame and indices map Source to Reference.
    """
    # Use robust matcher settings (no primitive reduction, allow supercells)
    matcher = StructureMatcher(
        primitive_cell=False, 
        scale=False, 
        attempt_supercell=True,
        comparator=ElementComparator()
    )
    transform = matcher.get_transformation(struct_src, struct_ref)

    if transform is None:
        raise ValueError("Structures do not match.")

    lattice_map, shift, site_map = transform
    
    # site_map is list of (ref_index for src_index i)
    # We want to reorder Source to match Reference.
    # So we want to find the source index for each reference index 0, 1, 2...
    # site_map[src_i] = ref_j
    # Pairs: (src_i, ref_j)
    pairs = list(enumerate(site_map))
    
    # Sort by ref_index (j)
    pairs.sort(key=lambda x: x[1])
    
    # Extract src_index (i) in order
    indices = [p[0] for p in pairs]
    
    # --- ADP Transformation ---
    # lat_src = struct_src.lattice.matrix
    # Pymatgen property inv_matrix is (3,3) where rows are reciprocal vectors
    # inv_matrix[i,j]: i=Cartesian, j=Fractional (reciprocal vector i component j?)
    # Pymatgen: f = x @ inv_matrix <=> f_j = x_i * inv_{ij}
    # So dim 0 is Cart, dim 1 is Frac.
    inv_lat_src = struct_src.lattice.inv_matrix

    adps_src_sorted = adps_src_cart[indices]
    
    # Transform Cartesian -> Fractional (Source Basis)
    # U_frac = A^T U_cart A where A = inv_lat (Cart, Frac)
    # U_uv = sum_xy (inv[x,u] * U[x,y] * inv[y,v])
    u_frac_src = np.einsum("xu,nxy,yv->nuv", inv_lat_src, adps_src_sorted, inv_lat_src)

    # Apply Lattice Vector Permutation (Basis Change)
    # lattice_map (M) maps Source Lattice -> Ref Lattice (Ref, Src) ???
    # M maps L_src to L_ref? Pymatgen: "matrix that transforms source to target".
    # Usually L_tgt = M @ L_src. M is (Ref, Src).
    # Then U_ref = (M^-1)^T U_src (M^-1).
    # m_inv = inv(M) is (Src, Ref).
    m_inv = np.linalg.inv(lattice_map)
    
    # U_ref_ij = sum_kl (m_inv[k,i] * U_src_kl * m_inv[l,j])
    u_frac_aligned = np.einsum("ki,nkl,lj->nij", m_inv, u_frac_src, m_inv)

    # Transform Fractional -> Cartesian (Reference Basis)
    # lat_ref (M) is (Frac, Cart).
    # U_cart = M^T U_frac M.
    # U_cart_ik = sum_ab (lat_ref[a,i] * U_frac_ab * lat_ref[b,k])
    lat_ref = struct_ref.lattice.matrix
    mapped_adps = np.einsum("ai,nab,bk->nik", lat_ref, u_frac_aligned, lat_ref)
    
    # --- Position Transformation (Verification) ---
    pos_src_sorted = struct_src.cart_coords[indices]
    
    # Cart -> Frac (Source)
    # pos_frac = pos @ inv_lat_src (coord vector is row)
    pos_frac_src = pos_src_sorted @ inv_lat_src
    
    # Basis Change
    # x_frac_new = x_frac_old @ m_inv ?
    # x_ref = x_src M^-1 ?
    # x_ref_i = sum_k x_src_k m_inv_ki
    # This matches matrix mult x @ m_inv
    pos_frac_aligned = pos_frac_src @ m_inv
    
    # Apply Shift (from get_transformation)
    pos_frac_shifted = pos_frac_aligned + shift
    
    # Frac -> Cart (Reference)
    mapped_coords = pos_frac_shifted @ lat_ref
    
    return mapped_adps, mapped_coords, indices

    print(f"Manual Sanity Check RMSD: {rmsd:.4f} Å (Max Dist: {max_dist:.4f} Å)")
    print(f"Manual Centered RMSD:     {rmsd_centered:.4f} Å (Should match StructureMatcher)")

def _extract_u_cart_exp(
    cif_path: str, 
    reference_structure: mbe_automation.storage.core.Structure | None = None,
) -> npt.NDArray:
    """
    Extract experimental Cartesian ADPs from a CIF file.
    
    Uses nomore_ase's CctbxAdapter to parse the CIF.
    
    If 'reference_structure' is provided, it matches atoms and lattice basis to 
    Reference Primitive cell using Pymatgen StructureMatcher and transforms ADPs.
    
    Args:
        cif_path: Path to the CIF file.
        reference_structure: Optional structure to match atom ordering against.
        
    Returns:
        npt.NDArray: Array of Cartesian ADPs with shape (N_atoms, 3, 3) in Å².
    """
    adapter = CctbxAdapter(cif_path)
    
    # Expand to P1 to get full unit cell content
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
        # Convert CIF to Pymatgen Structure
        struct_cif = PymatgenStructure(
            lattice=Lattice.from_parameters(*cif_cell_par),
            species=cif_symbols,
            coords=cif_pos_cart,
            coords_are_cartesian=True
        )

        # Convert Reference to Pymatgen Structure
        struct_ref = to_pymatgen(structure=reference_structure)
        
        # Map and Transform
        try:
            u_cart, mapped_coords, indices = _map_adps(struct_ref, struct_cif, u_cart)
            
            # --- Verification ---
            # Compute RMSD calculation (now very simple as coords are aligned)
            # Handle PBC wrapping not needed if mapped_coords are already in Target Basis roughly
            # But StructureMatcher mapping usually aligns them well.
            ref_coords = struct_ref.cart_coords
            diff_vecs = mapped_coords - ref_coords
            
            # Since mapped_coords includes shift, should be close.
            # Handle PBC
            frac_diff = diff_vecs @ struct_ref.lattice.inv_matrix
            frac_diff -= np.round(frac_diff)
            cart_diff = frac_diff @ struct_ref.lattice.matrix
            
            dists_sq = np.sum(cart_diff**2, axis=1)
            rmsd = np.sqrt(np.mean(dists_sq))
            
            print(f"ADP Mapping Verification RMSD: {rmsd:.4f} Å")
            
        except ValueError as e:
            raise ValueError(f"Robust ADP mapping failed: {e}")
            
    return u_cart

def fit_to_adps(
    fc: mbe_automation.storage.core.ForceConstants,
    cif_path: str,
    temperature: float,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
    restraint_weight: float = DEFAULT_RESTRAINT_WEIGHT,
    bounds: tuple[float, float] = (10.0, 1e4), # Default bounds in cm-1
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

    Returns:
        npt.NDArray: Refined frequencies in cm⁻¹.
    """
    # 0. Extract Experimental ADPs
    u_cart_exp = _extract_u_cart_exp(
        cif_path, 
        reference_structure=fc.primitive,
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

    print(result)

    if not result['success']:
        print(f"Warning: ADP fitting result reported failure: {result.get('message', 'Unknown error')}")

    return result['frequencies']

