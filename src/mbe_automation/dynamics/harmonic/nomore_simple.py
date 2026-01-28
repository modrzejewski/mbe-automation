import numpy as np
import numpy.typing as npt
from typing import Literal, Tuple, Dict, Any, List
from scipy.optimize import minimize # type: ignore
import mbe_automation.storage
from mbe_automation.api.classes import ForceConstants
from mbe_automation.dynamics.harmonic.modes import (
    phonopy_k_point_grid,
    at_k_points,
    _absolute_amplitude_eq_2,
    _to_cif
)
import phonopy.physical_units
import mbe_automation.storage.core
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core import Structure as PymatgenStructure, Lattice
from mbe_automation.storage.views import to_pymatgen
from nomore_ase.crystallography.cctbx_adapter import CctbxAdapter
from cctbx import adptbx

import scipy.sparse.csgraph

def _group_degenerate_bands(
    initial_freqs_q_THz: npt.NDArray[np.float64],
    tolerance: float
) -> Tuple[npt.NDArray[np.int64], int]:
    """
    Group bands that are degenerate at any q-point.
    
    If |freq[q, i] - freq[q, j]| < tolerance for ANY q, bands i and j are linked.
    Transitive closure is applied to find disconnected groups.
    
    Returns:
        - band_to_group_map: (n_bands,) integers group index for each band.
        - n_groups: Total number of unique groups.
    """
    n_bands = initial_freqs_q_THz.shape[1]
    n_qpoints = initial_freqs_q_THz.shape[0]
    
    adj = np.zeros((n_bands, n_bands), dtype=int)
    
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            diffs = np.abs(initial_freqs_q_THz[:, i] - initial_freqs_q_THz[:, j])
            if np.any(diffs < tolerance):
                adj[i, j] = 1
                adj[j, i] = 1 
                
    n_groups, labels = scipy.sparse.csgraph.connected_components(adj, directed=False)
    
    return labels, n_groups

def _map_adps(
    struct_ref: PymatgenStructure,
    struct_src: PymatgenStructure,
    adps_src_cart: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, list[int]]:
    """
    Map ADPs and Positions from source to reference using lattice transformation.
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
    # Pair: (src_i, ref_j)
    pairs = list(enumerate(site_map))
    pairs.sort(key=lambda x: x[1])
    indices = [p[0] for p in pairs]
    
    # --- ADP Transformation ---
    inv_lat_src = struct_src.lattice.inv_matrix
    adps_src_sorted = adps_src_cart[indices]
    
    # Transform Cartesian -> Fractional (Source Basis)
    u_frac_src = np.einsum("xu,nxy,yv->nuv", inv_lat_src, adps_src_sorted, inv_lat_src)

    # Apply Lattice Vector Permutation (Basis Change)
    m_inv = np.linalg.inv(lattice_map)
    u_frac_aligned = np.einsum("ki,nkl,lj->nij", m_inv, u_frac_src, m_inv)

    # Transform Fractional -> Cartesian (Reference Basis)
    lat_ref = struct_ref.lattice.matrix
    mapped_adps = np.einsum("ai,nab,bk->nik", lat_ref, u_frac_aligned, lat_ref)
    
    # --- Position Transformation (Verification) ---
    pos_src_sorted = struct_src.cart_coords[indices]
    pos_frac_src = pos_src_sorted @ inv_lat_src
    pos_frac_aligned = pos_frac_src @ m_inv
    pos_frac_shifted = pos_frac_aligned + shift
    mapped_coords = pos_frac_shifted @ lat_ref
    
    return mapped_adps, mapped_coords, indices

def _extract_u_cart_exp(
    cif_path: str, 
    reference_structure: mbe_automation.storage.core.Structure | None = None,
) -> npt.NDArray:
    """Extract experimental Cartesian ADPs from a CIF file. Returns ADPs in Å²."""
    adapter = CctbxAdapter(cif_path)
    
    # Expand to P1 to get full unit cell content
    xs_p1 = adapter.xray_structure.expand_to_p1(sites_mod_positive=True)
    
    cif_pos_cart = xs_p1.sites_cart().as_numpy_array()
    cif_symbols = [s.element_symbol() for s in xs_p1.scatterers()]
    cif_cell_par = xs_p1.unit_cell().parameters()
    
    # Extract ADPs
    unit_cell = xs_p1.unit_cell()
    u_cart_list = []
    
    for sc in xs_p1.scatterers():
        if sc.u_iso != -1.0 and sc.u_star == (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0):
            u_cart_val = adptbx.u_iso_as_u_cart(sc.u_iso)
        else:
            u_cart_val = adptbx.u_star_as_u_cart(unit_cell, sc.u_star)
        
        u_tensor = np.array([
            [u_cart_val[0], u_cart_val[3], u_cart_val[4]],
            [u_cart_val[3], u_cart_val[1], u_cart_val[5]],
            [u_cart_val[4], u_cart_val[5], u_cart_val[2]]
        ])
        u_cart_list.append(u_tensor)
        
    u_cart = np.array(u_cart_list)
    
    if reference_structure is not None:
        struct_cif = PymatgenStructure(
            lattice=Lattice.from_parameters(*cif_cell_par),
            species=cif_symbols,
            coords=cif_pos_cart,
            coords_are_cartesian=True
        )

        struct_ref = to_pymatgen(structure=reference_structure)
        
        try:
            u_cart, mapped_coords, indices = _map_adps(struct_ref, struct_cif, u_cart)
        except ValueError as e:
            raise ValueError(f"Robust ADP mapping failed: {e}")
            
    return u_cart

    return u_cart

def _flatten_u(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Flatten (N, 3, 3) ADPs into (N*6,) array of unique components.
    Upper triangle: xx, yy, zz, xy, xz, yz
    """
    flat = []
    for i in range(len(u)):
        # xx, yy, zz, xy, xz, yz
        flat.extend([
            u[i, 0, 0], u[i, 1, 1], u[i, 2, 2],
            u[i, 0, 1], u[i, 0, 2], u[i, 1, 2]
        ])
    return np.array(flat)

def _compute_adps(
    frequencies_q: npt.NDArray[np.float64],
    eigenvectors_q: npt.NDArray[np.complex128],
    masses_AMU: npt.NDArray[np.float64],
    temperature_K: float
) -> npt.NDArray[np.float64]:
    """
    Compute Cartesian ADPs from frequencies and eigenvectors.
    
    Args:
        frequencies_q: (n_q, n_modes) frequencies in THz.
        eigenvectors_q: (n_q, n_modes, 3*n_atoms) eigenvectors stored as rows.
        masses_AMU: (n_atoms,) atomic masses in AMU.
        temperature_K: Temperature in Kelvin.
        
    Returns:
        (n_atoms, 3, 3) Cartesian ADPs (U_cart) in Å².
    """
    n_qpoints = len(frequencies_q)
    n_atoms = len(masses_AMU)
    
    mean_sq_disp = np.zeros((n_atoms * 3, n_atoms * 3), dtype=np.float64)
    
    for i_q in range(n_qpoints):
        freqs = frequencies_q[i_q]
        # Filter low-frequency modes (e.g. acoustic modes at Gamma) to avoid divergence
        mask = freqs > 1e-4
        
        if not np.any(mask):
            continue
            
        freqs_masked = freqs[mask]
        ejk_masked = eigenvectors_q[i_q][mask]
        
        # Amplitude
        Ajk_primitive = _absolute_amplitude_eq_2(freqs_masked, temperature_K, masses_AMU)
        
        # Combine with eigenvectors
        # Ajk_primitive is (n_masked, 1) broadcasted over (n_masked, 3*n_atoms)
        Ajk_ejk = Ajk_primitive * ejk_masked
        
        U_q_local = np.einsum("jk,jl->kl", Ajk_ejk, Ajk_ejk.conj()).real
        mean_sq_disp += U_q_local
        
    mean_sq_disp /= n_qpoints
    
    # Extract diagonal blocks (ADPs are 3x3 per atom)
    msd_reshaped = mean_sq_disp.reshape(n_atoms, 3, n_atoms, 3)
    mean_sq_disp_diagonal = np.einsum('kikj->kij', msd_reshaped)
    
    return mean_sq_disp_diagonal

def fit_to_adps(
        force_constants: ForceConstants,
        cif_path: str,
        temperature_K: float,
        mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
        optimize_mask: npt.NDArray[np.bool_] | None = None,
        degeneracy_tolerance: float = 1e-4,
        bounds: Tuple[float, float] = (1e-6, np.inf),
        max_iterations: int = 200,
) -> Dict[str, Any]:
    """
    Fit phonon frequencies to experimental ADPs from a CIF file.
    
    See _fit_to_adps for implementation details.
    """
    # Extract ADPs from CIF
    u_exp = _extract_u_cart_exp(
        cif_path, 
        reference_structure=force_constants.primitive
    )
    
    return _fit_to_adps(
        force_constants=force_constants,
        u_exp=u_exp,
        temperature_K=temperature_K,
        mesh_size=mesh_size,
        optimize_mask=optimize_mask,
        degeneracy_tolerance=degeneracy_tolerance,
        bounds=bounds,
        max_iterations=max_iterations
    )

def _fit_to_adps(
        force_constants: ForceConstants,
        u_exp: npt.NDArray[np.float64],
        temperature_K: float,
        mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
        optimize_mask: npt.NDArray[np.bool_] | None = None,
        degeneracy_tolerance: float = 1e-4,
        bounds: Tuple[float, float] = (1e-6, np.inf),
        max_iterations: int = 200,
) -> Dict[str, Any]:
    """
    Fit phonon frequencies to experimental ADPs using band shifts with degeneracy handling.

    This function optimizes a constant frequency shift for each phonon BAND GROUP.
    Bands are grouped if they are degenerate (within tolerance) at ANY k-point.
    Degenerate bands share the same shift parameter to preserve symmetry/couplings.
    
    The optimization assumes that the eigenvectors (normal modes) are rigid and do not
    change during the frequency refinement (Rigid Mode Approximation).
    
    Args:
        force_constants: The harmonic force constants model.
        u_exp: (N_atoms, 3, 3) Experimental Cartesian ADPs in Å².
        temperature_K: Temperature (K). 
        mesh_size: k-point sampling mesh.
        optimize_mask: (n_bands,) Boolean mask. If True, the band is allowed to shift.
                       If any band in a degenerate group is masked True, the group is optimized.
        degeneracy_tolerance: Tolerance (THz) for linking degenerate bands.
        bounds: (min, max) allowed frequencies in THz (checked approximately or via penalties).
        max_iterations: Maximum iterations for the optimizer.
        
    Returns:
        Dictionary containing optimization results:
            - success: (bool) Whether optimization succeeded.
            - original_frequencies: (N_q, N_bands) Initial frequencies in THz.
            - refined_frequencies: (N_q, N_bands) Refined frequencies in THz.
            - message: (str) Optimizer message.
            - residual: (float) Final objective function value.
            - shifts: (N_bands,) Applied frequency shifts (reconstructed).
    """
    
    # 1. Setup Phonopy and Grid
    ph = mbe_automation.storage.to_phonopy(force_constants)
    qpoints, weights = phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=mesh_size,
        use_symmetry=False,
        center_at_gamma=False
    )
    
    n_qpoints = len(qpoints)
    
    # 2. Pre-calculate Eigenvectors and Initial Frequencies
    initial_freqs_q_THz, eigenvectors_q = at_k_points(
        dynamical_matrix=ph.dynamical_matrix,
        k_points=qpoints,
        compute_eigenvecs=True,
        eigenvectors_storage="rows"
    )
    
    if eigenvectors_q is None:
        raise ValueError("Failed to compute eigenvectors.") 
        
    n_bands = initial_freqs_q_THz.shape[1]
    n_atoms = len(ph.primitive.masses)
    masses_AMU = ph.primitive.masses

    # 3. Identify Degeneracy Groups
    band_group_labels, n_groups = _group_degenerate_bands(initial_freqs_q_THz, degeneracy_tolerance)
    
    # Setup Optimization Parameters (Group Shifts)
    if optimize_mask is None:
        # Default: Optimize all EXCEPT the first 3 bands (Acoustic)
        # We assume bands are sorted by frequency (standard Phonopy behavior)
        # Acoustic bands (indices 0, 1, 2) correspond to translation at Gamma.
        optimize_mask = np.ones(n_bands, dtype=bool)
        if n_bands >= 3:
            optimize_mask[:3] = False
    
    if len(optimize_mask) != n_bands:
        raise ValueError(f"optimize_mask length {len(optimize_mask)} must match number of bands {n_bands}")
    
    # Identify which groups to optimize
    # If any band in group G is masked True, then group G is optimized.
    group_optimize_mask = np.zeros(n_groups, dtype=bool)
    for band_idx, group_idx in enumerate(band_group_labels):
        if optimize_mask[band_idx]:
            group_optimize_mask[group_idx] = True
            
    n_params = np.sum(group_optimize_mask)
    initial_group_shifts = np.zeros(n_params, dtype=np.float64)
    
    # Pre-calculate derived data for objective function
    target_u_flat = _flatten_u(u_exp) if u_exp.ndim > 1 else u_exp
    
    # Bounds logic (approximate per group constraints)
    # We set bounds on the shift such that the LOWEST frequency in the group stays above min_bound
    
    param_to_group_map = np.where(group_optimize_mask)[0]
    shift_bounds = []
    min_freq_allowed, max_freq_allowed = bounds
    
    for i_param in range(n_params):
        group_idx = param_to_group_map[i_param]
        # Find all bands in this group
        bands_in_group = np.where(band_group_labels == group_idx)[0]
        
        # Get min/max freq across ALL bands in this group and ALL q-points
        # Because shift is applied equally to all bands in group
        bands_freqs = initial_freqs_q_THz[:, bands_in_group]
        
        lower_bound = min_freq_allowed - np.min(bands_freqs)
        upper_bound = max_freq_allowed - np.max(bands_freqs)
        
        shift_bounds.append((lower_bound, upper_bound))
    
    # 4. Objective Function
    def objective(group_shift_params):
        # 1. Map parameters to full group shifts
        full_group_shifts = np.zeros(n_groups, dtype=np.float64)
        full_group_shifts[group_optimize_mask] = group_shift_params
        
        # 2. Map group shifts to band shifts
        # full_shifts[band] = full_group_shifts[band_group_labels[band]]
        full_band_shifts = full_group_shifts[band_group_labels]
        
        # 3. Apply shifts
        current_freqs_q = initial_freqs_q_THz + full_band_shifts
        
        
        # 4. Calculate ADPs
        T = temperature_K
        calc_u_3x3 = _compute_adps(current_freqs_q, eigenvectors_q, masses_AMU, T)
        calc_u_flat = _flatten_u(calc_u_3x3)
        
        diff = calc_u_flat - target_u_flat
        obj = np.sum(diff**2) * 1e6 # Scale
        
        return obj

    # 5. Run Optimization
    if n_params > 0:
        res = minimize(
            objective,
            initial_group_shifts,
            method='L-BFGS-B',
            bounds=shift_bounds,
            options={'maxiter': max_iterations, 'disp': False, 'gtol': 1e-10, 'eps': 1e-6}
        )
        
        # 6. Final Results
        final_group_params = res.x
        final_full_group_shifts = np.zeros(n_groups, dtype=np.float64)
        final_full_group_shifts[group_optimize_mask] = final_group_params
        
        final_band_shifts = final_full_group_shifts[band_group_labels]
        final_freqs_q = initial_freqs_q_THz + final_band_shifts
        
        success = res.success
        message = res.message
        residual = res.fun / 1e6
        
    else:
        # No optimization requested/needed
        final_band_shifts = np.zeros(n_bands)
        final_freqs_q = initial_freqs_q_THz
        success = True
        message = "No parameters to optimize."
        
        # Calculate residual for initial state
        T = temperature_K
        calc_u_3x3 = _compute_adps(final_freqs_q, eigenvectors_q, masses_AMU, T)
        calc_u_flat = _flatten_u(calc_u_3x3)
        diff = calc_u_flat - target_u_flat
        residual = np.sum(diff**2)

    # Recalculate u_calc (or reuse if available, but cheap to recalc)
    T = temperature_K
    final_u_3x3 = _compute_adps(final_freqs_q, eigenvectors_q, masses_AMU, T)
    final_u_calc = _flatten_u(final_u_3x3)

    return {
        "success": success,
        "original_frequencies": initial_freqs_q_THz,
        "refined_frequencies": final_freqs_q,
        "shifts": final_band_shifts,
        "message": message,
        "residual": residual,
        "u_calc": final_u_calc
    }

def _print_adp_comparison(
    u_ref: npt.NDArray[np.float64],
    u_approx: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    title: str = "Comparison of ADPs",
    label_ref: str = "Ref U_iso",
    label_approx: str = "Approx U_iso"
) -> None:
    """unique helper to print ADP comparison table."""
    diff = u_approx - u_ref
    norm_diff = np.linalg.norm(diff, axis=(1, 2))  # Frobenius norm per atom
    
    print(f"\n{title}:")
    print(f"{'Atom':<6} {label_ref + ' (A^2)':<20} {label_approx + ' (A^2)':<20} {'Diff Norm (A^2)':<20}")
    print("-" * 75)
    
    for i in range(len(masses)):
        # Calculate U_iso = trace(U) / 3
        u_iso_ref = np.trace(u_ref[i]) / 3.0
        u_iso_approx = np.trace(u_approx[i]) / 3.0
        print(f"{i:<6} {u_iso_ref:<20.6f} {u_iso_approx:<20.6f} {norm_diff[i]:<20.6f}")

    avg_diff = np.mean(norm_diff)
    print(f"\nAverage difference (Frobenius norm) per atom: {avg_diff:.6f} A^2")

def _print_frequency_comparison(
    freqs_initial_THz: npt.NDArray[np.float64],
    freqs_refined_THz: npt.NDArray[np.float64],
    optimize_mask: npt.NDArray[np.bool_] | None = None
) -> None:
    """Helper to print starting vs refined frequencies in cm^-1."""
    
    # Check if we have multiple q-points; usually _self_fit uses this for Gamma point only (1 q-point)
    # If multiple q-points, we only print the first one (Gamma) or loop if needed.
    # The _self_fit calls _fit_to_adps with mesh_size="gamma", so we expect 1 q-point.
    
    n_q = freqs_initial_THz.shape[0]
    n_bands = freqs_initial_THz.shape[1]
    
    to_cm = phonopy.physical_units.get_physical_units().THzToCm
    
    print("\nComparison of Frequencies (Gamma point):")
    print(f"{'Mode':<6} {'Initial (cm^-1)':<20} {'Refined (cm^-1)':<20} {'Shift (cm^-1)':<20} {'Opt?':<6}")
    print("-" * 76)
    
    # Assuming q=0 is what we care about (since we fit Gamma)
    # If the optimization was done on multiple q-points (which _fit_to_adps supports), we might want to iterate.
    # But here we know it is for Gamma approximation.
    
    for i in range(n_q):
        if n_q > 1:
            print(f"--- q-point index {i} ---")
            
        freqs_init_cm = freqs_initial_THz[i] * to_cm
        freqs_refined_cm = freqs_refined_THz[i] * to_cm
        diff_cm = freqs_refined_cm - freqs_init_cm
        
        for band_idx in range(n_bands):
            is_opt = ""
            if optimize_mask is not None and optimize_mask[band_idx]:
                is_opt = "*"
            print(f"{band_idx:<6} {freqs_init_cm[band_idx]:<20.2f} {freqs_refined_cm[band_idx]:<20.2f} {diff_cm[band_idx]:<20.2f} {is_opt:<6}")

def _fit_gamma_point_model(
    force_constants: ForceConstants,
    u_exp: npt.NDArray[np.float64],
    temperature_K: float,
    optimize_mask: npt.NDArray[np.bool_] | None = None,
    degeneracy_tolerance: float = 1e-4,
    bounds: Tuple[float, float] = (1e-6, np.inf),
    max_iterations: int = 200,
) -> Dict[str, Any]:
    """
    Simplified frequency fitter designed specifically for Gamma-point models.
    
    This function optimizes random offsets to the Gamma-point frequencies
    to match the target ADPs (u_exp). It is a specialized version of
    _fit_to_adps restricted to a single q-point (Gamma).
    
    Args:
        force_constants: The harmonic force constants model.
        u_exp: (N_atoms, 3, 3) Experimental/Target Cartesian ADPs in Å².
        temperature_K: Temperature (K).
        optimize_mask: (n_bands,) Boolean mask. If True, the band is allowed to shift.
        degeneracy_tolerance: Tolerance (THz) for linking degenerate bands.
        bounds: (min, max) allowed frequencies in THz.
        max_iterations: Maximum iterations for the optimizer.
        
    Returns:
        Dictionary containing optimization results.
    """
    # 1. Setup Phonopy (Gamma point only)
    # We don't need phonopy_k_point_grid heavily here, just get Gamma point data.
    ph = mbe_automation.storage.to_phonopy(force_constants)
    
    # 2. Calculate Eigenvectors and Initial Frequencies at Gamma
    # Direct calculation at Gamma (q=[0,0,0])
    q_gamma = np.array([[0., 0., 0.]])
    
    initial_freqs_q_THz, eigenvectors_q = at_k_points(
        dynamical_matrix=ph.dynamical_matrix,
        k_points=q_gamma,
        compute_eigenvecs=True,
        eigenvectors_storage="rows"
    )
    
    if eigenvectors_q is None:
        raise ValueError("Failed to compute eigenvectors.")
        
    n_bands = initial_freqs_q_THz.shape[1]
    masses_AMU = ph.primitive.masses
    
    # 3. Identify Degeneracy Groups (at Gamma)
    # _group_degenerate_bands handles (n_q, n_bands), here n_q=1
    band_group_labels, n_groups = _group_degenerate_bands(initial_freqs_q_THz, degeneracy_tolerance)
    
    # Setup Optimization Parameters
    if optimize_mask is None:
        optimize_mask = np.ones(n_bands, dtype=bool)
        if n_bands >= 3:
            optimize_mask[:3] = False # exclude acoustic
            
    if len(optimize_mask) != n_bands:
        raise ValueError(f"optimize_mask length {len(optimize_mask)} must match n_bands {n_bands}")
        
    group_optimize_mask = np.zeros(n_groups, dtype=bool)
    for band_idx, group_idx in enumerate(band_group_labels):
        if optimize_mask[band_idx]:
            group_optimize_mask[group_idx] = True
            
    n_params = np.sum(group_optimize_mask)
    initial_group_shifts = np.zeros(n_params, dtype=np.float64)
    
    target_u_flat = _flatten_u(u_exp)
    
    # Bounds
    param_to_group_map = np.where(group_optimize_mask)[0]
    shift_bounds = []
    min_freq_allowed, max_freq_allowed = bounds
    
    # Since n_q=1, simplification:
    freqs_gamma = initial_freqs_q_THz[0] # (n_bands,)
    
    for i_param in range(n_params):
        group_idx = param_to_group_map[i_param]
        bands_in_group = np.where(band_group_labels == group_idx)[0]
        
        group_freqs = freqs_gamma[bands_in_group]
        
        # Shift limits constrained by ALL bands in the group
        lower_bound = min_freq_allowed - np.min(group_freqs)
        upper_bound = max_freq_allowed - np.max(group_freqs)
        shift_bounds.append((lower_bound, upper_bound))
        
    # 4. Objective Function
    def objective(group_shift_params):
        full_group_shifts = np.zeros(n_groups, dtype=np.float64)
        full_group_shifts[group_optimize_mask] = group_shift_params
        
        full_band_shifts = full_group_shifts[band_group_labels]
        
        # Apply shifts (broadcast to (1, n_bands))
        current_freqs_q = initial_freqs_q_THz + full_band_shifts
        
        # Calc ADPs
        # _compute_adps expects (n_q, n_modes) and (n_q, n_modes, dof)
        calc_u_3x3 = _compute_adps(current_freqs_q, eigenvectors_q, masses_AMU, temperature_K)
        calc_u_flat = _flatten_u(calc_u_3x3)
        
        diff = calc_u_flat - target_u_flat
        return np.sum(diff**2) * 1e6
        
    # 5. Run Optimization
    if n_params > 0:
        res = minimize(
            objective,
            initial_group_shifts,
            method='L-BFGS-B',
            bounds=shift_bounds,
            options={'maxiter': max_iterations, 'disp': False, 'gtol': 1e-10, 'eps': 1e-6}
        )
        
        final_group_params = res.x
        final_full_group_shifts = np.zeros(n_groups, dtype=np.float64)
        final_full_group_shifts[group_optimize_mask] = final_group_params
        
        final_band_shifts = final_full_group_shifts[band_group_labels]
        final_freqs_q = initial_freqs_q_THz + final_band_shifts
        
        success = res.success
        message = res.message
        residual = res.fun / 1e6
        
    else:
        final_band_shifts = np.zeros(n_bands)
        final_freqs_q = initial_freqs_q_THz
        success = True
        message = "No parameters to optimize."
        
        # Initial residual
        calc_u_3x3 = _compute_adps(final_freqs_q, eigenvectors_q, masses_AMU, temperature_K)
        calc_u_flat = _flatten_u(calc_u_3x3)
        diff = calc_u_flat - target_u_flat
        residual = np.sum(diff**2)

    final_u_3x3 = _compute_adps(final_freqs_q, eigenvectors_q, masses_AMU, temperature_K)
    final_u_calc = _flatten_u(final_u_3x3)
    
    return {
        "success": success,
        "original_frequencies": initial_freqs_q_THz,
        "refined_frequencies": final_freqs_q,
        "shifts": final_band_shifts,
        "message": message,
        "residual": residual,
        "u_calc": final_u_calc
    }

def _self_fit(
    force_constants: ForceConstants,
    temperature_K: float,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float,
    max_optimized_freq_THz: float | None = None,
    bounds: Tuple[float, float] = (1e-6, 500.0)
) -> Dict[str, Any]:
    """
    Fit Gamma-point frequencies to match ADPs computed from a full k-point mesh.

    This function:
    1. Computes ADPs using the specified k-point `mesh_size`.
    2. Computes ADPs using the Gamma point approximation.
    3. Prints a comparison of the two.
    4. Runs `_fit_gamma_point_model` to optimize Gamma-point frequencies such that the
       resulting Gamma-point ADPs match the k-point ADPs as closely as possible.
    
    Args:
        force_constants: The harmonic force constants model.
        temperature_K: Temperature in Kelvin.
        mesh_size: The k-point mesh for the reference calculation (e.g. [4, 4, 4]).
        max_optimized_freq_THz: Maximum frequency (in THz) for modes to be optimized.
                                Modes above this frequency will be fixed.
                                Acoustic modes are always fixed.
        bounds: (min, max) allowed frequencies in THz. Default is (1e-6, 500.0)
                to prevent unphysical hardening of frequencies.

    Returns:
        Result dictionary from `_fit_gamma_point_model`.
    """
    print(f"--- _self_fit: effective Gamma-point approximation ---")
    print(f"Temperature: {temperature_K} K")
    print(f"Reference mesh: {mesh_size}")
    if max_optimized_freq_THz is not None:
        print(f"Max optimized frequency: {max_optimized_freq_THz} THz")
    print(f"Frequency bounds (THz): {bounds}")
    
    # 1. Reference ADPs (k-point mesh)
    ph = mbe_automation.storage.to_phonopy(force_constants)
    qpoints_ref, _ = phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=mesh_size,
        use_symmetry=False,
        center_at_gamma=False
    )
    
    # Note: We must use rows storage for compatibility with _compute_adps which expects (n_q, n_modes, n_dof)
    freqs_ref, evecs_ref = at_k_points(
        dynamical_matrix=ph.dynamical_matrix,
        k_points=qpoints_ref,
        compute_eigenvecs=True,
        eigenvectors_storage="rows"
    )
    
    if evecs_ref is None:
        raise RuntimeError("Failed to compute eigenvectors for reference mesh.")

    masses = ph.primitive.masses
    u_ref_cart = _compute_adps(freqs_ref, evecs_ref, masses, temperature_K)
    
    # 2. Approximate ADPs (Gamma)
    qpoints_gamma, _ = phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size="gamma",
        use_symmetry=False,
        center_at_gamma=False
    )
    
    freqs_gamma, evecs_gamma = at_k_points(
        dynamical_matrix=ph.dynamical_matrix,
        k_points=qpoints_gamma,
        compute_eigenvecs=True,
        eigenvectors_storage="rows"
    )
    
    if evecs_gamma is None:
        raise RuntimeError("Failed to compute eigenvectors for Gamma point.")

    u_gamma_cart = _compute_adps(freqs_gamma, evecs_gamma, masses, temperature_K)
    
    # 3. Compare (Before Optimization)
    _print_adp_comparison(
        u_ref=u_ref_cart,
        u_approx=u_gamma_cart,
        masses=masses,
        title="Comparison of ADPs [Gamma (Initial) vs Reference (k-points)]",
        label_ref="Ref U_iso",
        label_approx="Gamma U_iso"
    )
    
    # Create optimize_mask
    n_bands = freqs_gamma.shape[1]
    if max_optimized_freq_THz is not None:
        # freqs_gamma[0] are the Gamma-point frequencies
        optimize_mask = freqs_gamma[0] <= max_optimized_freq_THz
        
        # Ensure acoustic modes (first 3) are strictly False, although they should be low freq
        # typically acoustic modes are at ~0 THz, so they would be True if only checking <= max
        # BUT we explicitely want to exclude them as per standard practice.
        # However, _fit_to_adps's default logic excludes first 3 if mask is None.
        # But here mask is NOT None, so we must manually exclude them.
        if n_bands >= 3:
            optimize_mask[:3] = False
    else:
        # Default behavior: optimize all optical modes (skip first 3)
        optimize_mask = np.ones(n_bands, dtype=bool)
        if n_bands >= 3:
            optimize_mask[:3] = False
            
    # 4. Run Fit
    print("\nRunning optimization to find effective Gamma-point frequencies...")
    
    # We use u_ref_cart (the k-point result) as the target 'experimental' data
    # We perform the fit using ONLY the Gamma point (mesh_size="gamma")
    result = _fit_gamma_point_model(
        force_constants=force_constants,
        u_exp=u_ref_cart,
        temperature_K=temperature_K,
        degeneracy_tolerance=1e-4,
        optimize_mask=optimize_mask,
        bounds=bounds
    )
    
    print(f"Optimization finished. Success: {result['success']}")
    print(f"Final Residual: {result['residual']:.6f}")
    
    # 5. Print Frequency Comparison
    _print_frequency_comparison(
        freqs_initial_THz=result['original_frequencies'],
        freqs_refined_THz=result['refined_frequencies'],
        optimize_mask=optimize_mask
    )
    
    # 6. Compare ADPs (After Optimization)
    # Recalculate u_3x3 from refined frequencies to verify
    # We use the Gamma eigenvectors which are assumed fixed
    u_final_cart = _compute_adps(result['refined_frequencies'], evecs_gamma, masses, temperature_K)
    
    _print_adp_comparison(
        u_ref=u_ref_cart,
        u_approx=u_final_cart,
        masses=masses,
        title="Comparison of ADPs [Gamma (Optimized) vs Reference (k-points)]",
        label_ref="Ref U_iso",
        label_approx="Opt. Gamma U_iso"
    )
        
    return result
