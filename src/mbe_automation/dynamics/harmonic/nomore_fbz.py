import numpy as np
import numpy.typing as npt
from typing import Literal, Tuple, Dict, Any, List
from scipy.optimize import minimize # type: ignore


from .euphonic import to_euphonic_modes
from mbe_automation.dynamics.harmonic.display import print_adps_comparison, print_frequency_comparison
from mbe_automation.dynamics.harmonic.modes import (
    at_k_points
)

from mbe_automation.storage.core import ForceConstants, Structure
import mbe_automation.storage

import euphonic

INITIAL_ACOUSTIC_FREQ_CM1 = 50.0

def _group_degenerate_bands(
    freqs_THz: npt.NDArray[np.float64],
    tolerance_THz: float
) -> Tuple[npt.NDArray[np.int64], int]:
    """
    Group degenerate bands.

    Sort freqs_THz internally.
    Use numpy.diff and numpy.split to find gaps larger than tolerance_THz.

    Returns:
        band_to_group_map: (n_bands,) integers group index for each band.
        n_groups: Total number of unique groups.
    """
    # Identify degeneracies by sorting frequencies.
    sorted_indices = np.argsort(freqs_THz)
    sorted_freqs = freqs_THz[sorted_indices]
    n_freqs = len(sorted_freqs)
    
    # Locate group boundaries where frequency difference exceeds tolerance.
    diffs = np.diff(sorted_freqs)
    split_indices = np.where(diffs > tolerance_THz)[0] + 1
    
    # Partition indices into degenerate groups.
    groups_indices = np.split(np.arange(n_freqs), split_indices)
    
    # Assign group labels back to original indices.
    labels = np.zeros(n_freqs, dtype=np.int64)
    for i, group_sorted_indices in enumerate(groups_indices):
        labels[sorted_indices[group_sorted_indices]] = i
        
    n_groups = len(groups_indices)
    
    return labels, n_groups

def _flatten_u(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Flatten ADPs to unique components.
    Order: xx, yy, zz, xy, xz, yz
    """
    flat = []
    for i in range(len(u)):
         flat.extend([
            u[i, 0, 0], u[i, 1, 1], u[i, 2, 2],
            u[i, 0, 1], u[i, 0, 2], u[i, 1, 2]
         ])
    return np.array(flat)


def _get_valid_atoms_mask(
    u_exp: npt.NDArray[np.float64],
    structure: Structure,
    ignore_hydrogen: bool = False
) -> npt.NDArray[np.bool_]:
    """
    Identify atoms with valid ADPs.

    Args:
        u_exp: (N_atoms, 3, 3) Experimental ADPs.
        structure: Structure object with atomic_numbers attribute.
        ignore_hydrogen: If True, mask out Hydrogen atoms (Z=1).

    Returns:
        (N_atoms,) Boolean mask. True if atom has valid ADPs.
    """
    # Reshape to (N, 9) to check for NaNs in any component
    u_flat_check = u_exp.reshape(u_exp.shape[0], -1)
    mask = ~np.any(np.isnan(u_flat_check), axis=1)

    if ignore_hydrogen:
         # User specified force_constants.primitive has atomic_numbers
         is_hydrogen = (structure.atomic_numbers == 1)
         mask = mask & (~is_hydrogen)

    return mask


def _apply_einstein_approximation(
    initial_freqs_q_THz: npt.NDArray[np.float64],
    gamma_idx: int,
    einstein_approximation: npt.NDArray[np.bool_] | None,
    optimize_mask: npt.NDArray[np.bool_] | None,
) -> None:
    """
    Apply Einstein approximation in place.

    If a band is selected (optimize_mask=True AND einstein_approximation=True),
    remove its frequency dispersion and set to the Gamma-point frequency 
    across the entire Brillouin zone.
    """
    if einstein_approximation is not None and optimize_mask is not None:
        mask_flatten = einstein_approximation & optimize_mask
        
        if np.any(mask_flatten):
             # Set dispersion to flat (Gamma value) for these bands
             initial_freqs_q_THz[:, mask_flatten] = initial_freqs_q_THz[gamma_idx, mask_flatten]


def _apply_scaling(
    group_scaling_params: npt.NDArray[np.float64],
    n_groups: int,
    group_optimize_mask: npt.NDArray[np.bool_],
    band_group_labels: npt.NDArray[np.int64],
    initial_freqs_q_THz: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate scaled frequencies.
    """
    # Map optimization parameters to full group scalings.
    # Initialize with 1.0 (identity scaling) for all groups.
    full_group_scalings = np.ones(n_groups, dtype=np.float64)
    full_group_scalings[group_optimize_mask] = group_scaling_params
    
    # Broadcast group scalings to individual bands.
    full_band_scalings = full_group_scalings[band_group_labels]
    
    # Apply scalings to all q-points (broadcasting).
    current_freqs_THz = initial_freqs_q_THz * full_band_scalings
    
    return current_freqs_THz, full_band_scalings


def _objective_fbz_model(
    group_scaling_params: npt.NDArray[np.float64],
    n_groups: int,
    group_optimize_mask: npt.NDArray[np.bool_],
    band_group_labels: npt.NDArray[np.int64],
    initial_freqs_q_THz: npt.NDArray[np.float64],
    modes: "euphonic.QpointPhononModes",
    temperature_K: float,
    target_u_flat: npt.NDArray[np.float64],
    valid_atoms_mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Calculate objective function value.
    """
    current_freqs_THz, _ = _apply_scaling(
        group_scaling_params=group_scaling_params,
        n_groups=n_groups,
        group_optimize_mask=group_optimize_mask,
        band_group_labels=band_group_labels,
        initial_freqs_q_THz=initial_freqs_q_THz
    )
    
    # Update Euphonic frequencies.
    modes.frequencies = current_freqs_THz * euphonic.ureg.THz
    
    # 4. Calculate ADPs (U = 2 * W).
    dw = modes.calculate_debye_waller(
        temperature=euphonic.ureg.Quantity(temperature_K, "K")
    )
    u_cart_3x3 = 2.0 * dw.debye_waller.to("angstrom**2").magnitude
    
    # Filter valid atoms if mask is provided
    if valid_atoms_mask is not None:
        u_cart_3x3 = u_cart_3x3[valid_atoms_mask]
        
    calc_u_flat = _flatten_u(u_cart_3x3)
    
    diff = calc_u_flat - target_u_flat
    obj = np.sum(diff**2)
    
    return obj


def fit_fbz_model(
    force_constants: ForceConstants,
    u_exp: npt.NDArray[np.float64],
    temperature_K: float,
    mesh_size: npt.NDArray[np.int64] | List[int] | Tuple[int, int, int],
    optimize_mask: npt.NDArray[np.bool_] | None = None,
    einstein_approximation: npt.NDArray[np.bool_] | None = None,
    degeneracy_tolerance: float | None = None,
    max_iterations: int = 200,
    ignore_hydrogen_adps: bool = True,
) -> Dict[str, Any]:
    """
    Fit phonon frequencies to experimental ADPs.

    1. Generate a k-point mesh with ODD numbers of points.
    2. Calculate frequencies and eigenvectors with Euphonic.
    3. Reorder frequencies.
    4. Identify band degeneracies at the Gamma point.
    5. Optimize frequency scaling factors.

    If optimize_mask is None, optimize ONLY acoustic bands.
    If einstein_approximation is None, apply Einstein approximation ONLY on acoustic bands.

    Args:
        force_constants: Force constants object.
        u_exp: (N_atoms, 3, 3) Experimental Cartesian ADPs in "angstrom**2".
               ADPs with NaNs are skipped.
        temperature_K: Temperature in Kelvin.
        mesh_size: k-point mesh dimensions (3 integers).
                   Forced to be odd.
        optimize_mask: (n_bands,) Boolean mask. True allows band scaling.
                       Must correspond to sorted bands at Gamma.
        einstein_approximation: (n_bands,) Boolean mask. True enables Einstein approximation
                                (flat dispersion).
        degeneracy_tolerance: Tolerance (THz) for linking degenerate bands at Gamma.
        max_iterations: Maximum iterations for the optimizer.
        ignore_hydrogen_adps: If True, ignore ADPs on Hydrogen atoms (Z=1).

    Returns:
        Dictionary containing optimization results.
    """
    # Initialize Euphonic object.
    modes = to_euphonic_modes(
        force_constants=force_constants, 
        mesh_size=mesh_size, 
        odd_numbers=True
    )
    
    # 2. Reorder frequencies.
    modes.reorder_frequencies(reorder_gamma=True)
    
    # Extract initial frequencies in THz.
    initial_freqs_q_THz = modes.frequencies.to("THz").magnitude
    n_q, n_bands = initial_freqs_q_THz.shape
    assert n_bands == force_constants.primitive.n_atoms * 3

    # Compute initial ADPs.
    dw_init = modes.calculate_debye_waller(temperature=euphonic.ureg.Quantity(temperature_K, "K"))
    initial_u_3x3 = 2.0 * dw_init.debye_waller.to("angstrom**2").magnitude
    
    # 3. Identify degeneracy groups at the Gamma point.
    # Gamma point [0, 0, 0] is required.
    q_norms = np.linalg.norm(modes.qpts, axis=1)
    gamma_idx = np.argmin(q_norms)
    
    # Verify Gamma point presence.
    if q_norms[gamma_idx] > 1e-5:
        raise RuntimeError("Gamma point [0, 0, 0] not found in the generated q-point mesh. Ensure mesh is odd-numbered.")
    
    freqs_gamma = initial_freqs_q_THz[gamma_idx]

    if np.any(np.diff(freqs_gamma) < 0):
        raise ValueError("Frequencies at Gamma point must be non-decreasing.")
    
    # Determine degenerate groups based on Gamma frequencies.
    if degeneracy_tolerance is None:
        # None: Optimize each band independently.
        band_group_labels = np.arange(n_bands, dtype=np.int64)
        n_groups = n_bands
    else:
        # Group degenerate bands.
        band_group_labels, n_groups = _group_degenerate_bands(freqs_gamma, degeneracy_tolerance)
    
    if optimize_mask is None:
        optimize_mask = np.zeros(n_bands, dtype=bool)
        if n_bands >= 3:
            optimize_mask[:3] = True
    elif len(optimize_mask) < n_bands:
        optimize_mask = np.pad(
            optimize_mask, 
            (0, n_bands - len(optimize_mask)), 
            mode='constant', 
            constant_values=False
        )

    if einstein_approximation is None:
        einstein_approximation = np.zeros(n_bands, dtype=bool)
        if n_bands >= 3:
            einstein_approximation[:3] = True
    elif len(einstein_approximation) < n_bands:
        einstein_approximation = np.pad(
            einstein_approximation, 
            (0, n_bands - len(einstein_approximation)), 
            mode='constant', 
            constant_values=False
        )

    if n_q == 1 and not np.all(einstein_approximation[:3]):
        raise ValueError(
            "Gamma-only sampling requires Einstein approximation for acoustic modes."
        )

    # If using Einstein approximation for acoustic modes, set initial frequency.
    if np.any(optimize_mask[:3] & einstein_approximation[:3]):
        # Convert 50 cm^-1 to THz
        init_freq_THz = (INITIAL_ACOUSTIC_FREQ_CM1 * euphonic.ureg("cm^-1")).to("THz").magnitude
        initial_freqs_q_THz[gamma_idx, :3] = init_freq_THz

    # Assumption: Either all acoustic modes are optimized, or none.
    if np.any(optimize_mask[:3]):
        # Ensure all acoustic modes are optimized if any are
        if not np.all(optimize_mask[:3]):
             raise ValueError("Partial optimization of acoustic modes is not supported. Optimize all 3 (indices 0-2) or none.")

    # Map bands to parameter groups.
    group_optimize_mask = np.zeros(n_groups, dtype=bool)
    for band_idx, group_idx in enumerate(band_group_labels):
        if optimize_mask[band_idx]:
            group_optimize_mask[group_idx] = True
            
    n_params = np.sum(group_optimize_mask)
    # Initialize scalings to 1.0
    initial_group_scalings = np.ones(n_params, dtype=np.float64)
    
    # 4a. Identify valid atoms.
    valid_atoms_mask = _get_valid_atoms_mask(
        u_exp, 
        force_constants.primitive, 
        ignore_hydrogen_adps
    )

    # Filter experimental ADPs
    u_exp_valid = u_exp[valid_atoms_mask]
    
    if len(u_exp_valid) == 0:
        raise ValueError("No valid ADPs found (all are NaN). Cannot fit.")

    target_u_flat = _flatten_u(u_exp_valid)

    # 4b. Apply Einstein Approximation.
    _apply_einstein_approximation(
        initial_freqs_q_THz,
        gamma_idx,
        einstein_approximation,
        optimize_mask
    )
    
    # 5. Execute Optimization.
    if n_params == 0:
        raise ValueError("No parameters to optimize.")

    # L-BFGS-B optimization.
    # We define a wrapper function to pass all arguments via kwargs, avoiding positional argument issues.
    def objective_wrapper(scalings: npt.NDArray[np.float64]) -> float:
        return _objective_fbz_model(
            group_scaling_params=scalings,
            n_groups=n_groups,
            group_optimize_mask=group_optimize_mask,
            band_group_labels=band_group_labels,
            initial_freqs_q_THz=initial_freqs_q_THz,
            modes=modes,
            temperature_K=temperature_K,
            target_u_flat=target_u_flat,
            valid_atoms_mask=valid_atoms_mask
        )
    
    # Bound scalings to be positive.
    bounds = [(1e-6, None) for _ in range(n_params)]

    res = minimize(
        objective_wrapper,
        initial_group_scalings,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations, 'gtol': 1e-8}
    )
    
    final_group_params = res.x
    final_freqs_q_THz, final_band_scalings = _apply_scaling(
        group_scaling_params=final_group_params,
        n_groups=n_groups,
        group_optimize_mask=group_optimize_mask,
        band_group_labels=band_group_labels,
        initial_freqs_q_THz=initial_freqs_q_THz
    )
    
    success = res.success
    message = res.message
    residual = res.fun

    # Compute final ADPs.
    modes.frequencies = final_freqs_q_THz * euphonic.ureg.THz
    dw = modes.calculate_debye_waller(temperature=euphonic.ureg.Quantity(temperature_K, "K"))
    final_u_3x3 = 2.0 * dw.debye_waller.to("angstrom**2").magnitude
    final_u_calc = _flatten_u(final_u_3x3)
    
    print_adps_comparison(
        adps_1=u_exp,
        adps_2=initial_u_3x3,
        labels=["reference", "initial", "refined"],
        symbols=modes.crystal.atom_type,
        adps_3=final_u_3x3
    )

    print_frequency_comparison(
        freqs_initial_THz=initial_freqs_q_THz[gamma_idx],
        freqs_refined_THz=final_freqs_q_THz[gamma_idx],
        optimize_mask=optimize_mask
    )

    return {
        "success": success,
        "original_frequencies": initial_freqs_q_THz,
        "refined_frequencies": final_freqs_q_THz,
        "scalings": final_band_scalings,
        "message": message,
        "residual": residual,
        "u_calc": final_u_calc
    }


def _self_fit(
    force_constants: ForceConstants,
    temperature_K: float,
    mesh_size_large: npt.NDArray[np.int64] | float,
    mesh_size_small: npt.NDArray[np.int64] | float,
    max_optimized_freq_THz: float | None = None,
    degeneracy_tolerance: float | None = None,
) -> Dict[str, Any]:
    """
    Fit frequencies of a small k-point mesh model to match reference ADPs.

    1. Calculate ADPs using `mesh_size_large`.
    2. Optimize `mesh_size_small` model to match reference ADPs.
    3. Print comparison.

    Args:
        force_constants: Harmonic force constants model.
        temperature_K: Temperature in Kelvin.
        mesh_size_large: Reference k-point mesh.
        mesh_size_small: Model k-point mesh.
        max_optimized_freq_THz: Maximum frequency (THz) for optimization.

    Returns:
        Result dictionary.
    """
    print(f"--- _self_fit_fbz: Small Mesh fitting to Large Mesh ADPs ---")
    print(f"Temperature: {temperature_K} K")
    print(f"Reference mesh (large): {mesh_size_large}")
    print(f"Model mesh (small):     {mesh_size_small}")
    if max_optimized_freq_THz is not None:
        print(f"Max optimized frequency: {max_optimized_freq_THz} THz")

    # 1. Calculate Reference ADPs using Euphonic with the large mesh
    # We use to_euphonic_modes to get the modes object
    modes_ref = to_euphonic_modes(
        force_constants=force_constants, 
        mesh_size=mesh_size_large
    )
    dw_ref = modes_ref.calculate_debye_waller(
        temperature=euphonic.ureg.Quantity(temperature_K, "K")
    )
    u_ref_cart = 2.0 * dw_ref.debye_waller.to("angstrom**2").magnitude
    
    # Calculate optimize_mask if max_optimized_freq_THz is set
    optimize_mask = None
    if max_optimized_freq_THz is not None:
        ph = mbe_automation.storage.to_phonopy(force_constants)
        # Calculate Gamma frequencies
        freqs_gamma, _ = at_k_points(
            dynamical_matrix=ph.dynamical_matrix,
            k_points=[[0.0, 0.0, 0.0]],
            compute_eigenvecs=False
        )
        # freqs_gamma is (1, n_bands)
        freqs_gamma_flat = freqs_gamma[0]
        
        optimize_mask = freqs_gamma_flat <= max_optimized_freq_THz
    
    # 2. Run Optimization using simple fit_fbz_model
    # We pass the calculated u_ref_cart as the "experimental" target.
    # We use mesh_size_small for the fitting model.
    result = fit_fbz_model(
        force_constants=force_constants,
        u_exp=u_ref_cart,
        temperature_K=temperature_K,
        mesh_size=mesh_size_small,
        optimize_mask=optimize_mask,
        degeneracy_tolerance=degeneracy_tolerance,
    )

    return result
