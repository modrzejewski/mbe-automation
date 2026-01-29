import numpy as np
import numpy.typing as npt
from typing import Literal, Tuple, Dict, Any, List
from scipy.optimize import minimize # type: ignore


from .euphonic import to_euphonic_modes
from mbe_automation.dynamics.harmonic.display import compare_adps
from mbe_automation.dynamics.harmonic.modes import (
    at_k_points
)

from mbe_automation.storage.core import ForceConstants
import mbe_automation.storage

import euphonic

def _group_degenerate_bands(
    freqs_THz: npt.NDArray[np.float64],
    tolerance_THz: float
) -> Tuple[npt.NDArray[np.int64], int]:
    """
    Group bands that are degenerate.
    
    Sorts freqs_THz internally.
    Uses numpy.diff and numpy.split to find gaps larger than tolerance_THz.
    
    Returns:
        - band_to_group_map: (n_bands,) integers group index for each band.
        - n_groups: Total number of unique groups.
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
    Flatten (N, 3, 3) ADPs to unique components (N * 6).
    Order: xx, yy, zz, xy, xz, yz
    """
    flat = []
    for i in range(len(u)):
         flat.extend([
            u[i, 0, 0], u[i, 1, 1], u[i, 2, 2],
            u[i, 0, 1], u[i, 0, 2], u[i, 1, 2]
         ])
    return np.array(flat)


def _apply_shifts(
    group_shift_params: npt.NDArray[np.float64],
    n_groups: int,
    group_optimize_mask: npt.NDArray[np.bool_],
    band_group_labels: npt.NDArray[np.int64],
    initial_freqs_q_THz: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate shifted frequencies and band shifts from optimization parameters.
    """
    # Map optimization parameters to full group shifts.
    full_group_shifts = np.zeros(n_groups, dtype=np.float64)
    full_group_shifts[group_optimize_mask] = group_shift_params
    
    # Broadcast group shifts to individual bands.
    full_band_shifts = full_group_shifts[band_group_labels]
    
    # Apply shifts to all q-points (broadcasting).
    current_freqs_THz = initial_freqs_q_THz + full_band_shifts
    
    return current_freqs_THz, full_band_shifts


def _objective_fbz_model(
    group_shift_params: npt.NDArray[np.float64],
    n_groups: int,
    group_optimize_mask: npt.NDArray[np.bool_],
    band_group_labels: npt.NDArray[np.int64],
    initial_freqs_q_THz: npt.NDArray[np.float64],
    modes: "euphonic.QpointPhononModes",
    temperature_K: float,
    target_u_flat: npt.NDArray[np.float64],
) -> float:
    """
    Objective function for fit_fbz_model.
    """
    current_freqs_THz, _ = _apply_shifts(
        group_shift_params,
        n_groups,
        group_optimize_mask,
        band_group_labels,
        initial_freqs_q_THz
    )
    
    # Update Euphonic frequencies.
    modes.frequencies = current_freqs_THz * euphonic.ureg.THz
    
    # 4. Calculate ADPs (U = 2 * W).
    dw = modes.calculate_debye_waller(
        temperature=euphonic.ureg.Quantity(temperature_K, "K")
    )
    u_cart_3x3 = 2.0 * dw.debye_waller.to("angstrom**2").magnitude
    
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
    degeneracy_tolerance: float = 1e-4,
    max_iterations: int = 200,
) -> Dict[str, Any]:
    """
    Fit phonon frequencies to match experimental ADPs using a Full Brillouin Zone model.

    This function:
    1.  Generates a k-point mesh with ODD numbers of points (ensuring Gamma point inclusion).
    2.  Uses Euphonic to calculate frequencies and eigenvectors.
    3.  Reorders frequencies using Euphonic's ``reorder_frequencies``.
    4.  Identifies band degeneracies at the Gamma point.
    5.  Optimizes frequency shifts (one parameter per degenerate band group).
        Acoustic bands (lowest 3 at Gamma) are EXCLUDED from optimization.

    Args:
        force_constants: The force constants object.
        u_exp: (N_atoms, 3, 3) Experimental Cartesian ADPs in Å².
        temperature_K: Temperature in Kelvin.
        mesh_size: k-point mesh dimensions (should be array-like of 3 integers).
                   Will be forced to be odd if not already.
        optimize_mask: (n_bands,) Boolean mask. If True, the band is allowed to shift.
                       If provided, it MUST correspond to sorted bands at Gamma.
                       Acoustic modes (indices 0, 1, 2) MUST be False.
        degeneracy_tolerance: Tolerance (THz) for linking degenerate bands at Gamma.
        max_iterations: Maximum iterations for the optimizer.

    Returns:
        Dictionary containing optimization results.
    """
    # 1. Initialize Euphonic object with odd-numbered mesh to ensure Gamma point inclusion.
    modes = to_euphonic_modes(
        force_constants=force_constants, 
        mesh_size=mesh_size, 
        odd_numbers=True
    )
    
    # 2. Reorder frequencies to ensure consistent band tracing.
    modes.reorder_frequencies(reorder_gamma=True)
    
    # Extract initial frequencies in THz.
    initial_freqs_q_THz = modes.frequencies.to("THz").magnitude
    n_q, n_bands = initial_freqs_q_THz.shape

    # Compute initial ADPs for comparison.
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
    
    # Determine degenerate groups based on Gamma frequencies.
    band_group_labels, n_groups = _group_degenerate_bands(freqs_gamma, degeneracy_tolerance)
    
    # 4. Configure optimization mask.
    if optimize_mask is None:
        # Default: Optimize all non-acoustic bands.
        # Assume sorted bands at Gamma: indices 0, 1, 2 are acoustic.
        optimize_mask = np.ones(n_bands, dtype=bool)
        optimize_mask[:3] = False
    
    # Validation: Acoustic modes must be excluded from optimization.
    if np.any(optimize_mask[:3]):
        raise ValueError("Optimization of acoustic modes (indices 0, 1, 2) is not supported.")

    if len(optimize_mask) != n_bands:
        raise ValueError(f"optimize_mask length {len(optimize_mask)} must match number of bands {n_bands}")

    # Map bands to parameter groups.
    group_optimize_mask = np.zeros(n_groups, dtype=bool)
    for band_idx, group_idx in enumerate(band_group_labels):
        if optimize_mask[band_idx]:
            group_optimize_mask[group_idx] = True
            
    n_params = np.sum(group_optimize_mask)
    initial_group_shifts = np.zeros(n_params, dtype=np.float64)
    
    target_u_flat = _flatten_u(u_exp)
    
    # 5. Execute Optimization.
    if n_params == 0:
        raise ValueError("No parameters to optimize.")

    # L-BFGS-B optimization without explicit bounds.
    res = minimize(
        _objective_fbz_model,
        initial_group_shifts,
        args=(
            n_groups,
            group_optimize_mask,
            band_group_labels,
            initial_freqs_q_THz,
            modes,
            temperature_K,
            target_u_flat,
        ),
        method='L-BFGS-B',
        options={'maxiter': max_iterations, 'gtol': 1e-8}
    )
    
    final_group_params = res.x
    final_freqs_q_THz, final_band_shifts = _apply_shifts(
        final_group_params,
        n_groups,
        group_optimize_mask,
        band_group_labels,
        initial_freqs_q_THz
    )
    
    success = res.success
    message = res.message
    residual = res.fun

    # Compute final ADPs.
    modes.frequencies = final_freqs_q_THz * euphonic.ureg.THz
    dw = modes.calculate_debye_waller(temperature=euphonic.ureg.Quantity(temperature_K, "K"))
    final_u_3x3 = 2.0 * dw.debye_waller.to("angstrom**2").magnitude
    final_u_calc = _flatten_u(final_u_3x3)
    
    compare_adps(
        adps_1=u_exp,
        adps_2=initial_u_3x3,
        labels=["reference", "initial", "refined"],
        symbols=modes.crystal.atom_type,
        adps_3=final_u_3x3
    )

    return {
        "success": success,
        "original_frequencies": initial_freqs_q_THz,
        "refined_frequencies": final_freqs_q_THz,
        "shifts": final_band_shifts,
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
) -> Dict[str, Any]:
    """
    Fit frequencies of a small k-point mesh model to match ADPs from a large reference mesh.

    This function:
    1. Computes ADPs using the `mesh_size_large` (reference).
    2. Runs `fit_fbz_model` using `mesh_size_small` to optimize frequencies such that the
       resulting ADPs match the reference ADPs.
    3. Prints a comparison of the ADPs.

    Args:
        force_constants: The harmonic force constants model.
        temperature_K: Temperature in Kelvin.
        mesh_size_large: The k-point mesh for the reference calculation (e.g. [10, 10, 10] or float density).
        mesh_size_small: The k-point mesh for the model to be optimized (e.g. [3, 3, 3] or float density).
                         Ideally should be odd-numbered to include Gamma.
        max_optimized_freq_THz: Maximum frequency (in THz) for modes to be optimized.
                                Modes above this frequency will be fixed.

    Returns:
        Result dictionary from `fit_fbz_model`.
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
        
        # Explicitly disable acoustic modes (assumed to be first 3)
        if len(optimize_mask) >= 3:
            optimize_mask[:3] = False
    
    # 2. Run Optimization using simple fit_fbz_model
    # We pass the calculated u_ref_cart as the "experimental" target.
    # We use mesh_size_small for the fitting model.
    result = fit_fbz_model(
        force_constants=force_constants,
        u_exp=u_ref_cart,
        temperature_K=temperature_K,
        mesh_size=mesh_size_small,
        optimize_mask=optimize_mask,
        # Default parameters for tolerance, etc.
    )

    return result
