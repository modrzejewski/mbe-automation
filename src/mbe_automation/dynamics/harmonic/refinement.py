"""
Integration module for normal mode refinement using
the NoMoRe library of Paul Niklas Ruth
https://github.com/Niolon
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Literal, List
import phonopy.physical_units
import mbe_automation.dynamics.harmonic.modes
import mbe_automation.common.display
from mbe_automation.dynamics.harmonic.display import print_frequency_comparison, print_adps_comparison
from phonopy.structure.atoms import symbol_map

try:
    import cctbx  # noqa: F401
    from nomore_ase.crystallography.cctbx_adapter import CctbxAdapter
    from nomore_ase.crystallography.smtbx_adapter import SmtbxAdapter
    from nomore_ase.core.calculator import NoMoReCalculator
    from nomore_ase.core.phonon_data import PhononData
    from nomore_ase.optimization.engine import RefinementEngine
    from nomore_ase.analysis.validation import extract_adps_from_structure
    from nomore_ase.analysis.s12_similarity import calculate_s12_per_atom
    from nomore_ase.utils.geometry import build_atom_mapping_periodic

    from mbe_automation.dynamics.harmonic.bands import (
        track_from_gamma,
        determine_degenerate_bands,
        reorder,
        DEFAULT_Q_SPACING,
        DEFAULT_DEGENERATE_FREQS_TOL
    )
    _NOMORE_AVAILABLE = True
except ImportError:
    _NOMORE_AVAILABLE = False
    from mbe_automation.dynamics.harmonic.bands import DEFAULT_Q_SPACING, DEFAULT_DEGENERATE_FREQS_TOL


def _select_refined_freqs(phonon_data: Any, n_refined: int) -> Any:
    """
    Build refinement groups using FixedThresholdStrategy, determining the
    high_limit frequency so that modes beyond the n_refined lowest-frequency 
    groups are strictly fixed rather than falling into a shared MFSF (MEDIUM) tier.
    """
    from nomore_ase.core.frequency_partition import (
        FixedThresholdStrategy, create_pre_groups
    )
    pre_groups = create_pre_groups(phonon_data)
    
    if pre_groups is None:
        # If no degeneracy or band grouping is needed, every mode is its own group.
        pre_groups = [[i] for i in range(len(phonon_data.frequencies_cm1))]

    if n_refined >= len(pre_groups):
        high_limit_val = np.inf
    else:
        # Sort by mean group frequency (like FixedThresholdStrategy does internally)
        group_means = [(np.mean(phonon_data.frequencies_cm1[g]), g) for g in pre_groups]
        group_means.sort(key=lambda x: x[0])
        
        remaining_groups = group_means[n_refined:]
        min_remaining_freq = min(np.min(phonon_data.frequencies_cm1[g]) for _, g in remaining_groups)
        high_limit_val = min_remaining_freq - 1e-4

    strategy = FixedThresholdStrategy(
        n_refined=n_refined,
        high_limit=max(0.0, high_limit_val)
    )
    return strategy.compute_groups(phonon_data, pre_groups)


@dataclass
class NormalModeRefinement:
    """
    Result of normal mode refinement against experimental ADPs.
    
    All U_cart arrays are in P1 (full primitive cell) with shape (n_atoms, 3, 3).
    Use asu_atoms to extract asymmetric unit quantities.
    """
    n_bands: int
    n_q_points: int
    irr_q_frac: npt.NDArray[np.float64]
    q_weights: npt.NDArray[np.float64]
    freqs_initial_reordered_THz: npt.NDArray[np.float64] # Shape (n_q, n_bands) reordered by band tracking
    freqs_final_reordered_THz: npt.NDArray[np.float64]   # Shape (n_q, n_bands) reordered by band tracking
    U_cart_exp_Angs2: npt.NDArray[np.float64]
    U_cart_comp_initial_Angs2: npt.NDArray[np.float64]
    U_cart_comp_final_Angs2: npt.NDArray[np.float64]
    asu_atoms: npt.NDArray[np.int64]
    s12_initial: npt.NDArray[np.float64]
    s12_final: npt.NDArray[np.float64]
    U_diff_squared_initial: npt.NDArray[np.float64]
    U_diff_squared_final: npt.NDArray[np.float64]
    band_scaling_factors: npt.NDArray[np.float64]
    optimized_bands: npt.NDArray[np.bool_]

def _band_scaling_factors(
    scale_factors: npt.NDArray[np.float64],
    groups: "mbe_automation.dynamics.harmonic.refinement.RefinementGroups",
    n_q: int,
    n_bands: int,
    gamma_idx: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Map group scaling factors to bands using Gamma-point group IDs."""
    gamma_group_ids = groups.group_ids.reshape(n_q, n_bands)[gamma_idx]
    band_scaling_factors = np.ones(n_bands)
    valid_mask = gamma_group_ids >= 0  # Skip fixed modes (ID -1)

    # nomore_ase RefinementEngine maps params to sorted unique group IDs >= 0
    unique_gids = np.sort(np.unique(groups.group_ids[groups.group_ids >= 0]))
    gid_to_param = {gid: i for i, gid in enumerate(unique_gids)}
    param_indices = np.array([gid_to_param[gid] for gid in gamma_group_ids[valid_mask]])
    band_scaling_factors[valid_mask] = scale_factors[param_indices]

    return band_scaling_factors, valid_mask


def _validate_scaling_factors(
    band_scaling_factors: npt.NDArray[np.float64],
    freqs_initial_reordered_cm1: npt.NDArray[np.float64],
    freqs_final_reordered_cm1: npt.NDArray[np.float64],
    atol: float = 1e-3
) -> None:
    """Verify that final frequencies are recovered from initial frequencies and scaling factors."""
    freqs_recovered = freqs_initial_reordered_cm1 * band_scaling_factors
    if not np.allclose(freqs_final_reordered_cm1, freqs_recovered, atol=atol):
        diff = np.abs(freqs_final_reordered_cm1 - freqs_recovered)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        raise ValueError(
            f"Final frequency recovery failed for band {max_diff_idx[1]} at q-point {max_diff_idx[0]}. "
            f"Expected {freqs_final_reordered_cm1[max_diff_idx]:.6f}, "
            f"recovered {freqs_recovered[max_diff_idx]:.6f}."
        )

def _validate_reasonable_range(
    band_scaling_factors: npt.NDArray[np.float64],
    reasonable_range: tuple[float, float]
) -> None:
    """Verify that all optimized scaling factors are within the reasonable range."""
    lower, upper = reasonable_range
    if np.any((band_scaling_factors < lower) | (band_scaling_factors > upper)):
        outliers = band_scaling_factors[(band_scaling_factors < lower) | (band_scaling_factors > upper)]
        raise ValueError(
            f"Optimized scaling factors hit boundaries of reasonable range: {outliers}. "
            f"Expected all to be within {reasonable_range}."
        )

def to_phonon_data(
    phonopy_object,
    irr_q_frac: npt.NDArray[np.floating],
    q_weights: npt.NDArray[np.integer],
    cif_adapter: Optional["CctbxAdapter"] = None,
    q_spacing: float = DEFAULT_Q_SPACING,
    degenerate_freqs_tol_cm1: float = DEFAULT_DEGENERATE_FREQS_TOL,
    symmetry_tolerance: float | None = None,
    symmetrize_Dq: bool = True,
    symprec: float = 1e-5,
) -> PhononData:
    """
    Create PhononData object from computed phonopy data, matching atoms to CIF.

    Args:
        phonopy_object: The phonopy object with initialized mesh.
        irr_q_frac: Irreducible q-points (fractional).
        q_weights: Weights of irreducible q-points.
        cif_adapter: Adapter containing the experimental structure (CIF). 
            If None, identity permutation is assumed.
        q_spacing: Spacing for path interpolation in Å⁻¹. Used for band tracking.
        degenerate_freqs_tol_cm1: Tolerance for detecting degenerate frequencies in cm⁻¹.
        symmetry_tolerance: Tolerance used by nomore_ase to determine if two 
            normal modes are degenerate. It defines the minimum required inner product 
            (overlap) between their eigenvectors after applying a lattice 
            symmetry operation.

    Returns:
        PhononData object ready for refinement.
    """
    if not _NOMORE_AVAILABLE:
        raise ImportError(
            "The `to_phonon_data` function requires the `nomore_ase` package. "
            "Install it in your environment to use this functionality."
        )

    ph = phonopy_object
    
    freqs_cm1_grid, eigenvectors_grid = mbe_automation.dynamics.harmonic.modes.at_k_points(
        phonopy_object=ph,
        k_points=irr_q_frac,
        compute_eigenvecs=True,
        freq_units="invcm",
        eigenvectors_storage="rows",
        symmetrize_Dq=symmetrize_Dq,
        symprec=symprec,
    )

    flat_freqs_cm1 = freqs_cm1_grid.flatten()
    n_atoms_fc = len(ph.primitive)
    
    if cif_adapter is not None:
        source_to_target_indices = compute_atom_permutation(ph.primitive, cif_adapter)
        reordering_needed = not np.all(source_to_target_indices == np.arange(n_atoms_fc))
        
        if reordering_needed:
            print("Reordering ForceConstants atoms to match CIF order. Permutation found.")
        else:
            print("Atom order matches between ForceConstants and CIF.")
        
        p1_structure = cif_adapter.xray_structure.expand_to_p1()
        target_symbols = [sc.element_symbol() for sc in p1_structure.scatterers()]
        target_frac = np.array([sc.site for sc in p1_structure.scatterers()]) % 1.0
        n_target = len(target_symbols)
        
        _validate_atomic_numbers(
            ph.primitive.numbers,
            target_symbols,
            source_to_target_indices
        )
    else:
        source_to_target_indices = np.arange(n_atoms_fc)
        reordering_needed = False
        target_symbols = ph.primitive.symbols
        target_frac = ph.primitive.scaled_positions % 1.0
        n_target = n_atoms_fc
    
    # Reorder Eigenvectors
    # eigenvectors_grid: (n_q, n_modes, n_modes)
    # n_modes = n_atoms * 3
    # We reshape to (n_q * n_modes, n_atoms, 3) to shuffle atoms
    
    n_modes_total = n_atoms_fc * 3
    flat_eigenvectors = eigenvectors_grid.reshape(-1, n_atoms_fc, 3)
    
    if reordering_needed:

        # Reorder Eigenvectors
        # flat_eigenvectors: (n_q * n_modes, n_atoms, 3) 
        # We need to permute along axis 1 (atoms)
        flat_eigenvectors = _permute_atoms(flat_eigenvectors, source_to_target_indices, axis=1)

        final_positions = target_frac
        final_symbols = target_symbols
        final_masses = _permute_atoms(ph.primitive.masses, source_to_target_indices, axis=0)
        
    else:

        final_positions = ph.primitive.scaled_positions % 1.0
        final_symbols = ph.primitive.symbols
        final_masses = ph.primitive.masses

    print(f"Computing band assignment for {len(irr_q_frac)} q-points...")
    band_indices = track_from_gamma(
        phonopy_object=ph,
        q_points=irr_q_frac,
        q_spacing=q_spacing,
        degenerate_freqs_tol_cm1=degenerate_freqs_tol_cm1,
        symmetrize_Dq=symmetrize_Dq,
        symprec=symprec,
    )
    # band_indices: (n_q, n_modes)
    gamma_index = np.argmin(np.linalg.norm(irr_q_frac, axis=1))
    degeneracy_groups = determine_degenerate_bands(
        ph, 
        band_indices, 
        gamma_index,
        symmetry_tolerance=symmetry_tolerance,
        symmetrize_Dq=symmetrize_Dq,
        symprec=symprec,
    )

    mode_q_indices = np.repeat(np.arange(len(irr_q_frac)), n_modes_total)

    return PhononData(
        frequencies_cm1=flat_freqs_cm1,
        eigenvectors=flat_eigenvectors,
        q_points=irr_q_frac,
        mode_q_indices=mode_q_indices,
        degeneracy_groups=degeneracy_groups,
        positions_frac=final_positions,
        cell=ph.primitive.cell,
        symbols=final_symbols,
        masses=final_masses,
        # Since Phonopy handles the supercell folding/unfolding internally and provides
        # us with frequencies and eigenvectors on a specific q-mesh, the supercell
        # attribute is effectively metadata and can be set to [1, 1, 1].
        supercell=[1, 1, 1],
        n_atoms=n_target,
        band_indices=band_indices.flatten()
    )
    
def _validate_atomic_numbers(
    source_numbers: npt.NDArray[np.int64],
    target_symbols: List[str],
    source_to_target_indices: npt.NDArray[np.int64]
) -> None:
    """
    Validate that permuted source atomic numbers match target elements.
    
    Args:
        source_numbers: Atomic numbers from source (e.g. Phonopy primitive).
        target_symbols: Element symbols from target (e.g. CIF).
        source_to_target_indices: Permutation mapping source -> target.

    Raises:
        ValueError: If atomic numbers do not match after permutation.
    """
    try:
         target_numbers = np.array([symbol_map[s] for s in target_symbols])
    except KeyError as e:
         raise ValueError(f"Unknown element symbol '{e.args[0]}' in target structure cannot be validated against atomic numbers.") from e

    permuted_source_numbers = _permute_atoms(source_numbers, source_to_target_indices, axis=0)
    
    if not np.array_equal(permuted_source_numbers, target_numbers):
         raise ValueError(
             f"Atomic number mismatch after permutation!\n"
             f"Source (permuted): {permuted_source_numbers}\n"
             f"Target: {target_numbers}"
         )


def _permute_atoms(
    data: np.ndarray, 
    source_to_target_indices: np.ndarray, 
    axis: int = 0
) -> np.ndarray:
    """
    Permute atom-related indices in an array according to a mapping.
    
    The mapping array maps source indices to target indices:
        source_to_target_indices[source_index] = target_index
        
    This function returns a new array where:
        new_data[target_index] = old_data[source_index]
        
    Args:
        data: Array to permute.
        source_to_target_indices: 1D array where value at [i] is the target index for atom i.
        axis: The axis corresponding to the atom index.
        
    Returns:
        Reordered array.
    """
    n_source = len(source_to_target_indices)
    if data.shape[axis] != n_source:
        raise ValueError(f"Data shape {data.shape} incompatible with permutation length {n_source} on axis {axis}")

    new_data = np.empty_like(data)
    indexer = [slice(None)] * data.ndim
    indexer[axis] = source_to_target_indices
    new_data[tuple(indexer)] = data
    return new_data

def _assert_equal_cells(fc_cell_matrix: np.ndarray, cif_unit_cell: Any, tolerance: float = 1e-3) -> None:
    """
    Assert that ForceConstants and CIF unit cells match in basis vectors.
    
    Args:
        fc_cell_matrix: 3x3 matrix from Phonopy (row vectors).
        cif_unit_cell: cctbx unit_cell object.
        tolerance: Tolerance for parameter comparison (degrees/Angstrom).
    """
    # Get orthogonalization matrix from cctbx (column vectors)
    # Transpose to match Phonopy's row vector convention
    cif_matrix = np.array(cif_unit_cell.orthogonalization_matrix()).reshape(3, 3).T
    
    if not np.allclose(fc_cell_matrix, cif_matrix, atol=tolerance):
        raise ValueError(
            f"Unit cell mismatch between ForceConstants and CIF!\n"
            f"FC (rows):\n{fc_cell_matrix}\n"
            f"CIF (ortho.T):\n{cif_matrix}\n"
            f"Tolerance: {tolerance}"
        )

def compute_atom_permutation(phonopy_primitive: Any, cif_adapter: "CctbxAdapter") -> npt.NDArray[np.int64]:
    """
    Compute permutation array to map ForceConstants atoms to CIF atoms.
    
    Args:
        phonopy_primitive: Primitive cell object from Phonopy.
        cif_adapter: Adapter for experimental structure.
        
    Returns:
        permutation: Array of indices such that permutation[i_fc] = i_target.
    """
    # sites_mod_positive=True wraps generated fractional coordinates to the [0, 1) range
    p1_structure = cif_adapter.xray_structure.expand_to_p1(sites_mod_positive=True)
    uc = p1_structure.unit_cell()
    n_target = p1_structure.scatterers().size()
    
    n_atoms_fc = len(phonopy_primitive)
    fc_cell = phonopy_primitive.cell
    
    if n_atoms_fc != n_target:
        raise ValueError(f"Atom count mismatch: FC has {n_atoms_fc}, CIF has {n_target}")

    _assert_equal_cells(fc_cell, uc)

    target_frac = np.array([sc.site for sc in p1_structure.scatterers()])
    target_symbols = np.array([sc.element_symbol() for sc in p1_structure.scatterers()])
    
    fc_frac = phonopy_primitive.scaled_positions
    fc_symbols = np.array(phonopy_primitive.symbols)
    
    indices, distances = build_atom_mapping_periodic(
        source_positions=fc_frac,
        target_positions=target_frac,
        unit_cell=uc,
        tolerance=2.0,
        source_elements=fc_symbols,
        target_elements=target_symbols
    )

    return indices

def _get_atom_mask(symbols: list[str], exclude_hydrogen: bool) -> npt.NDArray[np.bool_]:
    """Return a boolean mask array excluding hydrogen atoms if requested."""
    n = len(symbols)
    if exclude_hydrogen:
        return np.array([s.upper() != "H" for s in symbols], dtype=bool)
    return np.ones(n, dtype=bool)

def _s12_per_atom(
    u_comp: npt.NDArray[np.float64],
    u_exp: npt.NDArray[np.float64],
    symbols: list[str],
    exclude_hydrogen: bool
) -> npt.NDArray[np.float64]:
    """
    Compute per-atom S12 similarity, with NaN for excluded atoms.

    Args:
        u_comp: Computed ADPs (n_atoms, 3, 3).
        u_exp: Experimental ADPs (n_atoms, 3, 3).
        symbols: Element symbols for each atom.
        exclude_hydrogen: If True, set S12 to NaN for H atoms.

    Returns:
        Per-atom S12 array (n_atoms,). H atoms are NaN if excluded.
    """
    s12 = np.full(len(symbols), np.nan)
    mask = _get_atom_mask(symbols, exclude_hydrogen)
    if np.any(mask):
        vals, _ = calculate_s12_per_atom(u_comp[mask], u_exp[mask])
        s12[mask] = vals
    return s12


def _u_diff_squared_per_atom(
    u_comp: npt.NDArray[np.float64],
    u_exp: npt.NDArray[np.float64],
    symbols: list[str],
    exclude_hydrogen: bool
) -> npt.NDArray[np.float64]:
    """
    Compute per-atom squared Euclidean distance between U matrices.

    The squared distance for atom i is Σ_jk (U_comp_ijk - U_exp_ijk)².
    Returns NaN for excluded atoms.

    Args:
        u_comp: Computed ADPs (n_atoms, 3, 3).
        u_exp: Experimental ADPs (n_atoms, 3, 3).
        symbols: Element symbols for each atom.
        exclude_hydrogen: If True, set result to NaN for H atoms.

    Returns:
        Per-atom squared distance array (n_atoms,).
    """
    u_diff_sq = np.full(len(symbols), np.nan)
    mask = _get_atom_mask(symbols, exclude_hydrogen)
    if np.any(mask):
        diff = u_comp[mask] - u_exp[mask]
        u_diff_sq[mask] = np.sum(diff**2, axis=(1, 2))
    return u_diff_sq


def _clamp_acoustic_frequencies(
    frequencies_cm1: npt.NDArray[np.float64],
    min_freq_cm1: float = 10.0
) -> npt.NDArray[np.float64]:
    """
    Clamp low/imaginary frequencies across all q-points.
    
    Args:
        frequencies_cm1: Flat array of frequencies in cm⁻¹ (n_q * n_modes).
        min_freq_cm1: Minimum frequency threshold in cm⁻¹.
        
    Returns:
        Array with clamped frequencies.
    """
    n_low = np.sum(frequencies_cm1 < min_freq_cm1)
    if n_low > 0:
        print(f"  Clamping {n_low} modes below {min_freq_cm1} cm⁻¹")
        return np.where(frequencies_cm1 < min_freq_cm1, min_freq_cm1, frequencies_cm1)
    return frequencies_cm1

def _display_refinement_summary(
    refinement: NormalModeRefinement,
    asu_symbols: list[str] | None = None,
    exclude_hydrogen: bool = False
) -> None:
    """
    Display refinement summary: frequency comparison and ADP comparison.
    
    Args:
        refinement: NormalModeRefinement result object.
        asu_symbols: Element symbols for ASU atoms.
        exclude_hydrogen: If True, exclude H atoms from ADP display.
    """
    THz_to_cm1 = phonopy.physical_units.get_physical_units().THzToCm
    initial_freqs_cm1 = refinement.freqs_initial_reordered_THz * THz_to_cm1
    refined_freqs_cm1 = refinement.freqs_final_reordered_THz * THz_to_cm1
    
    gamma_idx = np.argmin(np.linalg.norm(refinement.irr_q_frac, axis=1))
    
    initial_band_avg_cm1 = np.average(
        initial_freqs_cm1, 
        axis=0, 
        weights=refinement.q_weights
    )
    refined_band_avg_cm1 = np.average(
        refined_freqs_cm1, 
        axis=0, 
        weights=refinement.q_weights
    )
    
    print_frequency_comparison(
        freqs_initial_gamma=initial_freqs_cm1[gamma_idx],
        freqs_refined_gamma=refined_freqs_cm1[gamma_idx],
        freqs_initial_avg=initial_band_avg_cm1,
        freqs_refined_avg=refined_band_avg_cm1,
        scaling_factors=refinement.band_scaling_factors,
        optimize_mask=refinement.optimized_bands,
        unit="cm1"
    )

    print_adps_comparison(
        adps_1=refinement.U_cart_exp_Angs2[refinement.asu_atoms],
        adps_2=refinement.U_cart_comp_initial_Angs2[refinement.asu_atoms],
        labels=["experimental", "initial", "refined"],
        symbols=asu_symbols,
        adps_3=refinement.U_cart_comp_final_Angs2[refinement.asu_atoms],
        s12_12=np.nanmean(refinement.s12_initial),
        s12_13=np.nanmean(refinement.s12_final),
        rmsd_12=np.sqrt(np.nanmean(refinement.U_diff_squared_initial)),
        rmsd_13=np.sqrt(np.nanmean(refinement.U_diff_squared_final)),
        exclude_hydrogen=exclude_hydrogen
    )


def _get_asu_atoms(smtbx_adapter) -> npt.NDArray[np.int64]:
    """
    Get P1 indices of atoms that map to ASU via identity transformation.

    For each ASU atom, find the P1 atom that maps to it via the identity
    rotation matrix (i.e., the ASU atom itself in the P1 expansion).

    Args:
        smtbx_adapter: SmtbxAdapter instance containing p1_to_asu mapping.

    Returns:
        np.ndarray: Indices of ASU atoms in P1 (one per ASU atom).

    Raises:
        RuntimeError: If any ASU atom lacks an identity-mapped representative.
    """
    n_asu = smtbx_adapter.n_asu_atoms
    indices = np.full(n_asu, -1, dtype=np.int64)

    for p1_idx, (asu_idx, r_cart) in enumerate(smtbx_adapter.p1_to_asu):
        if np.allclose(r_cart, np.eye(3), atol=1e-6):
            indices[asu_idx] = p1_idx

    if np.any(indices == -1):
        missing = np.where(indices == -1)[0]
        raise RuntimeError(
            f"Failed to find identity-mapped P1 atoms for ASU atoms {missing.tolist()}. "
            "This indicates a missing identity operation in the P1 expansion."
        )

    return indices


def _fit_to_adps(
    phonons: Any,
    U_cart_exp_p1: npt.NDArray[np.float64],
    initial_freqs: npt.NDArray[np.float64],
    groups: Any,
    total_q: float
) -> dict[str, Any]:
    """Perform ADP-only fitting with hydrogen exclusion."""
    print("Performing ADP-only fitting")
    print("Hydrogens are excluded from the ADP-only fit and the displayed metrics.")
    non_h_mask = np.array([sym.upper() != 'H' for sym in phonons.symbols])
    u_exp_heavy = U_cart_exp_p1[non_h_mask]

    adp_only_calc = NoMoReCalculator(
        eigenvectors=phonons.eigenvectors[:, non_h_mask, :],
        masses=phonons.masses[non_h_mask],
        temperature=phonons.temperature, 
        normalization_factor=float(total_q),
        degeneracy_groups=phonons.degeneracy_groups
    )
    engine_adp_only = RefinementEngine(
        calculator=adp_only_calc,
        smtbx_adapter=None,
    )

    result = engine_adp_only.fit_to_adps(
        initial_frequencies=initial_freqs,
        u_exp=u_exp_heavy,
        groups=groups,
        smoothness_weight=0.0, # Not exposed in run() yet
    )
    
    # Compute scale factors for compatibility
    final_freqs_flat = result["frequencies"]
    scale_factors = np.ones(groups.n_parameters())
    unique_gids = sorted(np.unique(groups.group_ids[groups.group_ids >= 0]))
    for i, gid in enumerate(unique_gids):
        mode_idx = groups.get_group_modes(gid)[0]
        if initial_freqs[mode_idx] > 1e-3:
            scale_factors[i] = final_freqs_flat[mode_idx] / initial_freqs[mode_idx]
    
    result["scale_factors"] = scale_factors
    result["groups"] = groups
    return result


def _fit_to_intensities(
    calculator: Any,
    smtbx_adapter: Any,
    initial_freqs: npt.NDArray[np.float64],
    groups: Any,
    optimizer_options: dict[str, Any],
    optimizer_method: str,
    fix_positions: bool,
    exclude_hydrogen_positions: bool
) -> dict[str, Any]:
    """Perform joint refinement against X-ray intensities."""
    print("Performing joint refinement against X-ray intensities")
    
    engine = RefinementEngine(calculator, smtbx_adapter)

    return engine.run_joint(
        initial_frequencies=initial_freqs,
        groups=groups,
        optimizer_options=optimizer_options,
        optimizer_method=optimizer_method,
        fix_positions=fix_positions,
        exclude_hydrogen_positions=exclude_hydrogen_positions
    )


def run(
    force_constants,
    cif_path: str | None = None,
    U_cart_ref: npt.NDArray[np.float64] | None = None,
    adp_only_fit: bool = False,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
    n_refined: int | None = None,
    weighting_scheme: Literal["sigma", "unit"] = "sigma",
    fix_positions: bool = True,
    exclude_hydrogen_positions: bool = True,
    temperature_K: float | None = None,
    q_spacing: float = DEFAULT_Q_SPACING,
    reasonable_range: tuple[float, float] | None = None,
    degenerate_freqs_tol_cm1: float = DEFAULT_DEGENERATE_FREQS_TOL,
    symmetry_tolerance: float | None = None,
    symmetrize_Dq: bool = True,
    symprec: float = 1e-5,
) -> NormalModeRefinement:
    """
    Perform normal mode refinement.
    
    Args:
        force_constants: mbe_automation ForceConstants object.
        cif_path: Path to experimental CIF. Optional if U_cart_ref is provided.
        U_cart_ref: Reference Cartesian ADPs (n_atoms, 3, 3) for direct fitting.
        mesh_size: k-point mesh size.
        n_refined: Number of lowest-frequency groups to optimize individually.
        weighting_scheme: Weighting scheme for refinement ('sigma' or 'unit').
        q_spacing: Spacing for path interpolation in Å⁻¹ along q-point paths.
        reasonable_range: Allowed range for optimized scaling factors.
        degenerate_freqs_tol_cm1: Tolerance for detecting degenerate frequencies in cm⁻¹.
        symmetry_tolerance: Tolerance used by nomore_ase to determine if two 
            normal modes are degenerate. It defines the minimum required inner product 
            (overlap) between their eigenvectors after applying a lattice 
            symmetry operation.
        
    Returns:
        NormalModeRefinement object with frequencies, ADPs, and mesh data.
    """
    if not _NOMORE_AVAILABLE:
        raise ImportError(
            "The `refinement.run` function requires the `nomore_ase` package. "
            "Install it in your environment to use this functionality."
        )

    if cif_path is None and U_cart_ref is None:
        raise ValueError("Either `cif_path` or `U_cart_ref` must be provided.")

    if cif_path is not None and U_cart_ref is not None:
        raise ValueError("Only one of `cif_path` or `U_cart_ref` can be provided.") 

    if U_cart_ref is not None and temperature_K is None:
        raise ValueError("`temperature_K` must be provided when `U_cart_ref` is provided.")

    optimizer_options = {"maxiter": 300, "ftol": 1e-9}
    optimizer_method = "SLSQP"
    use_irreducible_fbz = False

    if n_refined is None:
        n_refined = 6

    mbe_automation.common.display.framed([
        "Normal mode refinement",
    ])
    print("Using interface to the nomore library of Paul Niklas Ruth")
    print("https://github.com/Niolon")
    print(f"mesh_size                {mesh_size}")
    print(f"n_refined                {n_refined}")
    print(f"max_iter                 {optimizer_options['maxiter']}")
    print(f"optimizer_method         {optimizer_method}")
    print(f"weighting_scheme         {weighting_scheme}")
    print(f"fix_positions            {fix_positions}")
    print(f"exclude_hydrogen         {exclude_hydrogen_positions}")
    print(f"use_irreducible_fbz      {use_irreducible_fbz}")
    print(f"temperature_K            {temperature_K if temperature_K is not None else 'from CIF'}")
    print(f"degenerate_freqs_tol     {degenerate_freqs_tol_cm1} cm⁻¹")
    print(f"symmetry_tolerance       {symmetry_tolerance}")
    print(f"adp_only_fit             {adp_only_fit}")

    if isinstance(mesh_size, (list, tuple, np.ndarray)):
        mesh_size = np.array(mesh_size)

    if isinstance(mesh_size, np.ndarray):
        if np.any(mesh_size % 2 == 0):
            raise ValueError(
                f"Mesh size must consist of odd integers to ensure Gamma point inclusion.\n"
                f"Received: {mesh_size}. Please use odd numbers (e.g., [3, 3, 3])."
            )

    if cif_path is not None:
        print(f"Loading experimental data from {cif_path}")
        cctbx_adapter = CctbxAdapter(cif_path)
        print(f"  Space Group: {cctbx_adapter.space_group_symbol}")
    else:
        cctbx_adapter = None
    
    print("Computing phonon data on mesh...")
    
    ph = force_constants.to_phonopy()
    irr_q_frac, q_weights = mbe_automation.dynamics.harmonic.modes.phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=mesh_size,
        use_symmetry=use_irreducible_fbz,
        odd_numbers=True  # Enforce odd mesh to guarantee Gamma point (0,0,0) is included
    )
    
    phonons = to_phonon_data(
        phonopy_object=ph,
        irr_q_frac=irr_q_frac,
        q_weights=q_weights,
        cif_adapter=cctbx_adapter,
        q_spacing=q_spacing,
        degenerate_freqs_tol_cm1=degenerate_freqs_tol_cm1,
        symmetry_tolerance=symmetry_tolerance,
        symmetrize_Dq=symmetrize_Dq,
        symprec=symprec,
    )
    
    if temperature_K is not None:
        temperature = temperature_K
    elif cctbx_adapter is not None:
        temperature = cctbx_adapter.get_temperature()
    else:
        temperature = None

    if temperature is None:
        raise ValueError(
            "Temperature must be provided either as an argument (`temperature_K`) "
            "or be present in the CIF file metadata."
        )
        
    phonons.temperature = temperature
    
    # Calculate normalization factor (total weight of q-points in BZ)
    total_q = float(np.sum(q_weights))
    
    print(f"  Normalization factor (Total Q): {total_q}")
    print(f"  Temperature: {temperature} K")
    
    if reasonable_range is None:
        if len(irr_q_frac) == 1:
            reasonable_range = (0.1, 10.0)
        else:
            reasonable_range = (0.1, 2.0)

    calculator = NoMoReCalculator(
        eigenvectors=phonons.eigenvectors,
        masses=phonons.masses,
        temperature=phonons.temperature, 
        normalization_factor=float(total_q),
        degeneracy_groups=phonons.degeneracy_groups
    )

    # Initialize SmtbxAdapter if cif_path is provided and not adp_only_fit
    if cctbx_adapter is not None and not adp_only_fit:
        smtbx_adapter = SmtbxAdapter(
            xray_structure=cctbx_adapter.xray_structure,
            reflections=cctbx_adapter.get_reflections(),
            weighting_scheme=weighting_scheme,
            phonon_data=phonons
        )
        print(f"  Reflections: {smtbx_adapter.observations.size()} observations")
    else:
        smtbx_adapter = None

    # Extract experimental ADPs and ASU mappings before engine run
    if cctbx_adapter is not None:
        U_cart_exp_p1 = extract_adps_from_structure(
            cctbx_adapter.xray_structure.expand_to_p1(
                sites_mod_positive=True
            )
        )
        if smtbx_adapter is not None:
            asu_atoms = _get_asu_atoms(smtbx_adapter)
            asu_symbols = [sc.element_symbol() for sc in smtbx_adapter.structure.scatterers()]
        else:
            asu_atoms = np.arange(len(phonons.symbols))
            asu_symbols = phonons.symbols
    else:
        U_cart_exp_p1 = U_cart_ref
        asu_atoms = np.arange(len(phonons.symbols))
        asu_symbols = phonons.symbols
    
    # Create mode groups to handle degeneracies and bands
    groups = _select_refined_freqs(phonons, n_refined)
    
    initial_freqs = phonons.frequencies_cm1
    initial_freqs = _clamp_acoustic_frequencies(initial_freqs)

    is_adp_only = (U_cart_ref is not None) or adp_only_fit

    if is_adp_only:
        result = _fit_to_adps(
            phonons=phonons,
            U_cart_exp_p1=U_cart_exp_p1,
            initial_freqs=initial_freqs,
            groups=groups,
            total_q=total_q
        )
    else:
        result = _fit_to_intensities(
            calculator=calculator,
            smtbx_adapter=smtbx_adapter,
            initial_freqs=initial_freqs,
            groups=groups,
            optimizer_options=optimizer_options,
            optimizer_method=optimizer_method,
            fix_positions=fix_positions,
            exclude_hydrogen_positions=exclude_hydrogen_positions
        )

    n_modes_flat = len(initial_freqs)
    n_q = len(irr_q_frac)
    n_bands = n_modes_flat // n_q
    assert n_bands == force_constants.primitive.n_atoms * 3
    
    freqs_initial_reordered_cm1 = reorder(
        frequencies=initial_freqs.reshape(n_q, n_bands),
        band_indices=phonons.band_indices.reshape(n_q, n_bands)
    )
    freqs_final_reordered_cm1 = reorder(
        frequencies=result["frequencies"].reshape(n_q, n_bands),
        band_indices=phonons.band_indices.reshape(n_q, n_bands)
    )

    gamma_idx = np.argmin(np.linalg.norm(irr_q_frac, axis=1))
    band_scaling_factors, optimized_bands = _band_scaling_factors(
        scale_factors=result["scale_factors"],
        groups=result["groups"],
        n_q=n_q,
        n_bands=n_bands,
        gamma_idx=gamma_idx
    )
    _validate_scaling_factors(
        band_scaling_factors=band_scaling_factors,
        freqs_initial_reordered_cm1=freqs_initial_reordered_cm1,
        freqs_final_reordered_cm1=freqs_final_reordered_cm1
    )

    cm1_to_THz = 1.0 / phonopy.physical_units.get_physical_units().THzToCm
    freqs_initial_reordered_THz = freqs_initial_reordered_cm1 * cm1_to_THz
    freqs_final_reordered_THz = freqs_final_reordered_cm1 * cm1_to_THz
    
    U_cart_comp_initial_p1 = calculator.calculate_u_cart(initial_freqs)
    U_cart_comp_final_p1 = calculator.calculate_u_cart(result["frequencies"])
    
    U_cart_exp_asu = U_cart_exp_p1[asu_atoms]
    U_cart_comp_initial_asu = U_cart_comp_initial_p1[asu_atoms]
    U_cart_comp_final_asu = U_cart_comp_final_p1[asu_atoms]

    exclude_h_metrics = True if is_adp_only else exclude_hydrogen_positions

    refinement = NormalModeRefinement(
        n_bands=n_bands,
        n_q_points=n_q,
        irr_q_frac=irr_q_frac,
        q_weights=q_weights.astype(np.float64),
        freqs_initial_reordered_THz=freqs_initial_reordered_THz,
        freqs_final_reordered_THz=freqs_final_reordered_THz,
        U_cart_exp_Angs2=U_cart_exp_p1,
        U_cart_comp_initial_Angs2=U_cart_comp_initial_p1,
        U_cart_comp_final_Angs2=U_cart_comp_final_p1,
        asu_atoms=asu_atoms,
        band_scaling_factors=band_scaling_factors,
        optimized_bands=optimized_bands,
        s12_initial=_s12_per_atom(
            U_cart_comp_initial_asu, 
            U_cart_exp_asu, 
            asu_symbols, 
            exclude_h_metrics
        ),
        s12_final=_s12_per_atom(
            U_cart_comp_final_asu, 
            U_cart_exp_asu, 
            asu_symbols, 
            exclude_h_metrics
        ),
        U_diff_squared_initial=_u_diff_squared_per_atom(
            U_cart_comp_initial_asu, 
            U_cart_exp_asu, 
            asu_symbols, 
            exclude_h_metrics
        ),
        U_diff_squared_final=_u_diff_squared_per_atom(
            U_cart_comp_final_asu, 
            U_cart_exp_asu, 
            asu_symbols, 
            exclude_h_metrics
        )
    )

    _display_refinement_summary(
        refinement=refinement,
        asu_symbols=asu_symbols,
        exclude_hydrogen=exclude_h_metrics
    )
    #
    # We perform the validation of scaling factors here
    # after the summary so that the user can see
    # which bands are scaled by too much in case an exception
    # is raised.
    #
    _validate_reasonable_range(
        band_scaling_factors=band_scaling_factors,
        reasonable_range=reasonable_range
    )

    return refinement
