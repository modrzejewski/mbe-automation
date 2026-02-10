"""
Integration module for normal mode refinement using
the NoMoRe library of Paul Niklas Ruth
https://github.com/Niolon
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Dict, Any, Optional, Literal, List, TYPE_CHECKING
import phonopy.physical_units
import mbe_automation.dynamics.harmonic.modes
from scipy.spatial.distance import cdist
import mbe_automation.common.display
from mbe_automation.dynamics.harmonic.display import print_frequency_comparison, print_adps_comparison
from phonopy.structure.atoms import symbol_map

if TYPE_CHECKING:
    from mbe_automation.api.classes import ForceConstants
    from nomore_ase.crystallography.cctbx_adapter import CctbxAdapter

try:
    from nomore_ase.crystallography.cctbx_adapter import CctbxAdapter
    from nomore_ase.crystallography.smtbx_adapter import SmtbxAdapter
    from nomore_ase.core.calculator import NoMoReCalculator
    from nomore_ase.core.phonon_data import PhononData
    from nomore_ase.optimization.engine import RefinementEngine
    from nomore_ase.core.frequency_partition import (
        create_pre_groups,
        FrequencyPartitionStrategy,
        SensitivityBasedStrategy
    )
    from nomore_ase.optimization.restraints import BayesianFrequencyRestraint
    from nomore_ase.analysis.validation import extract_adps_from_structure
    from nomore_ase.analysis.s12_similarity import calculate_s12_per_atom

    from mbe_automation.dynamics.harmonic.bands import (
        compute_band_indices,
        determine_degenerate_bands,
        reorder_frequencies
    )
except ImportError:
    raise ImportError(
        "The `dynamics.harmonic.refinement` module requires the `nomore_ase` package. "
        "Install it in your environment to use this functionality."
    )


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
    freqs_initial_reordered_THz: npt.NDArray[np.float64] # Shape (n_q, n_bands) reordered by band tracing
    freqs_final_reordered_THz: npt.NDArray[np.float64]   # Shape (n_q, n_bands) reordered by band tracing
    U_cart_exp_Angs2: npt.NDArray[np.float64]
    U_cart_comp_initial_Angs2: npt.NDArray[np.float64]
    U_cart_comp_final_Angs2: npt.NDArray[np.float64]
    asu_atoms: npt.NDArray[np.int64]
    s12_initial: npt.NDArray[np.float64]
    s12_final: npt.NDArray[np.float64]
    chi_sq_initial: npt.NDArray[np.float64]
    chi_sq_final: npt.NDArray[np.float64]
    band_scaling_factors: npt.NDArray[np.float64]

def _map_group_scales_to_modes(
    scale_factors: npt.NDArray[np.float64],
    groups: "RefinementGroups"
) -> npt.NDArray[np.float64]:
    """Map group scaling factors back to individual modes."""
    group_ids = groups.group_ids
    n_modes = len(group_ids)
    mode_scaling_factors = np.ones(n_modes)
    unique_gids = sorted(np.unique(group_ids[group_ids >= 0]))
    for i, gid in enumerate(unique_gids):
        mode_scaling_factors[group_ids == gid] = scale_factors[i]
    return mode_scaling_factors


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

def to_phonon_data(
    phonopy_object,
    irr_q_frac: npt.NDArray[np.floating],
    q_weights: npt.NDArray[np.integer],
    cif_adapter: "CctbxAdapter"
) -> PhononData:
    """
    Create PhononData object from computed phonopy data, matching atoms to CIF.

    Args:
        phonopy_object: The phonopy object with initialized mesh.
        irr_q_frac: Irreducible q-points (fractional).
        q_weights: Weights of irreducible q-points.
        cif_adapter: Adapter containing the experimental structure (CIF).

    Returns:
        PhononData object ready for refinement.
    """
    ph = phonopy_object
    
    freqs_cm1_grid, eigenvectors_grid = mbe_automation.dynamics.harmonic.modes.at_k_points(
        dynamical_matrix=ph.dynamical_matrix,
        k_points=irr_q_frac,
        compute_eigenvecs=True,
        freq_units="invcm",
        eigenvectors_storage="rows"
    )

    flat_freqs_cm1 = freqs_cm1_grid.flatten()
    n_atoms_fc = len(ph.primitive)
    
    source_to_target_indices = compute_atom_permutation(ph.primitive, cif_adapter)
    
    reordering_needed = not np.all(source_to_target_indices == np.arange(n_atoms_fc))
    
    p1_structure = cif_adapter.xray_structure.expand_to_p1()
    target_symbols = [sc.element_symbol() for sc in p1_structure.scatterers()]
    target_frac = np.array([sc.site for sc in p1_structure.scatterers()]) % 1.0
    n_target = len(target_symbols)
    
    _validate_atomic_numbers(
        ph.primitive.numbers,
        target_symbols,
        source_to_target_indices
    )
    
    # Reorder Eigenvectors
    # eigenvectors_grid: (n_q, n_modes, n_modes)
    # n_modes = n_atoms * 3
    # We reshape to (n_q * n_modes, n_atoms, 3) to shuffle atoms
    
    n_modes_total = n_atoms_fc * 3
    flat_eigenvectors = eigenvectors_grid.reshape(-1, n_atoms_fc, 3)
    
    if reordering_needed:
        print(f"Reordering ForceConstants atoms to match CIF order. Permutation found.")
        
        # Reorder Eigenvectors
        # flat_eigenvectors: (n_q * n_modes, n_atoms, 3) 
        # We need to permute along axis 1 (atoms)
        flat_eigenvectors = _permute_atoms(flat_eigenvectors, source_to_target_indices, axis=1)

        final_positions = target_frac
        final_symbols = target_symbols
        final_masses = _permute_atoms(ph.primitive.masses, source_to_target_indices, axis=0)
        
    else:
        print("Atom order matches between ForceConstants and CIF.")
        final_positions = ph.primitive.scaled_positions % 1.0
        final_symbols = ph.primitive.symbols
        final_masses = ph.primitive.masses

    flat_weights = np.repeat(q_weights, n_modes_total)

    print(f"Computing band assignment for {len(irr_q_frac)} q-points...")
    band_indices = compute_band_indices(
        phonopy_object=ph,
        q_points=irr_q_frac
    )
    # band_indices: (n_q, n_modes)
    gamma_index = np.argmin(np.linalg.norm(irr_q_frac, axis=1))
    degeneracy_groups = determine_degenerate_bands(ph, band_indices, gamma_index)

    mode_q_indices = np.repeat(np.arange(len(irr_q_frac)), n_modes_total)

    return PhononData(
        frequencies_cm1=flat_freqs_cm1,
        eigenvectors=flat_eigenvectors,
        q_points=irr_q_frac,
        mode_q_indices=mode_q_indices,
        weights=flat_weights,
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
    p1_structure = cif_adapter.xray_structure.expand_to_p1()
    uc = p1_structure.unit_cell()
    n_target = p1_structure.scatterers().size()
    
    target_frac = np.array([sc.site for sc in p1_structure.scatterers()]) % 1.0
    target_cart = np.array([uc.orthogonalize(f) for f in target_frac])
    
    n_atoms_fc = len(phonopy_primitive)
    fc_frac = phonopy_primitive.scaled_positions % 1.0
    fc_cell = phonopy_primitive.cell
    
    if n_atoms_fc != n_target:
        raise ValueError(f"Atom count mismatch: FC has {n_atoms_fc}, CIF has {n_target}")

    _assert_equal_cells(fc_cell, uc)
    fc_cart_wrapped = (fc_frac @ fc_cell)
    
    distances = cdist(fc_cart_wrapped, target_cart)
    fc_to_target = distances.argmin(axis=1)
    
    used_targets = set()
    source_to_target_indices = np.zeros(n_atoms_fc, dtype=int)
    
    for i_fc, i_target in enumerate(fc_to_target):
        if i_target in used_targets:
            print(f"WARNING: Duplicate matching for target atom {i_target}. Check if atoms are overlapping or tolerance issue.")
        used_targets.add(i_target)
        source_to_target_indices[i_fc] = i_target

    if len(used_targets) != n_target:
        print(f"WARNING: Not all target atoms matched! Used {len(used_targets)} of {n_target}")

    return source_to_target_indices

def _get_refined_bands_mask(
    groups: "RefinementGroups", 
    band_indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """
    Identify which bands have been refined.
    
    A band is considered refined if ANY of its modes are assigned to a refinement group (group_id >= 0).
    Ideally, refinement strategies should refine either the whole band or none of it.
    
    Args:
        groups: RefinementGroups object from nomore_ase.
        band_indices: Array of band indices for each mode (n_modes,).
        
    Returns:
        Boolean array (n_unique_bands,) where True indicates the band was refined.
    """
    unique_bands = np.unique(band_indices[band_indices >= 0])
    # Assuming bands are 0-indexed and contiguous up to max(band_indices)
    # But unique_bands might not cover all if some are filtered out? 
    # Usually band indices cover 0 to n_bands-1.
    
    n_bands_total = np.max(band_indices) + 1 if len(band_indices) > 0 else 0
    refined_mask = np.zeros(n_bands_total, dtype=bool)
    
    refined_mode_indices = groups.get_refined_modes()
    
    # We can check intersection.
    # For each band, check if any mode in it is in refined_mode_indices.
    # Mask of all refined modes
    mode_is_refined = np.zeros(len(band_indices), dtype=bool)
    mode_is_refined[refined_mode_indices] = True
    
    for band_idx in unique_bands:
        modes_in_band = (band_indices == band_idx)
        # Check consistency: all or nothing
        refined_in_band = mode_is_refined[modes_in_band]
        
        if np.any(refined_in_band) and not np.all(refined_in_band):
            # This violates the "all or nothing" assumption, but we mark as refined anyway 
            # and maybe log a warning if we had a logger.
            pass
            
        if np.any(refined_in_band):
            refined_mask[band_idx] = True
            
    return refined_mask


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
    n = len(symbols)
    s12 = np.full(n, np.nan)
    if exclude_hydrogen:
        mask = np.array([s != "H" for s in symbols])
    else:
        mask = np.ones(n, dtype=bool)
    vals, _ = calculate_s12_per_atom(u_comp[mask], u_exp[mask])
    s12[mask] = vals
    return s12


def _chi_sq_per_atom(
    u_comp: npt.NDArray[np.float64],
    u_exp: npt.NDArray[np.float64],
    symbols: list[str],
    exclude_hydrogen: bool
) -> npt.NDArray[np.float64]:
    """
    Compute per-atom χ², with NaN for excluded atoms.

    χ²_i = Σ_jk (U_comp_ijk - U_exp_ijk)²

    Args:
        u_comp: Computed ADPs (n_atoms, 3, 3).
        u_exp: Experimental ADPs (n_atoms, 3, 3).
        symbols: Element symbols for each atom.
        exclude_hydrogen: If True, set χ² to NaN for H atoms.

    Returns:
        Per-atom χ² array (n_atoms,). H atoms are NaN if excluded.
    """
    n = len(symbols)
    chi_sq = np.full(n, np.nan)
    if exclude_hydrogen:
        mask = np.array([s != "H" for s in symbols])
    else:
        mask = np.ones(n, dtype=bool)
    diff = u_comp[mask] - u_exp[mask]
    chi_sq[mask] = np.sum(diff**2, axis=(1, 2))
    return chi_sq


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
    band_indices: npt.NDArray[np.int64] | None,
    groups: "RefinementGroups",
    asu_symbols: list[str] | None = None,
    exclude_hydrogen: bool = False
) -> None:
    """
    Display refinement summary: frequency comparison and ADP comparison.
    
    Args:
        refinement: NormalModeRefinement result object.
        band_indices: Flat array of band indices.
        groups: Refinement groups from the optimization.
        asu_symbols: Element symbols for ASU atoms.
        exclude_hydrogen: If True, exclude H atoms from ADP display.
    """
    THz_to_cm1 = phonopy.physical_units.get_physical_units().THzToCm
    initial_freqs_cm1 = refinement.freqs_initial_reordered_THz * THz_to_cm1
    refined_freqs_cm1 = refinement.freqs_final_reordered_THz * THz_to_cm1
    
    if band_indices is not None:
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
            freqs_initial=initial_band_avg_cm1,
            freqs_refined=refined_band_avg_cm1,
            optimize_mask=_get_refined_bands_mask(groups, band_indices),
            unit="cm1"
        )
    else:
        print("Warning: Band indices not available, skipping frequency comparison.")

    print_adps_comparison(
        adps_1=refinement.U_cart_exp_Angs2[refinement.asu_atoms],
        adps_2=refinement.U_cart_comp_initial_Angs2[refinement.asu_atoms],
        labels=["experimental", "initial", "refined"],
        symbols=asu_symbols,
        adps_3=refinement.U_cart_comp_final_Angs2[refinement.asu_atoms],
        s12_12=np.nanmean(refinement.s12_initial),
        s12_13=np.nanmean(refinement.s12_final),
        chi_sq_12=np.sqrt(np.nanmean(refinement.chi_sq_initial)),
        chi_sq_13=np.sqrt(np.nanmean(refinement.chi_sq_final)),
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


def run(
    force_constants,
    cif_path: str,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
    restraint_weight: float | None = None,
    band_selection_strategy: Optional["FrequencyPartitionStrategy"] = None,
    weighting_scheme: Literal["sigma", "unit"] = "sigma",
    fix_positions: bool = True,
    exclude_hydrogen_positions: bool = True,
    use_irreducible_fbz: bool = False,
    temperature_K: float | None = None
) -> NormalModeRefinement:
    """
    Perform normal mode refinement.
    
    Args:
        force_constants: mbe_automation ForceConstants object.
        cif_path: Path to experimental CIF.
        mesh_size: k-point mesh size.
        restraint_weight: Weight for restraining to initial frequencies.
        mode_selection_strategy: Strategy for frequency partitioning. Defaults to SensitivityBased(0.6, 0.9).
        weighting_scheme: Weighting scheme for refinement ('sigma' or 'unit').
        
    Returns:
        NormalModeRefinement object with frequencies, ADPs, and mesh data.
    """
    optimizer_options = {"maxiter": 300, "ftol": 1e-9}
    optimizer_method = "SLSQP"

    # Strategy applied to determine which frequencies to refine
    if band_selection_strategy is None:
        band_selection_strategy = SensitivityBasedStrategy(low_threshold=0.75, high_threshold=0.90)

    mbe_automation.common.display.framed([
        "Normal mode refinement",
    ])
    
    print(f"mesh_size            {mesh_size}")
    print(f"restraint_weight     {restraint_weight}")
    print(f"strategy             {band_selection_strategy.__class__.__name__}")
    print(f"max_iter             {optimizer_options['maxiter']}")
    print(f"optimizer_method     {optimizer_method}")
    print(f"weighting_scheme     {weighting_scheme}")
    print(f"fix_positions        {fix_positions}")
    print(f"exclude_hydrogen     {exclude_hydrogen_positions}")
    print(f"use_irreducible_fbz  {use_irreducible_fbz}")
    print(f"temperature_K        {temperature_K if temperature_K is not None else 'from CIF'}")

    if isinstance(mesh_size, (list, tuple, np.ndarray)):
        mesh_size = np.array(mesh_size)

    if isinstance(mesh_size, np.ndarray):
        if np.any(mesh_size % 2 == 0):
            raise ValueError(
                f"Mesh size must consist of odd integers to ensure Gamma point inclusion.\n"
                f"Received: {mesh_size}. Please use odd numbers (e.g., [3, 3, 3])."
            )
    
    print(f"Loading experimental data from {cif_path}")
    cctbx_adapter = CctbxAdapter(cif_path)
    
    print(f"  Space Group: {cctbx_adapter.space_group_symbol}")
    
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
        cif_adapter=cctbx_adapter
    )
    
    if temperature_K is not None:
        temperature = temperature_K
    else:
        temperature = cctbx_adapter.get_temperature()

    if temperature is None:
        raise ValueError(
            "Temperature must be provided either as an argument (`temperature_K`) "
            "or be present in the CIF file metadata."
        )
        
    phonons.temperature = temperature
    
    # Calculate normalization factor (total weight of q-points in BZ)
    total_q = np.sum(q_weights)
    
    print(f"  Normalization factor (Total Q): {total_q}")
    print(f"  Temperature: {temperature} K")
    
    calculator = NoMoReCalculator(
        eigenvectors=phonons.eigenvectors,
        masses=phonons.masses,
        temperature=phonons.temperature, 
        normalization_factor=float(total_q),
        weights=phonons.weights,
        degeneracy_groups=phonons.degeneracy_groups
    )

    # Initialize SmtbxAdapter
    # Requires xray_structure, reflections, phonons (for mapping)
    smtbx_adapter = SmtbxAdapter(
        xray_structure=cctbx_adapter.xray_structure,
        reflections=cctbx_adapter.get_reflections(),
        weighting_scheme=weighting_scheme,
        phonon_data=phonons
    )
    
    print(f"  Reflections: {smtbx_adapter.observations.size()} observations")
    
    # Create mode groups to handle degeneracies and bands
    pre_groups = create_pre_groups(phonons)
    groups = band_selection_strategy.compute_groups(phonons, pre_groups)
    
    engine = RefinementEngine(calculator, smtbx_adapter)
    
    if restraint_weight is None:
        restraint_weight = 0.0
        
    initial_freqs = phonons.frequencies_cm1
    # Clamp low/imaginary frequencies (degeneracy groups are already determined)
    initial_freqs = _clamp_acoustic_frequencies(initial_freqs)
    
    restraint_instance = None
    if restraint_weight > 0.0:
        restraint_instance = BayesianFrequencyRestraint(
            initial_frequencies=initial_freqs,
            temperature=temperature
        )

    result = engine.run_joint(
        initial_frequencies=initial_freqs,
        restraint=restraint_instance,
        restraint_weight=restraint_weight,
        groups=groups,
        optimizer_options=optimizer_options,
        optimizer_method=optimizer_method,
        fix_positions=fix_positions,
        exclude_hydrogen_positions=exclude_hydrogen_positions
    )

    n_atoms_p1 = phonons.n_atoms
    n_modes_flat = len(initial_freqs)
    n_q = len(irr_q_frac)
    n_bands = n_modes_flat // n_q
    assert n_bands == force_constants.primitive.n_atoms * 3
    
    freqs_initial_reordered_cm1 = reorder_frequencies(
        frequencies=initial_freqs.reshape(n_q, n_bands),
        band_indices=phonons.band_indices.reshape(n_q, n_bands)
    )
    freqs_final_reordered_cm1 = reorder_frequencies(
        frequencies=result["frequencies"].reshape(n_q, n_bands),
        band_indices=phonons.band_indices.reshape(n_q, n_bands)
    )

    mode_scaling_factors = _map_group_scales_to_modes(result["scale_factors"], result["groups"])
    
    band_scaling_factors = np.zeros(n_bands)
    for b in range(n_bands):
        band_scaling_factors[b] = np.mean(mode_scaling_factors[phonons.band_indices == b])
        
    # Validation assumes all modes in a band share
    # the same scaling factor.
    _validate_scaling_factors(
        band_scaling_factors, 
        freqs_initial_reordered_cm1, 
        freqs_final_reordered_cm1
    )

    cm1_to_THz = 1.0 / phonopy.physical_units.get_physical_units().THzToCm
    freqs_initial_reordered_THz = freqs_initial_reordered_cm1 * cm1_to_THz
    freqs_final_reordered_THz = freqs_final_reordered_cm1 * cm1_to_THz
    
    U_cart_comp_initial_p1 = calculator.calculate_u_cart(initial_freqs)
    U_cart_comp_final_p1 = calculator.calculate_u_cart(result["frequencies"])
    U_cart_exp_p1 = extract_adps_from_structure(
        cctbx_adapter.xray_structure.expand_to_p1(
            sites_mod_positive=True
        )
    )
    
    asu_atoms = _get_asu_atoms(smtbx_adapter)
    asu_symbols = [sc.element_symbol() for sc in smtbx_adapter.structure.scatterers()]
    U_cart_exp_asu = U_cart_exp_p1[asu_atoms]
    U_cart_comp_initial_asu = U_cart_comp_initial_p1[asu_atoms]
    U_cart_comp_final_asu = U_cart_comp_final_p1[asu_atoms]

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
        s12_initial=_s12_per_atom(
            U_cart_comp_initial_asu, 
            U_cart_exp_asu, 
            asu_symbols, 
            exclude_hydrogen_positions
        ),
        s12_final=_s12_per_atom(
            U_cart_comp_final_asu, 
            U_cart_exp_asu, 
            asu_symbols, 
            exclude_hydrogen_positions
        ),
        chi_sq_initial=_chi_sq_per_atom(
            U_cart_comp_initial_asu, 
            U_cart_exp_asu, 
            asu_symbols, 
            exclude_hydrogen_positions
        ),
        chi_sq_final=_chi_sq_per_atom(
            U_cart_comp_final_asu, 
            U_cart_exp_asu, 
            asu_symbols, 
            exclude_hydrogen_positions
        ),
        band_scaling_factors=band_scaling_factors
    )

    _display_refinement_summary(
        refinement=refinement,
        band_indices=phonons.band_indices,
        groups=groups,
        asu_symbols=asu_symbols,
        exclude_hydrogen=exclude_hydrogen_positions
    )

    return refinement
