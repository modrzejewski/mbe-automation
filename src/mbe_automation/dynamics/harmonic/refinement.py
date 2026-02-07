"""
Integration module for specialized normal mode refinement using
RefinementEngine and direct CCTBX/SMTBX adapters.

"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Dict, Any, Optional, Literal, List, TYPE_CHECKING
import phonopy.physical_units
import mbe_automation.dynamics.harmonic.modes
from scipy.spatial.distance import cdist

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

    from mbe_automation.dynamics.harmonic.bands import compute_band_indices
except ImportError:
    raise ImportError(
        "The `dynamics.harmonic.refinement` module requires the `nomore_ase` package. "
        "Install it in your environment to use this functionality."
    )

def _to_phonon_data(
    phonopy_object,
    irr_q_frac: npt.NDArray[np.floating],
    q_weights: npt.NDArray[np.integer],
    cif_adapter: "CctbxAdapter",
    compute_bands: bool = True
) -> PhononData:
    """
    Create PhononData object from computed phonopy data, matching atoms to CIF.

    Args:
        phonopy_object: The phonopy object with initialized mesh.
        irr_q_frac: Irreducible q-points (fractional).
        q_weights: Weights of irreducible q-points.
        cif_adapter: Adapter containing the experimental structure (CIF).
        compute_bands: Whether to compute band indices.

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
    
    source_to_target_indices = _compute_atom_permutation(ph.primitive, cif_adapter)
    
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

    degeneracy_groups = _find_degeneracy_groups(flat_freqs_cm1, tolerance=1.0)
    flat_freqs_cm1 = _average_degenerate_frequencies(flat_freqs_cm1, degeneracy_groups)

    band_indices = None
    if compute_bands:
        print(f"Computing band assignment for {len(irr_q_frac)} q-points...")
        band_indices = compute_band_indices(
            phonopy_object=ph,
            q_points=irr_q_frac
        )
        band_indices = band_indices.flatten()
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
        supercell=[1, 1, 1],
        n_atoms=n_target,
        band_indices=band_indices
    )


def _validate_atomic_numbers(
    source_numbers: npt.NDArray[np.int64],
    target_symbols: List[str],
    source_to_target_indices: npt.NDArray[np.int64]
) -> None:
    """
    Validate that permuted source atomic numbers match target symbols.
    
    Args:
        source_numbers: Atomic numbers from source (e.g. Phonopy primitive).
        target_symbols: Element symbols from target (e.g. CIF).
        source_to_target_indices: Permutation mapping source -> target.
        
    Raises:
        ValueError: If atomic numbers do not match after permutation.
    """
    from phonopy.structure.atoms import symbol_map
    
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


def _find_degeneracy_groups(freqs: np.ndarray, tolerance: float) -> List[List[int]]:
    """
    Find index groups of degenerate modes.
    
    Args:
        freqs: Array of frequencies.
        tolerance: Tolerance for degeneracy (same units as freqs).
    """
    n_modes = len(freqs)
    visited = np.zeros(n_modes, dtype=bool)
    groups = []
    
    argsort = np.argsort(freqs)
    
    current_group = [argsort[0]]
    visited[argsort[0]] = True
    
    for i in range(1, n_modes):
        idx = argsort[i]
        prev_idx = argsort[i-1]
        
        if np.abs(freqs[idx] - freqs[prev_idx]) < tolerance:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
        visited[idx] = True
    groups.append(current_group)
    
    return groups


def _average_degenerate_frequencies(freqs: np.ndarray, groups: List[List[int]]) -> np.ndarray:
    """Enforce exact degeneracy by averaging."""
    new_freqs = freqs.copy()
    for group in groups:
        if len(group) > 1:
            avg = np.mean(freqs[group])
            new_freqs[group] = avg
    return new_freqs


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


def _compute_atom_permutation(phonopy_primitive: Any, cif_adapter: "CctbxAdapter") -> npt.NDArray[np.int64]:
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


def run(
    force_constants,
    cif_path: str,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
    restraint_weight: float | None = None,
    strategy: Optional["FrequencyPartitionStrategy"] = None,
    max_iter: int = 200,
    optimizer_method: str = "SLSQP",
    weighting_scheme: Literal["sigma", "unit"] = "sigma",
    fix_positions: bool = True,
    exclude_hydrogen_positions: bool = True,
) -> Dict[str, Any]:
    """
    Run NoMoRe refinement using the v2 API (RefinementEngine + Adapter).
    
    Args:
        force_constants: mbe_automation ForceConstants object.
        cif_path: Path to experimental CIF.
        mesh_size: k-point mesh size.
        restraint_weight: Weight for restraining to initial frequencies.
        strategy: Strategy for frequency partitioning. Defaults to SensitivityBased(0.6, 0.9).
        max_iter: Maximum optimization iterations.
        optimizer_method: Optimizer method (e.g., 'SLSQP', 'L-BFGS-B').
        weighting_scheme: Weighting scheme for refinement ('sigma' or 'unit').
        
    Returns:
        Dictionary with refinement results.
    """
    
    # 1. Initialize CctbxAdapter (loads CIF)
    print(f"Loading experimental data from {cif_path}")
    cctbx_adapter = CctbxAdapter(cif_path)
    
    print(f"  Space Group: {cctbx_adapter.space_group_symbol}")
    
    # 2. Prepare Phonons (computes freqs/eigenvectors on mesh)
    print("Computing phonon data on mesh...")
    
    # 2a. Initialize Phonopy and get mesh
    ph = force_constants.to_phonopy()
    irr_q_frac, q_weights = mbe_automation.dynamics.harmonic.modes.phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=mesh_size,
        use_symmetry=True,
        odd_numbers=True  # Enforce odd mesh to guarantee Gamma point (0,0,0) is included
    )
    
    # 2b. Build PhononData
    phonons = _to_phonon_data(
        phonopy_object=ph,
        irr_q_frac=irr_q_frac,
        q_weights=q_weights,
        cif_adapter=cctbx_adapter
    )
    
    # Extract temperature from CIF or set default
    temperature = cctbx_adapter.get_temperature()
    if temperature is None:
        print("WARNING: Temperature not found in CIF. Defaulting to 298.15 K.")
        temperature = 298.15
        
    phonons.temperature = temperature
    
    # 3. Create NoMoReCalculator
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

    # 4. Initialize SmtbxAdapter
    # Requires xray_structure, reflections, phonons (for mapping)
    smtbx_adapter = SmtbxAdapter(
        xray_structure=cctbx_adapter.xray_structure,
        reflections=cctbx_adapter.get_reflections(),
        weighting_scheme=weighting_scheme,
        phonon_data=phonons
    )
    
    print(f"  Reflections: {smtbx_adapter.observations.size()} observations")

    # 5. Frequency Partition Strategy
    if strategy is None:
        print("  Strategy: Default (SensitivityBasedStrategy 0.60 - 0.90)")
        strategy = SensitivityBasedStrategy(low_threshold=0.60, high_threshold=0.90)
    else:
         print(f"  Strategy: Provided {strategy}")
    
    # Create pre-groups (handles degeneracies and bands)
    pre_groups = create_pre_groups(phonons)
    
    # Compute refinement groups
    groups = strategy.compute_groups(phonons, pre_groups)
    
    # 6. Initialize Refinement Engine
    engine = RefinementEngine(calculator, smtbx_adapter)
    
    # 7. Run Refinement
    if restraint_weight is None:
        restraint_weight = 0.0
        
    # We need initial frequencies from phonons
    initial_freqs = phonons.frequencies_cm1
    
    # Create restraint object if weight is positive
    restraint_instance = None
    if restraint_weight > 0.0:
        from nomore_ase.optimization.restraints import BayesianFrequencyRestraint
        restraint_instance = BayesianFrequencyRestraint(
            initial_frequencies=initial_freqs,
            temperature=temperature
        )

    result = engine.run_joint(
        initial_frequencies=initial_freqs,
        restraint=restraint_instance,
        restraint_weight=restraint_weight,
        groups=groups,
        max_iter=max_iter,
        optimizer_method=optimizer_method,
        fix_positions=fix_positions,
        exclude_hydrogen_positions=exclude_hydrogen_positions
    )
    
    return result
