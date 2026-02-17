from __future__ import annotations
import numpy as np
import numpy.typing as npt
import phonopy
import phonopy.phonon.band_structure

from mbe_automation.storage import core
from . import modes

try:
    import nomore_ase
    from . import bands
    from nomore_ase.optimization.band_assignment import (
        match_modes_hungarian,
        resolve_degenerate_basis,
        eigenvector_overlap_matrix,
    )
    _NOMORE_AVAILABLE = True
except ImportError:
    _NOMORE_AVAILABLE = False

def from_phonopy(
    band_structure: phonopy.phonon.band_structure.BandStructure,
) -> core.BrillouinZonePath:
    """Convert a BandStructure object to a BrillouinZonePath.

    Args
    ----
    band_structure : phonopy.phonon.band_structure.BandStructure
        Phonopy BandStructure object containing frequencies, q-points,
        distances, and path connections.

    Returns
    -------
    core.BrillouinZonePath
        Converted object.
    """
    return core.BrillouinZonePath(
        kpoints=band_structure.qpoints,
        frequencies=band_structure.frequencies,
        path_connections=np.array(band_structure.path_connections, dtype=bool),
        labels=np.array(band_structure.labels, dtype=str),
        distances=band_structure.distances,
    )

def _adapt_basis_to_perturbation(
    D_q_list: list[npt.NDArray[np.complex128]],
    evals_list: list[npt.NDArray[np.float64]],
    evecs_list: list[npt.NDArray[np.complex128]],
    degenerate_freqs_tol_cm1: float = 0.5,
) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.complex128]]]:
    """Refine eigenvectors using degenerate perturbation theory.
    
    Resolves mixing in degenerate subspaces by diagonalizing the perturbation
    matrix dD (change in dynamical matrix along the path). This ensures
    continuous bands across the FBZ path.
    """
    n_q = len(D_q_list)
    refined_evecs = []
    refined_evals = []
    
    for k in range(n_q):
        w2 = evals_list[k]
        v = evecs_list[k]
        D_current = D_q_list[k]
        
        if k < n_q - 1:
            dD = D_q_list[k+1] - D_current
        elif k > 0:
            dD = D_current - D_q_list[k-1]
        else:
            dD = np.zeros_like(D_current)
            
        w2_resolved, v_resolved = resolve_degenerate_basis(
            w2, v, dD, degenerate_freqs_tol_cm1=degenerate_freqs_tol_cm1
        )
        refined_evals.append(w2_resolved)
        
        n_modes = v_resolved.shape[1]
        n_atoms = v_resolved.shape[0] // 3
        evecs_reshaped = v_resolved.T.reshape(n_modes, n_atoms, 3)
        refined_evecs.append(evecs_reshaped)

    return refined_evals, refined_evecs

def _segment_freqs(
    q_paths: list[npt.NDArray[np.float64]],
    path_connections: npt.NDArray[np.bool_],
    phonopy_object: phonopy.Phonopy,
) -> list[npt.NDArray[np.float64]]:
    """Extract start points of segments and track bands from Gamma.

    Calculates phonon frequencies and eigenvectors along the specified q-paths.
    Uses degenerate perturbation theory to resolve band crossings and
    Hungarian matching of eigenvector overlaps to ensure continuous band
    indices (tracking).
    
    If segments are connected (as indicated by path_connections), the band 
    mapping is propagated from the end of one segment to the start of the next,
    ensuring visual continuity. Otherwise, the mapping is reset to the 
    global assignment determined by `track_from_gamma`.

    Args
    ----
    q_paths : list[np.ndarray]
        List of arrays of q-points, where each array represents a continuous 
        segment of the brillouin zone path.
    path_connections : np.ndarray
        Boolean array indicating if segment i connects to segment i+1.
    phonopy_object : phonopy.Phonopy
        Phonopy object.

    Returns
    -------
    list[np.ndarray]
        List of frequency arrays for each segment, where each array has
        shape (n_qpoints, n_bands) and bands are tracked/continuous.
    """

    start_points = [path[0] for path in q_paths]
    start_assignments = bands.track_from_gamma(phonopy_object, np.array(start_points))
    
    physical_units = phonopy.physical_units.get_physical_units()
    to_THz = physical_units.DefaultToTHz
    
    segment_frequencies = []
    
    # Store the mapping from the end of the previous segment
    previous_end_mapping = None
    
    for i, path in enumerate(q_paths):
        n_q = len(path)
        D_q_list = []
        for q in path:
            phonopy_object.dynamical_matrix.run(q)
            D_q_list.append(phonopy_object.dynamical_matrix.dynamical_matrix)
            
        evals_list = []
        evecs_list = []
        for D in D_q_list:
            w2, v = np.linalg.eigh(D)
            evals_list.append(w2)
            evecs_list.append(v)
            
        refined_evals, refined_evecs = _adapt_basis_to_perturbation(
            D_q_list, evals_list, evecs_list
        )
            
        n_modes = len(refined_evals[0])
        
        # Determine starting mapping
        # If this segment is connected to the previous one, use propagated mapping for continuity.
        # Otherwise, reset to the global "track from Gamma" assignment.
        if i > 0 and path_connections[i-1] and previous_end_mapping is not None:
             current_mapping = previous_end_mapping.copy()
        else:
             current_mapping = start_assignments[i].copy()
             
        segment_freqs_sorted = np.zeros((n_q, n_modes))
        
        freqs_0 = (np.sqrt(np.abs(refined_evals[0])) * np.sign(refined_evals[0])) * to_THz
        
        for m in range(n_modes):
            segment_freqs_sorted[0, current_mapping[m]] = freqs_0[m]
            
        for k in range(n_q - 1):
            evecs_k = refined_evecs[k]
            evecs_kplus1 = refined_evecs[k+1]
            
            overlap = eigenvector_overlap_matrix(evecs_k, evecs_kplus1)
            step_assignment = match_modes_hungarian(overlap)
            
            next_mapping = np.zeros_like(current_mapping)
            for m in range(n_modes):
                next_mapping[step_assignment[m]] = current_mapping[m]
            current_mapping = next_mapping
            
            freqs_kplus1 = (np.sqrt(np.abs(refined_evals[k+1])) * np.sign(refined_evals[k+1])) * to_THz
            for m in range(n_modes):
                segment_freqs_sorted[k+1, current_mapping[m]] = freqs_kplus1[m]
                
        segment_frequencies.append(segment_freqs_sorted)
        previous_end_mapping = current_mapping
        
    return segment_frequencies

def _calculate_segment_distances(
    segment_qpoints: npt.NDArray[np.float64],
    current_distance: float,
    reciprocal_lattice_inv_T: npt.NDArray[np.float64],
) -> tuple[list[float], float]:
    """Calculate cumulative distances for a path segment.
    
    Computes the path length in reciprocal space units (1/Angstrom) accounting
    for the metric tensor of the primitive cell.
    """
    distances = []
    distances.append(current_distance)
    last_q = segment_qpoints[0].copy()

    for q in segment_qpoints[1:]:
        diff = q - last_q
        metric_diff = np.dot(diff, reciprocal_lattice_inv_T)
        dist_step = np.linalg.norm(metric_diff)
        current_distance += dist_step
        distances.append(current_distance)
        last_q = q.copy()

    return distances, current_distance

def init_fbz_path(
    phonopy_object: phonopy.Phonopy,
    n_points: int = 20,
    track_bands: bool = False,
) -> core.BrillouinZonePath:
    """Determine high-symmetry path and calculate phonon dispersion.

    Generates a standard high-symmetry path in the Brillouin zone using
    the SeeK-path library via Phonopy. Calculates frequencies and 
    optionally tracks band connectivity using degenerate perturbation theory 
    if available.

    Args
    ----
    phonopy_object : phonopy.Phonopy
        Phonopy object with force constants calculated.
    n_points : int, optional
        Requested number of q-points along a single segment of the path.
    track_bands : bool, optional
        Whether to enforce continuous band tracking using eigenvector overlaps
        and degenerate perturbation theory. Requires `nomore_ase`.

    Returns
    -------
    core.BrillouinZonePath
        Object containing the calculated dispersion, q-points, labels, and distances.
    """
    bands, labels, path_connections = phonopy.phonon.band_structure.get_band_qpoints_by_seekpath(
        phonopy_object.primitive,
        n_points,
        is_const_interval=True,
    )
    current_distance = 0.0
    reciprocal_lattice_inv_T = np.linalg.inv(phonopy_object.primitive.cell).T
    all_kpoints = []
    
    # Prepare q_paths for batch processing if band connection is requested
    q_paths = [np.array(segment) for segment in bands]
    
    # Try using advanced band assignment if requested and available
    segment_frequencies = None
    if track_bands:
        if not _NOMORE_AVAILABLE:
            raise ImportError(
                "nomore_ase is required for track_bands=True. "
                "Please install it or set track_bands=False."
            )
        segment_frequencies = _segment_freqs(
            q_paths=q_paths, 
            path_connections=np.array(path_connections, dtype=bool),
            phonopy_object=phonopy_object
        )
        
    all_frequencies = []
    all_distances = []

    for i, segment_qpoints in enumerate(bands):
        path = np.array(segment_qpoints)
        distances_segment, current_distance = _calculate_segment_distances(
            path, current_distance, reciprocal_lattice_inv_T
        )

        all_distances.append(np.array(distances_segment))
        all_kpoints.append(path)

        if segment_frequencies is not None:
            # Use tracked frequencies
            all_frequencies.append(segment_frequencies[i])
        else:
            # Fallback to standard calculation (no tracking)
            freqs, vecs = modes.at_k_points(
                dynamical_matrix=phonopy_object.dynamical_matrix,
                k_points=path,
                compute_eigenvecs=True,
                eigenvectors_storage="columns",
            )
            all_frequencies.append(freqs)

    return core.BrillouinZonePath(
        kpoints=all_kpoints,
        frequencies=all_frequencies,
        path_connections=np.array(path_connections, dtype=bool),
        labels=np.array(labels, dtype=str),
        distances=all_distances,
    )
