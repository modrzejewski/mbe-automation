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
        eigenvector_overlap_matrix,
        interpolate_and_track_bands,
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


def _resolve_segment_connection(
    q_start: npt.NDArray, 
    q_end: npt.NDArray, 
    phonopy_object: phonopy.Phonopy,
    delta_q: float = 0.05,
    degenerate_freqs_tol_cm1: float = 0.5,
    symmetrize_Dq: bool = False,
    symprec: float = 1e-5,
) -> npt.NDArray:
    """
    Bridge two discontinuous q-points by generating an interpolated straight path.
    
    Generates a virtual path between q_start and q_end and tracks the evolution
    of eigenvectors to determine how modes map from one point to the other.
    
    Parameters
    ----------
    q_start : array-like, shape (3,)
        Starting q-point in fractional coordinates.
    q_end : array-like, shape (3,)
        Ending q-point in fractional coordinates.
    phonopy_object : phonopy.Phonopy
        Phonopy object containing force constants and structure.
        
    Returns
    -------
    mapping : np.ndarray, shape (n_modes,)
        Array of indices where mapping[i] = j means mode i at q_start maps to mode j at q_end.
    """
    from . import bands
    adapter = bands.PhonopyASEAdapter(phonopy_object, symmetrize_Dq=symmetrize_Dq, symprec=symprec)
    D_N = adapter.get_force_constant()
    
    # Use default spacing from bands module if available, else hardcode
    q_spacing = getattr(bands, "DEFAULT_Q_SPACING", 0.05)
    
    mapping = interpolate_and_track_bands(
        phonons=adapter,
        D_N=D_N,
        q_start=q_start,
        q_end=q_end,
        q_spacing=q_spacing,
        use_degenerate_pt=True,
        degenerate_freqs_tol_cm1=degenerate_freqs_tol_cm1,
        delta_q=delta_q,
    )
    return mapping


def _segment_labels(
    labels: list[str] | npt.NDArray[np.str_],
    path_connections: list[bool] | npt.NDArray[np.bool_],
) -> list[list[str]]:
    """Unpack compressed path labels into start and end labels per segment.

    The input `labels` list uses a compressed format where connected segments
    share a single label point. If segment i is connected to segment i+1,
    the end label of segment i serves as the start label of segment i+1 and
    appears only once in the list. Disconnected segments list both labels.

    Example
    -------
    Connected path A->B->C:
        labels=['A', 'B', 'C'], path_connections=[True, False]
        Result: [['A', 'B'], ['B', 'C']]

    Discontinuous path A->B, C->D:
        labels=['A', 'B', 'C', 'D'], path_connections=[False, False]
        Result: [['A', 'B'], ['C', 'D']]

    Args
    ----
    labels : list[str] | np.ndarray
        List of high-symmetry point labels in compressed format.
    path_connections : list[bool] | np.ndarray
        Boolean array indicating if segment i connects to segment i+1.

    Returns
    -------
    list[list[str]]
        List of [start_label, end_label] pairs for each segment.
    """
    segment_labels = []
    idx = 0
    for connection in path_connections:
        start = labels[idx]
        end = labels[idx+1]
        segment_labels.append([start, end])

        if connection:
            idx += 1
        else:
            idx += 2

    return segment_labels


def _segment_freqs(
    q_paths: list[npt.NDArray[np.float64]],
    path_connections: npt.NDArray[np.bool_],
    phonopy_object: phonopy.Phonopy,
    labels: list[str] | npt.NDArray[np.str_],
    delta_q: float = 0.05,
    degenerate_freqs_tol_cm1: float = 0.5,
    symmetrize_Dq: bool = False,
    symprec: float = 1e-5,
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
    labels : list[str] | np.ndarray
        List of high-symmetry point labels in compressed format.

    Returns
    -------
    list[np.ndarray]
        List of frequency arrays for each segment, where each array has
        shape (n_qpoints, n_bands) and bands are tracked/continuous.
    """

    start_points = [path[0] for path in q_paths]
    map_path_from_gamma = bands.track_from_gamma(
        phonopy_object, 
        np.array(start_points),
        delta_q=delta_q,
        degenerate_freqs_tol_cm1=degenerate_freqs_tol_cm1,
        symmetrize_Dq=symmetrize_Dq,
        symprec=symprec,
    )
    
    physical_units = phonopy.physical_units.get_physical_units()
    to_THz = physical_units.DefaultToTHz

    segment_labels = _segment_labels(labels, path_connections)
    map_high_symmetry_path = {}
    
    segment_frequencies = []
    
    # Store the mapping from the end of the previous segment
    previous_end_mapping = None
    
    for i, path in enumerate(q_paths):
        n_q = len(path)
        D_q_list = []
        for q in path:
            if symmetrize_Dq:
                from mbe_automation.dynamics.harmonic.symmetry import symmetrized_dynamical_matrix
                D_q_list.append(symmetrized_dynamical_matrix(phonopy_object, q, tolerance=symprec))
            else:
                phonopy_object.dynamical_matrix.run(q)
                D_q_list.append(phonopy_object.dynamical_matrix.dynamical_matrix)

        epsilon = 0.001
        dD_segment = D_q_list[-1] - D_q_list[0]
        refined_evals = []
        refined_evecs = []
        for D_q in D_q_list:
            _, v = np.linalg.eigh((1.0 - epsilon) * D_q + epsilon * dD_segment)
            w2 = np.real(np.einsum("ij,ij->j", v.conj(), D_q @ v))
            refined_evals.append(w2)
            n_modes = v.shape[1]
            n_atoms = v.shape[0] // 3
            refined_evecs.append(v.T.reshape(n_modes, n_atoms, 3))
            
        n_modes = len(refined_evals[0])
        
        start_label, end_label = segment_labels[i]

        # Determine starting mapping
        # 1. Path Continuity
        if i > 0 and previous_end_mapping is not None and path_connections[i-1]:
             current_mapping = previous_end_mapping.copy()
        
        # 2. Label Cache Lookup
        elif start_label in map_high_symmetry_path:
             current_mapping = map_high_symmetry_path[start_label].copy()

        # 3. Geometric Proximity & 4. Gamma Fallback
        else:
             # Check geometric proximity (effective continuity) fallback
             if i > 0 and previous_end_mapping is not None:
                 q_prev = q_paths[i-1][-1]
                 q_curr = q_paths[i][0]
                 # Use squared Euclidean distance in fractional coordinates
                 # 1e-10 tolerance is sufficient for identifying "same point"
                 if np.sum((q_prev - q_curr)**2) < 1e-10:
                      bridge_mapping = _resolve_segment_connection(q_prev, q_curr, phonopy_object, symmetrize_Dq=symmetrize_Dq, symprec=symprec)
                      current_mapping = bridge_mapping[previous_end_mapping]
                 else:
                      current_mapping = map_path_from_gamma[i].copy()
             else:
                 current_mapping = map_path_from_gamma[i].copy()
             
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
        
        if end_label not in map_high_symmetry_path:
            map_high_symmetry_path[end_label] = current_mapping.copy()
        
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
    delta_q: float = 0.05,
    degenerate_freqs_tol_cm1: float = 0.5,
    symmetrize_Dq: bool = False,
    symprec: float = 1e-5,
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
            phonopy_object=phonopy_object,
            labels=list(labels),
            delta_q=delta_q,
            degenerate_freqs_tol_cm1=degenerate_freqs_tol_cm1,
            symmetrize_Dq=symmetrize_Dq,
            symprec=symprec,
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
                phonopy_object=phonopy_object,
                k_points=path,
                compute_eigenvecs=True,
                eigenvectors_storage="columns",
                symmetrize_Dq=symmetrize_Dq,
                symprec=symprec,
            )
            all_frequencies.append(freqs)

    return core.BrillouinZonePath(
        kpoints=all_kpoints,
        frequencies=all_frequencies,
        path_connections=np.array(path_connections, dtype=bool),
        labels=np.array(labels, dtype=str),
        distances=all_distances,
    )
