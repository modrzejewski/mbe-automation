from __future__ import annotations
import numpy as np
import numpy.typing as npt
import phonopy
import phonopy.phonon.band_structure

from mbe_automation.storage import core
from . import modes

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
    symmetrize_Dq: bool = True,
    symprec: float = 1e-5,
) -> core.BrillouinZonePath:
    """Determine high-symmetry path and calculate phonon dispersion.

    Generates a standard high-symmetry path in the Brillouin zone using
    the SeeK-path library via Phonopy. Calculates frequencies.

    Args
    ----
    phonopy_object : phonopy.Phonopy
        Phonopy object with force constants calculated.
    n_points : int, optional
        Requested number of q-points along a single segment of the path.

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
    
    all_frequencies = []
    all_distances = []

    for i, segment_qpoints in enumerate(bands):
        path = np.array(segment_qpoints)
        distances_segment, current_distance = _calculate_segment_distances(
            path, current_distance, reciprocal_lattice_inv_T
        )

        all_distances.append(np.array(distances_segment))
        all_kpoints.append(path)

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
