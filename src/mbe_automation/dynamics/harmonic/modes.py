import numpy as np
import numpy.typing as npt
import ase

import mbe_automation.storage

def commensurate_kpoints(
        dataset: str,
        key: str
) -> npt.NDArray:
    
    """Generate a list of the commensurate k-points (in reduced units)
    which fit into to the force model's supercell.

    Parameters
    ----------
    dataset
        Path to the dataset file.
    key
        Key to the harmonic force constants model.
    """

    mp = mbe_automation.storage.to_dynasor_mode_projector(
        dataset=dataset,
        key=key
    )
    return mp.q_reduced


def trajectory(
    dataset: str,
    key: str,
    k_point: npt.NDArray[np.floating],
    band_index: int,
    max_amplitude: float,
    n_frames: int = 100,
) -> mbe_automation.storage.Structure:
    
    """Generate a trajectory for a single vibrational mode.

    Parameters
    ----------
    dataset
        Path to the dataset file.
    key
        Key to the harmonic force constants model.
    k_point
        Reduced coordinates of the k-point.
    band_index
        Index of the vibrational mode.
    max_amplitude
        Controls the maximum displacement.
    n_frames
        Number of frames for the trajectory.

    Returns
    -------
    Structure
        A Structure object containing the mode trajectory.
    """
    mp = mbe_automation.storage.to_dynasor_mode_projector(
        dataset=dataset,
        key=key
    )
    try:
        q_point = mp(k_point)
        selected_mode = q_point.bands[band_index]
    except ValueError:
        raise ValueError(f"k-point {k_point} not found on the commensurate grid.")
    except IndexError:
        n_bands = len(q_point.bands)
        raise IndexError(
            f"Band index {band_index} is out of bounds. "
            f"There are {n_bands} bands at this k-point (indices 0 to {n_bands-1})."
        )
    supercell = mp.supercell.to_ase()
    n_atoms_supercell = len(supercell)
    atomic_numbers = supercell.get_atomic_numbers()
    masses = supercell.get_masses()
    cell = supercell.get_cell()

    positions_array = np.zeros((n_frames, n_atoms_supercell, 3))
    amplitudes = np.linspace(-max_amplitude, max_amplitude, n_frames)

    for i, amp in enumerate(amplitudes):
        selected_mode.Q.amplitude = amp
        displaced_atoms = mp.get_atoms()
        positions_array[i] = displaced_atoms.get_positions()

    trajectory = mbe_automation.storage.Structure(
        positions=positions_array,
        atomic_numbers=atomic_numbers,
        masses=masses,
        cell_vectors=cell,
        periodic=True,
        n_atoms=n_atoms_supercell,
        n_frames=n_frames
    )
    return trajectory





