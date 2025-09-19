from __future__ import annotations
import numpy as np
import numpy.typing as npt
import ase

import phonopy

import mbe_automation.storage

def trajectory(
        dataset: str,
        key: str,
        k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
        band_index: int = 0,
        max_amplitude: float = 1.0,
        n_frames: int = 10,
        wrap: bool = False,
) -> mbe_automation.storage.Structure:
    """
    Generate a primitive cell trajectory for a single vibrational mode.

    This function uses the explicit algorithm found in phonopy's Animation
    class, manually calculating displacements from the dynamical matrix
    eigenvectors to produce a smooth, sinusoidal oscillation. The trajectory
    is calculated for the primitive cell.

    Args:
        dataset: Path to the dataset file.
        key: Key to the harmonic force constants model.
        k_point: Reduced coordinates of the k-point.
        band_index: Index of the vibrational mode.
        max_amplitude: Controls the maximum displacement of the mode.
        n_frames: Number of frames for the trajectory.
        wrap: If True, wrap atomic coordinates back into the primary
              primitive cell.

    Returns:
        A Structure object containing the mode trajectory for the primitive cell.
    """
    ph = mbe_automation.storage.to_phonopy(
        dataset=dataset,
        key=key
    )

    primitive_cell = ph.primitive
    n_bands = 3 * len(primitive_cell)
    if not (0 <= band_index < n_bands):
        raise IndexError(
            f"band_index must be on the interval [0, {n_bands - 1}]."
        )

    # 1. Run dynamical matrix and get eigenvectors for the primitive cell
    dm = ph.dynamical_matrix
    dm.run(k_point)
    eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)

    # 2. Calculate mass-normalized displacements for the primitive cell
    masses = primitive_cell.masses
    eigvec = eigenvectors[:, band_index]
    displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)

    # 3. Generate frames using the sinusoidal motion formula for the primitive cell
    equilibrium_positions = primitive_cell.positions
    positions_array = np.zeros((n_frames, len(primitive_cell), 3))

    for i in range(n_frames):
        phase = 2j * np.pi * i / n_frames
        # Use .imag for a sine-based oscillation to match write_xyz
        frame_displacement = (displacements * np.exp(phase)).imag * max_amplitude
        positions_array[i] = equilibrium_positions + frame_displacement

    if wrap:
        lattice = primitive_cell.cell
        inv_lattice = np.linalg.inv(lattice)
        # Reshape for vectorized operation on all frames at once
        flat_pos = positions_array.reshape(-1, 3)
        frac_pos = np.dot(flat_pos, inv_lattice)
        wrapped_frac_pos = frac_pos - np.floor(frac_pos)
        wrapped_cart_pos = np.dot(wrapped_frac_pos, lattice)
        positions_array = wrapped_cart_pos.reshape(n_frames, -1, 3)

    trajectory = mbe_automation.storage.Structure(
        positions=positions_array,
        atomic_numbers=primitive_cell.numbers,
        masses=primitive_cell.masses,
        cell_vectors=primitive_cell.cell,
        n_atoms=len(primitive_cell),
        n_frames=n_frames,
    )
    return trajectory




# def trajectory(
#         dataset: str,
#         key: str,
#         k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
#         band_index: int = 0,
#         max_amplitude: float = 1.0,
#         n_frames: int = 10,
#         wrap: bool = False,
# ) -> mbe_automation.storage.Structure:
#     """
#     Generate a trajectory for a single vibrational mode.

#     This function uses the explicit algorithm found in phonopy's Animation
#     class, manually calculating displacements from the dynamical matrix
#     eigenvectors to produce a smooth, sinusoidal oscillation.

#     Args:
#         dataset: Path to the dataset file.
#         key: Key to the harmonic force constants model.
#         k_point: Reduced coordinates of the k-point.
#         band_index: Index of the vibrational mode.
#         max_amplitude: Controls the maximum displacement of the mode.
#         n_frames: Number of frames for the trajectory.
#         wrap: If True, wrap atomic coordinates back into the primary
#               supercell. Set to False for smooth animation trajectories.

#     Returns:
#         A Structure object containing the mode trajectory.
#     """
#     ph = mbe_automation.storage.to_phonopy(
#         dataset=dataset,
#         key=key
#     )

#     n_bands = 3 * len(ph.primitive)
#     if not (0 <= band_index < n_bands):
#         raise IndexError(
#             f"band_index must be on the interval [0, {n_bands - 1}]."
#         )

#     # 1. Run dynamical matrix and get eigenvectors
#     dm = ph.dynamical_matrix
#     dm.run(k_point)
#     eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)

#     # 2. Calculate mass-normalized displacements
#     masses = ph.primitive.masses
#     eigvec = eigenvectors[:, band_index]
#     displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)

#     # 3. Generate frames using the sinusoidal motion formula
#     base_supercell = ph.supercell
#     equilibrium_positions = base_supercell.positions
#     positions_array = np.zeros((n_frames, len(base_supercell), 3))

#     for i in range(n_frames):
#         phase = 2j * np.pi * i / n_frames
#         # Use .imag for a sine-based oscillation to match write_xyz
#         frame_displacement = (displacements * np.exp(phase)).imag * max_amplitude
#         positions_array[i] = equilibrium_positions + frame_displacement

#     if wrap:
#         lattice = base_supercell.cell
#         inv_lattice = np.linalg.inv(lattice)
#         # Reshape for vectorized operation on all frames at once
#         flat_pos = positions_array.reshape(-1, 3)
#         frac_pos = np.dot(flat_pos, inv_lattice)
#         wrapped_frac_pos = frac_pos - np.floor(frac_pos)
#         wrapped_cart_pos = np.dot(wrapped_frac_pos, lattice)
#         positions_array = wrapped_cart_pos.reshape(n_frames, -1, 3)

#     trajectory = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=base_supercell.numbers,
#         masses=base_supercell.masses,
#         cell_vectors=base_supercell.cell,
#         n_atoms=len(base_supercell),
#         n_frames=n_frames,
#     )
#     return trajectory



# def trajectory(
#         dataset: str,
#         key: str,
#         k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
#         band_index: int = 0,
#         max_amplitude: float = 1.0,
#         n_frames: int = 100,
#         wrap: bool = True,
# ) -> mbe_automation.storage.Structure:
#     """
#     Generate a trajectory for a single vibrational mode using phonopy.

#     Args:
#         dataset: Path to the dataset file.
#         key: Key to the harmonic force constants model.
#         k_point: Reduced coordinates of the k-point.
#         band_index: Index of the vibrational mode.
#         max_amplitude: Controls the maximum displacement of the mode.
#         n_frames: Number of frames for the trajectory.
#         wrap: If True, wrap atomic coordinates back into the primary
#               supercell. Set to False for smooth animation trajectories.

#     Returns:
#         A Structure object containing the mode trajectory.
#     """
#     ph = mbe_automation.storage.to_phonopy(
#         dataset=dataset,
#         key=key
#     )

#     n_bands = 3 * len(ph.primitive)
#     if not (0 <= band_index < n_bands):
#         raise IndexError(
#             f"band_index must be on the interval [0, {n_bands - 1}]."
#         )
    
#     amplitudes = np.linspace(-max_amplitude, max_amplitude, n_frames)
#     phonon_modes = [[k_point, band_index, amp, 0.0] for amp in amplitudes]
#     mod = Modulation(
#         dynamical_matrix=ph.dynamical_matrix,
#         dimension=ph.supercell_matrix,
#         phonon_modes=phonon_modes,
#     )
#     mod.run()

#     raw_displacements, base_supercell = mod.get_modulations_and_supercell()
#     positions_array = base_supercell.positions + raw_displacements.real

#     if wrap:
#         lattice = base_supercell.cell
#         inv_lattice = np.linalg.inv(lattice)
#         frac_pos = np.dot(positions_array, inv_lattice)
#         wrapped_frac_pos = frac_pos - np.floor(frac_pos)
#         positions_array = np.dot(wrapped_frac_pos, lattice)

#     trajectory = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=base_supercell.numbers,
#         masses=base_supercell.masses,
#         cell_vectors=base_supercell.cell,
#         n_atoms=len(base_supercell),
#         n_frames=n_frames,
#     )
#     return trajectory



# def commensurate_kpoints(
#         dataset: str,
#         key: str
# ) -> npt.NDArray:
    
#     """Generate a list of the commensurate k-points (in reduced units)
#     which fit into to the force model's supercell.

#     Parameters
#     ----------
#     dataset
#         Path to the dataset file.
#     key
#         Key to the harmonic force constants model.
#     """

#     mp = mbe_automation.storage.to_dynasor_mode_projector(
#         dataset=dataset,
#         key=key
#     )
#     return mp.q_reduced


# def trajectory(
#     dataset: str,
#     key: str,
#     k_point: npt.NDArray[np.floating],
#     band_index: int,
#     max_amplitude: float,
#     n_frames: int = 100,
# ) -> mbe_automation.storage.Structure:
    
#     """Generate a trajectory for a single vibrational mode.

#     Parameters
#     ----------
#     dataset
#         Path to the dataset file.
#     key
#         Key to the harmonic force constants model.
#     k_point
#         Reduced coordinates of the k-point.
#     band_index
#         Index of the vibrational mode.
#     max_amplitude
#         Controls the maximum displacement.
#     n_frames
#         Number of frames for the trajectory.

#     Returns
#     -------
#     Structure
#         A Structure object containing the mode trajectory.
#     """
#     mp = mbe_automation.storage.to_dynasor_mode_projector(
#         dataset=dataset,
#         key=key
#     )
#     try:
#         q_point = mp(k_point)
#         selected_mode = q_point.bands[band_index]
#     except ValueError:
#         raise ValueError(f"k-point {k_point} not found on the commensurate grid.")
#     except IndexError:
#         n_bands = len(q_point.bands)
#         raise IndexError(
#             f"Band index {band_index} is out of bounds. "
#             f"There are {n_bands} bands at this k-point (indices 0 to {n_bands-1})."
#         )
#     supercell = mp.supercell.to_ase()
#     n_atoms_supercell = len(supercell)
#     atomic_numbers = supercell.get_atomic_numbers()
#     masses = supercell.get_masses()
#     cell = supercell.get_cell()

#     positions_array = np.zeros((n_frames, n_atoms_supercell, 3))
#     amplitudes = np.linspace(-max_amplitude, max_amplitude, n_frames)

#     for i, amp in enumerate(amplitudes):
#         selected_mode.Q.amplitude = amp
#         displaced_atoms = mp.get_atoms()
#         positions_array[i] = displaced_atoms.get_positions()

#     trajectory = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=atomic_numbers,
#         masses=masses,
#         cell_vectors=cell,
#         periodic=True,
#         n_atoms=n_atoms_supercell,
#         n_frames=n_frames
#     )
#     return trajectory





