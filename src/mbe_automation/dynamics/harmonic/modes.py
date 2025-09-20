from __future__ import annotations
import numpy as np
import numpy.typing as npt
import mbe_automation.storage
import ase
import phonopy
from ase.neighborlist import natural_cutoffs, build_neighbor_list
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

def _get_modulated_displacements(
    ph: phonopy.Phonopy,
    eigvec: npt.NDArray[np.complex128],
    q: npt.NDArray[np.floating]
) -> npt.NDArray[np.complex128]:
    """
    Compute complex displacements for a supercell from a primitive cell eigenvector.
    """
    supercell = ph.supercell
    spos = supercell.scaled_positions
    n_atoms_supercell = len(supercell)
    
    # Create the composite mapping from supercell to primitive, as in phonopy
    s2u_map = supercell.s2u_map
    u2u_map = supercell.u2u_map
    s2uu_map = [u2u_map[x] for x in s2u_map]

    # Calculate phase factor and combine with mass normalization for each supercell atom
    scaled_pos_in_prim = np.dot(spos, supercell.supercell_matrix.T)
    phase_factors = np.exp(2j * np.pi * np.dot(scaled_pos_in_prim, q))
    s_masses = supercell.masses
    coefs = phase_factors / np.sqrt(s_masses)

    # Map primitive displacements to supercell atoms with correct coefficient
    u = np.zeros((n_atoms_supercell, 3), dtype=np.complex128)
    for i in range(n_atoms_supercell):
        p_index = s2uu_map[i]
        u[i] = eigvec[p_index*3 : p_index*3 + 3] * coefs[i]

    # Apply final normalization, as seen in phonopy.modulation
    u /= np.sqrt(n_atoms_supercell)
    return u


def _get_molecules(
    cell: phonopy.structure.atoms.PhonopyAtoms
) -> list[npt.NDArray[np.int64]]:
    """
    Return a list of arrays of indices of whole molecules inside a cell.
    """
    atoms = ase.Atoms(
        symbols=cell.symbols,
        positions=cell.positions,
        cell=cell.cell,
        pbc=True
    )
    n_atoms = len(atoms)
    
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = build_neighbor_list(atoms, cutoffs=cutoffs)
    
    # Build adjacency matrix using only bonds with zero offset (internal to cell)
    adj_matrix = lil_matrix((n_atoms, n_atoms), dtype=int)
    for i in range(n_atoms):
        neighbors, offsets = neighbor_list.get_neighbors(i)
        for j, offset in zip(neighbors, offsets):
            if np.all(offset == 0):
                adj_matrix[i, j] = 1
    
    # Find connected components based on internal bonds only
    n_molecules, labels = connected_components(adj_matrix.tocsr())
    
    molecules = []
    for i in range(n_molecules):
        molecules.append(np.where(labels == i)[0])
        
    return molecules


def _filter_closest_molecules(
    cell: phonopy.structure.atoms.PhonopyAtoms,
    molecules: list[npt.NDArray[np.int64]],
    n_molecules_to_keep: int
) -> npt.NDArray[np.int64]:
    """
    Filter out all atoms except those in N molecules closest to the cell center.
    """
    positions = cell.positions

    # First, compute the COM and mass for each individual molecule
    molecular_coms = []
    molecular_masses = []
    for mol_indices in molecules:
        molecule = ase.Atoms(
            symbols=np.array(cell.symbols)[mol_indices],
            positions=positions[mol_indices],
            masses=cell.masses[mol_indices]
        )
        molecular_coms.append(molecule.get_center_of_mass())
        molecular_masses.append(np.sum(molecule.get_masses()))
        
    molecular_coms = np.array(molecular_coms)
    molecular_masses = np.array(molecular_masses)

    # Second, compute the overall COM of all whole molecules
    total_mass = np.sum(molecular_masses)
    overall_com = np.sum(molecular_coms * molecular_masses[:, np.newaxis], axis=0) / total_mass
    
    # Third, compute distance of each molecule's COM to the overall COM
    com_distances = []
    for com in molecular_coms:
        dist = np.linalg.norm(com - overall_com)
        com_distances.append(dist)

    # Sort molecules by distance and get indices of the N closest
    sorted_mol_indices = np.argsort(com_distances)
    closest_mol_indices = sorted_mol_indices[:n_molecules_to_keep]

    # Flatten the list of atom indices for the selected molecules
    filtered_atom_indices = np.concatenate(
        [molecules[i] for i in closest_mol_indices]
    )
    
    return np.sort(filtered_atom_indices)


def trajectory(
        dataset: str,
        key: str,
        k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
        band_index: int = 0,
        max_amplitude: float = 1.0,
        n_frames: int = 10,
        use_supercell: bool = False,
        filter_molecules: int | None = None,
) -> mbe_automation.storage.Structure:
    """
    Generate a trajectory for a single vibrational mode.

    Args:
        dataset: Path to the dataset file.
        key: Key to the harmonic force constants model.
        k_point: Reduced coordinates of the k-point.
        band_index: Index of the vibrational mode.
        max_amplitude: Controls the maximum displacement of the mode.
        n_frames: Number of frames for the trajectory.
        use_supercell: If True, generate trajectory for the supercell.
                       Otherwise, use the primitive cell.
        filter_molecules: If not None, keep only the `m` molecules closest
                          to the center of mass.

    Returns:
        A Structure object containing the mode trajectory.
    """
    # --- Part 1: Common Setup ---
    ph = mbe_automation.storage.to_phonopy(
        dataset=dataset,
        key=key
    )
    n_bands = 3 * len(ph.primitive)
    if not (0 <= band_index < n_bands):
        raise IndexError(
            f"band_index must be on the interval [0, {n_bands - 1}]."
        )

    dm = ph.dynamical_matrix
    dm.run(k_point)
    eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)
    eigvec = eigenvectors[:, band_index]

    # --- Part 2: Cell-Specific Setup ---
    if use_supercell:
        base_cell = ph.supercell
        displacements = _get_modulated_displacements(ph, eigvec, k_point)
    else:
        base_cell = ph.primitive
        masses = base_cell.masses
        n_atoms_primitive = len(base_cell)
        displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)
        displacements /= np.sqrt(n_atoms_primitive)

    n_atoms = len(base_cell)
    base_positions = base_cell.positions
    
    # --- Part 3: Generate Full Trajectory ---
    positions_array = np.zeros((n_frames, n_atoms, 3))
    for i in range(n_frames):
        phase = 2j * np.pi * i / n_frames
        frame_displacement = (displacements * np.exp(phase)).imag * max_amplitude
        positions_array[i] = base_positions + frame_displacement

    full_trajectory = mbe_automation.storage.Structure(
        positions=positions_array,
        atomic_numbers=base_cell.numbers,
        masses=base_cell.masses,
        cell_vectors=base_cell.cell,
        n_atoms=n_atoms,
        n_frames=n_frames,
    )

    # --- Part 4: Apply Filtering Post-Generation (if requested) ---
    if filter_molecules is not None:
        molecules = _get_molecules(base_cell)
        indices_to_keep = _filter_closest_molecules(base_cell, molecules, filter_molecules)
        
        if len(indices_to_keep) == 0:
            raise ValueError(
                "Molecule filtering resulted in zero atoms. "
                "Check if `filter_molecules` is too small or if whole "
                "molecules were found in the cell."
            )
        
        filtered_trajectory = mbe_automation.storage.Structure(
            positions=full_trajectory.positions[:, indices_to_keep, :],
            atomic_numbers=full_trajectory.atomic_numbers[indices_to_keep],
            masses=full_trajectory.masses[indices_to_keep],
            cell_vectors=full_trajectory.cell_vectors,
            n_atoms=len(indices_to_keep),
            n_frames=full_trajectory.n_frames,
        )
        return filtered_trajectory
    else:
        return full_trajectory





# from __future__ import annotations
# import numpy as np
# import numpy.typing as npt
# import ase
# import phonopy

# import mbe_automation.storage


# def _get_modulated_displacements(
#     ph: phonopy.Phonopy,
#     eigvec: npt.NDArray[np.complex128],
#     q: npt.NDArray[np.floating]
# ) -> npt.NDArray[np.complex128]:
#     """
#     Compute complex displacements for a supercell from a primitive cell eigenvector.
#     """
#     supercell = ph.supercell
#     spos = supercell.scaled_positions
#     n_atoms_supercell = len(supercell)
    
#     # Create the composite mapping from supercell to primitive, as in phonopy
#     s2u_map = supercell.s2u_map
#     u2u_map = supercell.u2u_map
#     s2uu_map = [u2u_map[x] for x in s2u_map]

#     # Calculate phase factor and combine with mass normalization for each supercell atom
#     scaled_pos_in_prim = np.dot(spos, supercell.supercell_matrix.T)
#     phase_factors = np.exp(2j * np.pi * np.dot(scaled_pos_in_prim, q))
#     s_masses = supercell.masses
#     coefs = phase_factors / np.sqrt(s_masses)

#     # Map primitive displacements to supercell atoms with correct coefficient
#     u = np.zeros((n_atoms_supercell, 3), dtype=np.complex128)
#     for i in range(n_atoms_supercell):
#         p_index = s2uu_map[i]
#         u[i] = eigvec[p_index*3 : p_index*3 + 3] * coefs[i]

#     # Apply final normalization, as seen in phonopy.modulation
#     u /= np.sqrt(n_atoms_supercell)
#     return u


# def trajectory(
#         dataset: str,
#         key: str,
#         k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
#         band_index: int = 0,
#         max_amplitude: float = 1.0,
#         n_frames: int = 10,
#         use_supercell: bool = False,
# ) -> mbe_automation.storage.Structure:
#     """
#     Generate a trajectory for a single vibrational mode.

#     Args:
#         dataset: Path to the dataset file.
#         key: Key to the harmonic force constants model.
#         k_point: Reduced coordinates of the k-point.
#         band_index: Index of the vibrational mode.
#         max_amplitude: Controls the maximum displacement of the mode.
#         n_frames: Number of frames for the trajectory.
#         use_supercell: If True, generate trajectory for the supercell.
#                        Otherwise, use the primitive cell.

#     Returns:
#         A Structure object containing the mode trajectory.
#     """
#     # --- Part 1: Common Setup ---
#     ph = mbe_automation.storage.to_phonopy(
#         dataset=dataset,
#         key=key
#     )
#     n_bands = 3 * len(ph.primitive)
#     if not (0 <= band_index < n_bands):
#         raise IndexError(
#             f"band_index must be on the interval [0, {n_bands - 1}]."
#         )

#     dm = ph.dynamical_matrix
#     dm.run(k_point)
#     eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)
#     eigvec = eigenvectors[:, band_index]

#     # --- Part 2: Cell-Specific Setup ---
#     if use_supercell:
#         base_cell = ph.supercell
#         displacements = _get_modulated_displacements(ph, eigvec, k_point)
#     else:
#         base_cell = ph.primitive
#         masses = base_cell.masses
#         n_atoms_primitive = len(base_cell)
#         displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-it, 3)
#         displacements /= np.sqrt(n_atoms_primitive)

#     n_atoms = len(base_cell)
#     base_positions = base_cell.positions
    
#     # --- Part 3: Generate Trajectory ---
#     positions_array = np.zeros((n_frames, n_atoms, 3))
#     for i in range(n_frames):
#         phase = 2j * np.pi * i / n_frames
#         frame_displacement = (displacements * np.exp(phase)).imag * max_amplitude
#         positions_array[i] = base_positions + frame_displacement

#     # --- Part 4: Create Structure Object ---
#     trajectory = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=base_cell.numbers,
#         masses=base_cell.masses,
#         cell_vectors=base_cell.cell,
#         n_atoms=n_atoms,
#         n_frames=n_frames,
#     )
#     return trajectory







# def _get_modulated_displacements(
#     ph: phonopy.Phonopy,
#     eigvec: npt.NDArray[np.complex128],
#     q: npt.NDArray[np.floating]
# ) -> npt.NDArray[np.complex128]:
#     """
#     Compute complex displacements for a supercell from a primitive cell eigenvector.
#     """
#     supercell = ph.supercell
#     primitive = ph.primitive
#     s2p_map = primitive.s2p_map
#     p2p_map = primitive.p2p_map
#     spos = supercell.scaled_positions
#     n_atoms_supercell = len(supercell)
    
#     # Calculate phase factor and combine with mass normalization for each supercell atom
#     scaled_pos_in_prim = np.dot(spos, supercell.supercell_matrix.T)
#     phase_factors = np.exp(2j * np.pi * np.dot(scaled_pos_in_prim, q))
#     s_masses = supercell.masses
#     coefs = phase_factors / np.sqrt(s_masses)

#     # Map primitive displacements to supercell atoms with correct coefficient
#     u = np.zeros((n_atoms_supercell, 3), dtype=np.complex128)
#     for i in range(n_atoms_supercell):
#         p_supercell_index = s2p_map[i]
#         p_index = p2p_map[p_supercell_index]
#         u[i] = eigvec[p_index*3 : p_index*3 + 3] * coefs[i]

#     # Apply final normalization, as seen in phonopy.modulation
#     u /= np.sqrt(n_atoms_supercell)
#     return u


# def trajectory(
#         dataset: str,
#         key: str,
#         k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
#         band_index: int = 0,
#         max_amplitude: float = 1.0,
#         n_frames: int = 10,
#         use_supercell: bool = False,
# ) -> mbe_automation.storage.Structure:
#     """
#     Generate a trajectory for a single vibrational mode.

#     Args:
#         dataset: Path to the dataset file.
#         key: Key to the harmonic force constants model.
#         k_point: Reduced coordinates of the k-point.
#         band_index: Index of the vibrational mode.
#         max_amplitude: Controls the maximum displacement of the mode.
#         n_frames: Number of frames for the trajectory.
#         use_supercell: If True, generate trajectory for the supercell.
#                        Otherwise, use the primitive cell.

#     Returns:
#         A Structure object containing the mode trajectory.
#     """
#     # --- Part 1: Common Setup ---
#     ph = mbe_automation.storage.to_phonopy(
#         dataset=dataset,
#         key=key
#     )
#     n_bands = 3 * len(ph.primitive)
#     if not (0 <= band_index < n_bands):
#         raise IndexError(
#             f"band_index must be on the interval [0, {n_bands - 1}]."
#         )

#     dm = ph.dynamical_matrix
#     dm.run(k_point)
#     eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)
#     eigvec = eigenvectors[:, band_index]

#     # --- Part 2: Cell-Specific Setup ---
#     if use_supercell:
#         base_cell = ph.supercell
#         displacements = _get_modulated_displacements(ph, eigvec, k_point)
#     else:
#         base_cell = ph.primitive
#         masses = base_cell.masses
#         n_atoms_primitive = len(base_cell)
#         displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)
#         displacements /= np.sqrt(n_atoms_primitive)

#     n_atoms = len(base_cell)
#     base_positions = base_cell.positions
    
#     # --- Part 3: Generate Trajectory ---
#     positions_array = np.zeros((n_frames, n_atoms, 3))
#     for i in range(n_frames):
#         phase = 2j * np.pi * i / n_frames
#         frame_displacement = (displacements * np.exp(phase)).imag * max_amplitude
#         positions_array[i] = base_positions + frame_displacement

#     # --- Part 4: Create Structure Object ---
#     trajectory = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=base_cell.numbers,
#         masses=base_cell.masses,
#         cell_vectors=base_cell.cell,
#         n_atoms=n_atoms,
#         n_frames=n_frames,
#     )
#     return trajectory





# def _get_modulated_displacements(
#     ph: phonopy.Phonopy,
#     eigvec: npt.NDArray[np.complex128],
#     q: npt.NDArray[np.floating]
# ) -> npt.NDArray[np.complex128]:
#     """
#     Compute complex displacements for a supercell from a primitive cell eigenvector.
#     """
#     supercell = ph.supercell
#     primitive = ph.primitive
#     s2p_map = primitive.s2p_map
#     spos = supercell.scaled_positions
#     n_atoms_supercell = len(supercell)
    
#     # Calculate phase factor and combine with mass normalization for each supercell atom
#     scaled_pos_in_prim = np.dot(spos, supercell.supercell_matrix.T)
#     phase_factors = np.exp(2j * np.pi * np.dot(scaled_pos_in_prim, q))
#     s_masses = supercell.masses
#     coefs = phase_factors / np.sqrt(s_masses)

#     # Map primitive displacements to supercell atoms with correct coefficient
#     u = np.zeros((n_atoms_supercell, 3), dtype=np.complex128)
#     for i in range(n_atoms_supercell):
#         p_index = s2p_map[i]
#         u[i] = eigvec[p_index*3 : p_index*3 + 3] * coefs[i]

#     # Apply final normalization, as seen in phonopy.modulation
#     u /= np.sqrt(n_atoms_supercell)
#     return u


# def trajectory(
#         dataset: str,
#         key: str,
#         k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
#         band_index: int = 0,
#         max_amplitude: float = 1.0,
#         n_frames: int = 10,
#         use_supercell: bool = False,
# ) -> mbe_automation.storage.Structure:
#     """
#     Generate a trajectory for a single vibrational mode.

#     Args:
#         dataset: Path to the dataset file.
#         key: Key to the harmonic force constants model.
#         k_point: Reduced coordinates of the k-point.
#         band_index: Index of the vibrational mode.
#         max_amplitude: Controls the maximum displacement of the mode.
#         n_frames: Number of frames for the trajectory.
#         use_supercell: If True, generate trajectory for the supercell.
#                        Otherwise, use the primitive cell.

#     Returns:
#         A Structure object containing the mode trajectory.
#     """
#     # --- Part 1: Common Setup ---
#     ph = mbe_automation.storage.to_phonopy(
#         dataset=dataset,
#         key=key
#     )
#     n_bands = 3 * len(ph.primitive)
#     if not (0 <= band_index < n_bands):
#         raise IndexError(
#             f"band_index must be on the interval [0, {n_bands - 1}]."
#         )

#     dm = ph.dynamical_matrix
#     dm.run(k_point)
#     eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)
#     eigvec = eigenvectors[:, band_index]

#     # --- Part 2: Cell-Specific Setup ---
#     if use_supercell:
#         base_cell = ph.supercell
#         displacements = _get_modulated_displacements(ph, eigvec, k_point)
#     else:
#         base_cell = ph.primitive
#         masses = base_cell.masses
#         n_atoms_primitive = len(base_cell)
#         displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)
#         displacements /= np.sqrt(n_atoms_primitive)

#     n_atoms = len(base_cell)
#     base_positions = base_cell.positions
    
#     # --- Part 3: Generate Trajectory ---
#     positions_array = np.zeros((n_frames, n_atoms, 3))
#     for i in range(n_frames):
#         phase = 2j * np.pi * i / n_frames
#         frame_displacement = (displacements * np.exp(phase)).imag * max_amplitude
#         positions_array[i] = base_positions + frame_displacement

#     # --- Part 4: Create Structure Object ---
#     trajectory = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=base_cell.numbers,
#         masses=base_cell.masses,
#         cell_vectors=base_cell.cell,
#         n_atoms=n_atoms,
#         n_frames=n_frames,
#     )
#     return trajectory





# def _get_modulated_displacements(
#     ph: phonopy.Phonopy,
#     eigvec: npt.NDArray[np.complex128],
#     q: npt.NDArray[np.floating]
# ) -> npt.NDArray[np.complex128]:
#     """
#     Compute complex displacements for a supercell from a primitive cell eigenvector.
#     """
#     supercell = ph.supercell
#     primitive = ph.primitive
#     s2p_map = primitive.s2p_map
#     spos = supercell.scaled_positions
#     n_atoms_supercell = len(supercell)
    
#     # Calculate phase factor for each atom in the supercell
#     scaled_pos_in_prim = np.dot(spos, supercell.supercell_matrix.T)
#     phase_factors = np.exp(2j * np.pi * np.dot(scaled_pos_in_prim, q))

#     # Mass-normalized primitive eigenvector
#     p_masses = primitive.masses
#     norm_eigvec = eigvec / np.sqrt(np.repeat(p_masses, 3))

#     # Map primitive displacements to supercell atoms with phase factor
#     u = np.zeros((n_atoms_supercell, 3), dtype=np.complex128)
#     for i in range(n_atoms_supercell):
#         p_index = s2p_map[i]
#         u[i] = norm_eigvec[p_index*3 : p_index*3 + 3] * phase_factors[i]

#     return u


# def trajectory(
#         dataset: str,
#         key: str,
#         k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
#         band_index: int = 0,
#         max_amplitude: float = 1.0,
#         n_frames: int = 10,
#         use_supercell: bool = False,
# ) -> mbe_automation.storage.Structure:
#     """
#     Generate a trajectory for a single vibrational mode.

#     Args:
#         dataset: Path to the dataset file.
#         key: Key to the harmonic force constants model.
#         k_point: Reduced coordinates of the k-point.
#         band_index: Index of the vibrational mode.
#         max_amplitude: Controls the maximum displacement of the mode.
#         n_frames: Number of frames for the trajectory.
#         use_supercell: If True, generate trajectory for the supercell.
#                        Otherwise, use the primitive cell.

#     Returns:
#         A Structure object containing the mode trajectory.
#     """
#     # --- Part 1: Common Setup ---
#     ph = mbe_automation.storage.to_phonopy(
#         dataset=dataset,
#         key=key
#     )
#     n_bands = 3 * len(ph.primitive)
#     if not (0 <= band_index < n_bands):
#         raise IndexError(
#             f"band_index must be on the interval [0, {n_bands - 1}]."
#         )

#     dm = ph.dynamical_matrix
#     dm.run(k_point)
#     eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)
#     eigvec = eigenvectors[:, band_index]

#     # --- Part 2: Cell-Specific Setup ---
#     if use_supercell:
#         base_cell = ph.supercell
#         displacements = _get_modulated_displacements(ph, eigvec, k_point)
#     else:
#         base_cell = ph.primitive
#         masses = base_cell.masses
#         displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)

#     n_atoms = len(base_cell)
#     base_positions = base_cell.positions
    
#     # --- Part 3: Generate Trajectory ---
#     positions_array = np.zeros((n_frames, n_atoms, 3))
#     for i in range(n_frames):
#         phase = 2j * np.pi * i / n_frames
#         frame_displacement = (displacements * np.exp(phase)).imag * max_amplitude
#         positions_array[i] = base_positions + frame_displacement

#     # --- Part 4: Create Structure Object ---
#     trajectory = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=base_cell.numbers,
#         masses=base_cell.masses,
#         cell_vectors=base_cell.cell,
#         n_atoms=n_atoms,
#         n_frames=n_frames,
#     )
#     return trajectory





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
#     Generate a primitive cell trajectory for a single vibrational mode.

#     Args:
#         dataset: Path to the dataset file.
#         key: Key to the harmonic force constants model.
#         k_point: Reduced coordinates of the k-point.
#         band_index: Index of the vibrational mode.
#         max_amplitude: Controls the maximum displacement of the mode.
#         n_frames: Number of frames for the trajectory.
#         wrap: If True, wrap atomic coordinates back into the primary
#               primitive cell.

#     Returns:
#         A Structure object containing the mode trajectory for the primitive cell.
#     """
#     ph = mbe_automation.storage.to_phonopy(
#         dataset=dataset,
#         key=key
#     )

#     primitive_cell = ph.primitive
#     n_bands = 3 * len(primitive_cell)
#     if not (0 <= band_index < n_bands):
#         raise IndexError(
#             f"band_index must be on the interval [0, {n_bands - 1}]."
#         )

#     # 1. Run dynamical matrix and get eigenvectors for the primitive cell
#     dm = ph.dynamical_matrix
#     dm.run(k_point)
#     eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)

#     # 2. Calculate mass-normalized displacements for the primitive cell
#     masses = primitive_cell.masses
#     eigvec = eigenvectors[:, band_index]
#     displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)

#     # 3. Generate frames using the sinusoidal motion formula for the primitive cell
#     equilibrium_positions = primitive_cell.positions
#     positions_array = np.zeros((n_frames, len(primitive_cell), 3))

#     for i in range(n_frames):
#         phase = 2j * np.pi * i / n_frames
#         frame_displacement = (displacements * np.exp(phase)).imag * max_amplitude
#         positions_array[i] = equilibrium_positions + frame_displacement

#     if wrap:
#         lattice = primitive_cell.cell
#         inv_lattice = np.linalg.inv(lattice)
#         flat_pos = positions_array.reshape(-1, 3)
#         frac_pos = np.dot(flat_pos, inv_lattice)
#         wrapped_frac_pos = frac_pos - np.floor(frac_pos)
#         wrapped_cart_pos = np.dot(wrapped_frac_pos, lattice)
#         positions_array = wrapped_cart_pos.reshape(n_frames, -1, 3)

#     trajectory = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=primitive_cell.numbers,
#         masses=primitive_cell.masses,
#         cell_vectors=primitive_cell.cell,
#         n_atoms=len(primitive_cell),
#         n_frames=n_frames,
#     )
#     return trajectory


