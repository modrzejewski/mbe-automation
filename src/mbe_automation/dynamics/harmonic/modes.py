from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import ase
import phonopy
from ase.neighborlist import natural_cutoffs, build_neighbor_list
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from ase.calculators.calculator import Calculator as ASECalculator

import mbe_automation.storage

@dataclass
class Mode:
    """
    Results of a vibrational mode analysis.
    
    Attributes:
        trajectory: The generated Structure object containing atomic coordinates.
        scan_coordinates: Phase angle for 'cyclic' sampling and amplitude for 'linear' sampling.
        potential_energies: Energy for unit cell/supercell/molecular cluster (eV)
    """
    trajectory: mbe_automation.storage.Structure
    scan_coordinates: npt.NDArray
    potential_energies: npt.NDArray | None
    

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
        extract_molecular_cluster: int | None = None,
        sampling: Literal["cyclic", "linear"] = "cyclic",
        calculator: ASECalculator | None = None
) -> Mode:
    """
    Generate a trajectory for a single vibrational mode and optionally
    calculate its potential energy curve.

    Args:
        dataset: Path to the dataset file.
        key: Key to the harmonic force constants model.
        k_point: Reduced coordinates of the k-point.
        band_index: Index of the vibrational mode.
        max_amplitude: Controls the maximum displacement of the mode.
        n_frames: Number of frames for the trajectory.
        use_supercell: If True, generate trajectory for the supercell.
        Otherwise, use the primitive cell.
        extract_molecular_cluster: If not None, keep only the `m` molecules closest
        to the center of mass.
        sampling: "cyclic" for animation, "linear" for PES scan.
        calculator: If not None, calculate the potential energy curve.

    Returns:
        A Mode object containing the mode trajectory and, optionally,
        the potential energy at each frame of the trajectory.
    """

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
    scan_coordinates = np.zeros(n_frames)
    potential_energies = []
    #
    # Perform scan in the direction of the selected normal coordinate.
    # This generates the trajectory and optionally the energy points.
    #
    positions_array = np.zeros((n_frames, n_atoms, 3))
    for i in range(n_frames):
        if sampling == "cyclic":
            phase = 2 * np.pi * i / n_frames
            frame_displacement = (displacements * np.exp(1j * phase)).imag * max_amplitude
            scan_coordinates[i] = phase
        elif sampling == "linear":
            amplitude = np.linspace(-max_amplitude, max_amplitude, n_frames)[i]
            frame_displacement = displacements.real * amplitude
            scan_coordinates[i] = amplitude
        else:
            raise ValueError(f"Unknown sampling type: {sampling}")

        current_positions = base_positions + frame_displacement
        positions_array[i] = current_positions

    traj = mbe_automation.storage.Structure(
        positions=positions_array,
        atomic_numbers=base_cell.numbers,
        masses=base_cell.masses,
        cell_vectors=base_cell.cell,
        n_atoms=n_atoms,
        n_frames=n_frames,
    )
    #
    # Filter out everything except for a cluster of n molecules
    # near the center of mass. This converts the structure to
    # finite (non-periodic) system.
    #
    if extract_molecular_cluster is not None:
        n_molecules = extract_molecular_cluster
        molecules = _get_molecules(base_cell)
        indices_to_keep = _filter_closest_molecules(base_cell, molecules, n_molecules)
        
        if len(indices_to_keep) == 0:
            raise ValueError(
                "Molecule filtering resulted in zero atoms."
            )
        
        traj = mbe_automation.storage.Structure(
            positions=traj.positions[:, indices_to_keep, :],
            atomic_numbers=traj.atomic_numbers[indices_to_keep],
            masses=traj.masses[indices_to_keep],
            #
            # The system is no longer considered as periodic
            # after we extract a cluster of molecules
            #
            cell_vectors=None, 
            n_atoms=len(indices_to_keep),
            n_frames=traj.n_frames,
        )

    if calculator is not None:
        potential_energies = []
        for i in range(traj.n_frames):
            single_frame = mbe_automation.storage.to_ase_atoms(
                structure=traj,
                frame_index=i
            )
            single_frame.calc = calculator
            potential_energies.append(single_frame.get_potential_energy())
            
        potential_energies = np.array(potential_energies)
        
    else:
        potential_energies = None

    return Mode(
        trajectory=traj,
        scan_coordinates=scan_coordinates,
        potential_energies=potential_energies
    )
