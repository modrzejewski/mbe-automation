from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal
import math
from collections import deque
import numpy as np
import numpy.typing as npt
import ase.geometry
import ase.io
import ase.build
import ase.spacegroup.symmetrize
import ase.spacegroup.utils
from ase import Atoms, neighborlist
from ase.neighborlist import natural_cutoffs, build_neighbor_list
from scipy.sparse.csgraph import connected_components
from scipy import sparse
import scipy

import mbe_automation.storage

def Label(Constituents, NMonomers):
    d = math.ceil(math.log(NMonomers, 10))
    prefixes = {1:"monomer", 2:"dimer", 3:"trimer", 4:"tetramer"}
    Label = prefixes[len(Constituents)] + "-" + "-".join([str(i).zfill(d) for i in Constituents])
    return Label


def IntermolecularDistance(MolA, MolB):
    posA = MolA.get_positions()
    posB = MolB.get_positions()
    Rij = scipy.spatial.distance.cdist(posA, posB)
    MinRij = np.min(Rij)
    MaxRij = np.max(Rij)
    AvRij = np.mean(Rij)
    return MinRij, AvRij, MaxRij


def WriteClusterXYZ(FilePath, Constituents, Monomers):
    #
    # Write an xyz file with a comment line (i.e., the second line in the file)
    # which specifies the number of atoms in each monomer.
    #
    ClusterSize = len(Constituents)
    N = [len(Monomers[i]) for i in Constituents]
    if ClusterSize == 1:
        s = ""
    else:
        s = " ".join(str(i) for i in N)
        
    xyz = open(FilePath, "w")
    xyz.write(f"{sum(N)}\n")
    xyz.write(f"{s}\n")
    for i in Constituents:
        M = Monomers[i]
        for element, (x, y, z) in zip(M.symbols, M.positions):
            xyz.write(f"{element:6} {x:16.8f} {y:16.8f} {z:16.8f} \n")
    xyz.write("\n")
    xyz.close()


def GhostAtoms(Monomers, MinRij, Reference, MonomersWithinCutoff, Cutoffs):
    if Cutoffs["ghosts"] >= Cutoffs["dimers"]:
        print(f"Cutoff for ghosts ({Cutoffs['ghosts']} Å) must be smaller than cutoff for dimers ({Cutoffs['dimers']} Å)")
        sys.exit(1)

    Ghosts = Atoms()
    Rmax = Cutoffs["ghosts"]
    posA = Monomers[Reference].get_positions()
    for M in MonomersWithinCutoff["dimers"]:
        MonomerB = Monomers[M]
        if MinRij[Reference, M] < Rmax:
            posB = MonomerB.get_positions()
            Rij = scipy.spatial.distance.cdist(posA, posB)
            columns_below_cutoff = np.where(np.any(Rij < Rmax, axis=0))[0]
            selected_atoms = MonomerB[columns_below_cutoff]
            Ghosts.extend(selected_atoms)
            
    return Ghosts


def extract_molecules(UnitCell, Na=1, Nb=1, Nc=1):
    """
    Extract a list of molecules for which all atoms are within
    the Na x Nb x Nc supercell. The molecules for which any covalent
    bond goes through the boundary of the supercell are discarded.

    """
    Supercell = ase.build.make_supercell(UnitCell, np.diag(np.array([Na, Nb, Nc])))
    BondCutoffs = neighborlist.natural_cutoffs(Supercell)
    NeighborList = neighborlist.build_neighbor_list(Supercell, BondCutoffs)
    ConnectivityMatrix = NeighborList.get_connectivity_matrix(sparse=True)
    NMolecules, MoleculeIndices = sparse.csgraph.connected_components(ConnectivityMatrix)
    NAtoms = len(MoleculeIndices)
    print(f"Supercell: {Na}×{Nb}×{Nc}, includes {NAtoms} atoms")
    print("Supercell vectors")
    for q in range(3):
        x, y, z = Supercell.cell[q]
        v = ("a", "b", "c")[q]
        print(f"{v} = [{x:10.3f}, {y:10.3f}, {z:10.3f}]")
    #
    # Find molecules for which no bond goes through
    # the boundary of the supercell
    #
    Monomers = []
    for m in range(NMolecules):
        ConstituentAtoms = np.where(MoleculeIndices == m)[0]
        #
        # If any bond goes through the boundary of the supercell,
        # this will be evident from nonzero vector Offsets,
        # which contains the lattice vector multipliers
        #
        WholeMolecule = True
        for a in ConstituentAtoms:
            Neighbors, Offsets = NeighborList.get_neighbors(a)
            for x in Offsets:
                if np.count_nonzero(x) > 0:
                    WholeMolecule = False
                    break
            if not WholeMolecule:
                break
        if WholeMolecule:
            Molecule = Atoms()
            for a in ConstituentAtoms:
                Molecule.append(Supercell[a])
            Monomers.append(Molecule)
                
    print(f"{len(Monomers)} monomers with all atoms within the supercell")
    return Monomers


def GetSupercellDimensions(UnitCell, SupercellRadius):
    #
    #     Determine the size of the supercell according to
    #     the SupercellRadius parameter.
    #
    #     SupercellRadius is the maximum intermolecular distance, R,
    #     measured between a molecule in the central unit cell and
    #     any molecule belonging to the supercell.
    #
    #     The supercell dimension Na x Nb x Nc is automatically determined
    #     so that the following inequality is satisfied
    #
    #     Dq > Hq + 2 * R
    #
    #     where
    #            Dq is the height of the supercell in the qth direction
    #            Hq is the height of the unit cell in the qth direction
    #             R is the maximum intermolecular distance 
    #
    #     In other words, R is the thickness of the layer of cells
    #     added to the central unit cell in order to build the supercell.
    #
    LatticeVectors = UnitCell.get_cell()
    Volume = UnitCell.cell.volume
    #
    # Extra layer of cells to prevent the worst-case scenario: the outermost molecule
    # is within the cutoff radius R (defined as the minimum interatomic distance),
    # but the covalent bonds go through the boundary of the cell.
    #
    Delta = 1
    N = [0, 0, 0]
    for i in range(3):
        axb = np.cross(LatticeVectors[(i + 1) % 3, :], LatticeVectors[(i + 2) % 3, :])
        #
        # Volume of a parallelepiped = ||a x b|| ||c|| |Cos(gamma)|
        # Here, h is the height in the a x b direction
        #
        h = Volume / np.linalg.norm(axb)
        N[i] = 2 * (math.ceil(SupercellRadius / h)+Delta) + 1
        
    return N[0], N[1], N[2]


def _test_identical_composition(
    system: mbe_automation.storage.Structure, 
    index_map: List[npt.NDArray[np.integer]]
) -> bool:
    """
    Tests if all molecules have the same elemental composition using np.bincount.

    Args:
        system: The structure object containing atomic numbers for all atoms.
        index_map: A list where each element is a NumPy array of atom indices
                   for a single identified molecule.

    Returns:
        True if all molecules have identical elemental composition, False otherwise.
    """
    if len(index_map) <= 1:
        return True

    all_atomic_numbers = system.atomic_numbers
    reference_indices = index_map[0]
    max_z = np.max(all_atomic_numbers)
    reference_composition = np.bincount(all_atomic_numbers[reference_indices], minlength=max_z + 1)

    for i in range(1, len(index_map)):
        current_indices = index_map[i]
        
        if len(current_indices) != len(reference_indices):
            return False

        current_composition = np.bincount(all_atomic_numbers[current_indices], minlength=max_z + 1)
        if not np.array_equal(reference_composition, current_composition):
            return False

    return True


def detect_molecules(
        system: mbe_automation.storage.Structure,
        frame_index: int = 0,
        assert_identical_composition=True,
) -> mbe_automation.storage.Clustering:
    """
    Identify all molecules in a periodic system.
    
    (1) For a periodic system, shift the periodic images by the lattice
    vectors in such a way that the atoms are contiguous in space if they
    belong to the same molecule.

    (2) For the molecules from step (1), shift the molecular centers of mass
    to the inside of the original unit cell.

    """
    if not system.periodic:
        raise ValueError("extract_all_molecules is not designed for non-periodic systems")

    atoms = mbe_automation.storage.to_ase(
        structure=system,
        frame_index=frame_index
    )
    neighbor_list = build_neighbor_list(
        atoms,
        cutoffs=natural_cutoffs(atoms),
        bothways=True
    )
    adj_matrix = neighbor_list.get_connectivity_matrix()    
    n_molecules, molecule_labels = connected_components(adj_matrix)
    assert n_molecules > 0
    
    grouped_indices = [np.where(molecule_labels == i)[0] for i in range(n_molecules)]    
    contiguous_positions = atoms.positions.copy()
    processed_indices = set()
    
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    masses = atoms.get_masses()
    com_vectors = np.zeros((n_molecules, 3))

    for molecule_idx, indices in enumerate(grouped_indices):
        if len(indices) == 0 or indices[0] in processed_indices:
            continue
        #
        # Translate periodic images to make the atomic positions
        # within the molecule contiguous in space
        #
        q = deque()
        start_idx = indices[0]
        q.append(start_idx)
        processed_indices.add(start_idx)

        # We need to keep track of visited atoms within this molecule's traversal
        # because the global `processed_indices` is only for starting new traversals.
        visited_in_molecule = {start_idx}

        while q:
            current_idx = q.popleft()
            neighbors, offsets = neighbor_list.get_neighbors(current_idx)

            for neighbor_idx, offset in zip(neighbors, offsets):
                if neighbor_idx not in visited_in_molecule:
                    
                    visited_in_molecule.add(neighbor_idx)
                    processed_indices.add(neighbor_idx)
                    
                    offset_vec = offset @ cell
                    bond_vector = (atoms.positions[neighbor_idx] + offset_vec) - atoms.positions[current_idx]
                    contiguous_positions[neighbor_idx] = contiguous_positions[current_idx] + bond_vector
                    q.append(neighbor_idx)

        mol_masses = masses[indices]
        total_mol_mass = np.sum(mol_masses)
        com_vectors[molecule_idx] = np.sum(contiguous_positions[indices] * mol_masses[:, np.newaxis], axis=0) / total_mol_mass

    #
    # Translate molecular centers of mass to the interior of the original
    # unit cell
    #
    shifted_com_vectors = ase.geometry.wrap_positions(
        positions=com_vectors,
        cell=atoms.cell,
        pbc=atoms.pbc
    )
    delta_r = shifted_com_vectors - com_vectors
    for molecule_idx, indices in enumerate(grouped_indices):
        contiguous_positions[indices] += delta_r[molecule_idx][np.newaxis, :]
    #
    # Set the center of mass of the total system to (0, 0, 0)
    #
    total_system_com = np.sum(contiguous_positions * masses[:, np.newaxis], axis=0) / np.sum(masses)
    contiguous_positions -= total_system_com[np.newaxis, :]
    shifted_com_vectors -= total_system_com[np.newaxis, :]

    identical = _test_identical_composition(
        system,
        grouped_indices
    )
    if assert_identical_composition:
        assert identical, "Found molecules which differ in composition"
    if identical:
        grouped_indices = np.array(grouped_indices)

    total_system = system.copy()
    if system.n_frames == 1:
        total_system.positions = contiguous_positions
    else:
        delta_r = contiguous_positions - total_system.positions[frame_index]
        total_system.positions += delta_r[np.newaxis, :, :]

    com_distances_from_origin = np.linalg.norm(shifted_com_vectors, axis=1)
    central_molecule_index = int(np.argmin(com_distances_from_origin))
    ref_positions = contiguous_positions[grouped_indices[central_molecule_index]]
    min_distances = np.zeros(n_molecules)
    max_distances = np.zeros(n_molecules)
    for i in range(n_molecules):
        if i == central_molecule_index:
            min_distances[i] = 0.0
            max_distances[i] = 0.0
        else:
            neighbor_positions = contiguous_positions[grouped_indices[i]]
            pairwise_distances = scipy.spatial.distance.cdist(ref_positions, neighbor_positions)
            min_distances[i] = np.min(pairwise_distances) 
            max_distances[i] = np.max(pairwise_distances)
            
    return mbe_automation.storage.Clustering(
        supercell=total_system,
        index_map=grouped_indices,
        centers_of_mass=shifted_com_vectors,
        identical_composition=identical,
        n_molecules=n_molecules,
        central_molecule_index=central_molecule_index,
        min_distances_to_central_molecule=min_distances,
        max_distances_to_central_molecule=max_distances
    )


def extract_finite_subsystem(
        clustering: mbe_automation.storage.Clustering,
        filter: Literal[
            "closest_to_center_of_mass",
            "closest_to_central_molecule",
            "max_min_distance_to_central_molecule",
            "max_max_distance_to_central_molecule"
        ],
        n_molecules: int | None = None,
        distance: float | None = None,
) -> mbe_automation.storage.FiniteSubsystem:
    """
    Filter out a finite molecular cluster from a periodic structure.
    The molecules belonging to the cluster are chosen according
    to their distance from the origin of the coordinate system.

    For a structure with n_frames > 1, it is assumed that
    (1) the covalent bonds do not change between frames,
    (2) there is no permutation of atoms between frames.
    """

    if filter in ["closest_to_center_of_mass",
                     "closest_to_central_molecule"]:
        if not (n_molecules is not None and distance is None):
            raise ValueError("n_molecules must be set and distance must be None.")
    elif filter in ["max_min_distance_to_central_molecule",
                       "max_max_distance_to_central_molecule"]:
        if not (distance is not None and n_molecules is None):
            raise ValueError("distance must be set and n_molecules must be None.")

    if filter == "closest_to_center_of_mass":
        com_distances_from_origin = np.linalg.norm(clustering.centers_of_mass, axis=1)
        sorted_indices = np.argsort(com_distances_from_origin, stable=True)
        filtered_molecule_indices = sorted_indices[0:n_molecules]
    elif filter == "closest_to_central_molecule":
        sorted_indices = np.argsort(clustering.min_distances_to_central_molecule, stable=True)
        filtered_molecule_indices = sorted_indices[0:n_molecules]
    elif filter == "max_max_distance_to_central_molecule":
        mask = clustering.max_distances_to_central_molecule < distance
        filtered_molecule_indices = np.where(mask)[0]
    elif filter == "max_min_distance_to_central_molecule":
        mask = clustering.min_distances_to_central_molecule < distance            
        filtered_molecule_indices = np.where(mask)[0]
    else:
        raise ValueError(f"Invalid filter: {filter}")

    filtered_atom_indices = np.concatenate(
        [clustering.index_map[i] for i in filtered_molecule_indices]
    )
    if clustering.supercell.positions.ndim == 3:
        subsystem_pos = clustering.supercell.positions[:, filtered_atom_indices, :]
    elif clustering.supercell.positions.ndim == 2:
        subsystem_pos = clustering.supercell.positions[filtered_atom_indices, :]
    else:
        raise ValueError(f"Invalid rank of clustering.supercell.positions: {clustering.supercell.positions.ndim}")
        
    finite_subsystem = mbe_automation.storage.FiniteSubsystem(
        cluster_of_molecules=mbe_automation.storage.Structure(
            positions=subsystem_pos,
            atomic_numbers=clustering.supercell.atomic_numbers[filtered_atom_indices],
            masses=clustering.supercell.masses[filtered_atom_indices], 
            cell_vectors=None,
            n_frames=clustering.supercell.n_frames,
            n_atoms=len(filtered_atom_indices),
            periodic=False
        ),
        molecule_indices=filtered_molecule_indices,
        n_molecules=len(filtered_molecule_indices)
    )

    return finite_subsystem
