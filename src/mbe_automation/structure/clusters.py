from __future__ import annotations
from dataclasses import dataclass, field
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
import networkx
import pymatgen
import pymatgen.analysis
import pymatgen.analysis.local_env
from pymatgen.analysis.local_env import NearNeighbors, CutOffDictNN
import pymatgen.analysis.graphs

import mbe_automation.storage
import mbe_automation.structure.crystal
import mbe_automation.common.display

NUMBER_SELECTION = [
    "closest_to_center_of_mass",
    "closest_to_central_molecule"
]
DISTANCE_SELECTION = [
    "max_min_distance_to_central_molecule",
    "max_max_distance_to_central_molecule"
]

@dataclass(kw_only=True)
class FiniteSubsystemFilter:
                                   # ------------------------------------------------------------------------
                                   # Filter used to select molecules          Size parameter, which controls
                                   # from a PBC structure to create           how many molecules to include
                                   # a finite cluster                 
                                   # ------------------------------------------------------------------------
                                   # closest_to_center_of_mass,               n_molecules      
                                   # closest_to_central_molecule
                                   #
                                   # max_min_distance_to_central_molecule     distances
                                   # max_max_distance_to_central_molecule
                                   #
    selection_rule: Literal[
        *DISTANCE_SELECTION,
        *NUMBER_SELECTION
    ] = "closest_to_central_molecule"
    
    n_molecules: npt.NDArray[np.integer] | None = field(
        default_factory=lambda: np.array([1, 2, 3, 4, 5, 6, 7, 8])
    )
    
    distances: npt.NDArray[np.floating] | None  = None
                                   #
                                   # Assert that all molecules in the PBC structure
                                   # have identical elemental composition.
                                   #
                                   # Used only for validation during the clustering
                                   # step. Setting this parameter to False disables
                                   # the sanity check.
                                   #
    assert_identical_composition: bool = True

    def __post_init__(self):
        if self.selection_rule in NUMBER_SELECTION:
            if not (self.n_molecules is not None and self.distances is None):
                raise ValueError("n_molecules must be set and distance must be None.")
            
        elif self.selection_rule in DISTANCE_SELECTION:
            if not (self.distances is not None and self.n_molecules is None):
                raise ValueError("distance must be set and n_molecules must be None.")
            
        else:
            raise ValueError(f"Invalid selection_rule: {self.selection_rule}")


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
        reference_frame_index: int = 0,
        assert_identical_composition: bool = True,
        bonding_algo: NearNeighbors=CutOffDictNN.from_preset("vesta_2019"),
        validate_pbc_structure: bool = False
) -> mbe_automation.storage.MolecularCrystal:
    """
    Identify molecules in a periodic Structure.
    """
    mbe_automation.common.display.framed("Molecule detection")
    print(f"n_frames                {system.n_frames}")
    print(f"reference_frame_index   {reference_frame_index}")
    
    if not system.periodic:
        raise ValueError("detect_molecules is designed for periodic systems.")

    if system.positions.ndim == 3:
        positions_ref = system.positions[reference_frame_index]
    else:
        positions_ref = system.positions

    if system.cell_vectors.ndim == 3:
        unit_cell_vectors = system.cell_vectors[reference_frame_index]
    else:
        unit_cell_vectors = system.cell_vectors
        
    n_atoms_unit_cell = system.n_atoms

    supercell = pymatgen.core.Structure(
        lattice=unit_cell_vectors,
        species=system.atomic_numbers,
        coords=positions_ref,
        coords_are_cartesian=True,
        site_properties={"original_index": np.arange(n_atoms_unit_cell)}
    ).make_supercell([3, 3, 3])
    supercell_to_unit_cell = np.array(supercell.site_properties["original_index"])

    print("Computing graph of covalent bonds...", flush=True)
    structure_graph = pymatgen.analysis.graphs.StructureGraph.from_local_env_strategy(
        structure=supercell,
        strategy=bonding_algo
    )
    print("Graph completed", flush=True)
    
    components = list(networkx.weakly_connected_components(structure_graph.graph))
    masses = np.array([site.specie.atomic_mass for site in supercell.sites])
    scaled_positions = supercell.frac_coords
    
    supercell_subset = []
    n_atoms_found = 0
    for component in components:
        atom_indices = np.array(list(component))        
        subgraph = structure_graph.graph.subgraph(atom_indices)
        if any(d["to_jimage"] != (0, 0, 0) for _, _, d in subgraph.edges(data=True)):
            contiguous = False
        else:
            contiguous = True
        if contiguous:
            mol_positions = scaled_positions[atom_indices]
            mol_masses = masses[atom_indices]
            com_scaled = np.sum(mol_positions * mol_masses[:, np.newaxis], axis=0) / np.sum(mol_masses)
            if np.all(com_scaled >= 1.0/3.0) and np.all(com_scaled < 2.0/3.0):
                supercell_subset.append(atom_indices)
                n_atoms_found += len(atom_indices)
    
    assert n_atoms_found == n_atoms_unit_cell

    grouped_indices = []
    for x in supercell_subset:
        grouped_indices.append(supercell_to_unit_cell[x])
    supercell_subset = np.concatenate(supercell_subset)

    positions_ref_unwrapped = np.zeros((n_atoms_unit_cell, 3))
    positions_ref_unwrapped[supercell_to_unit_cell[supercell_subset]] = supercell.cart_coords[supercell_subset]
    com = (np.sum(positions_ref_unwrapped * system.masses[:, np.newaxis], axis=0)
           / np.sum(system.masses))
    positions_ref_unwrapped -= com[np.newaxis, :]
    shifts_cart = positions_ref_unwrapped - positions_ref
    shifts_frac = shifts_cart @ np.linalg.inv(unit_cell_vectors)

    system_unwrapped = system.copy()
    if system.n_frames > 1:
        for i in range(system.n_frames):
            if system.cell_vectors.ndim == 3:
                #
                # Cell vectors are frame-dependent:
                # NPT simulation
                #
                cell_i = system.cell_vectors[i]
            else:
                #
                # Cell vectors are frame-independent:
                # NVT simulation, phonon sampling
                #
                cell_i = system.cell_vectors
            unwrapped_cart_i = (system.positions[i] @ np.linalg.inv(cell_i) + shifts_frac) @ cell_i
            system_unwrapped.positions[i] = unwrapped_cart_i
    else:        
        system_unwrapped.positions = positions_ref_unwrapped
        
    n_molecules_unit_cell = len(grouped_indices)
    centers_of_mass = np.zeros((n_molecules_unit_cell, 3))
    for i in range(n_molecules_unit_cell):
        r = positions_ref_unwrapped[grouped_indices[i]]
        m = system_unwrapped.masses[grouped_indices[i]]
        centers_of_mass[i] = np.sum(r * m[:, np.newaxis], axis=0) / np.sum(m)

    identical = _test_identical_composition(system_unwrapped, grouped_indices)
    if assert_identical_composition and not identical:
        raise ValueError("Found molecules which differ in composition.")

    if identical:
        grouped_indices = np.stack(grouped_indices, axis=0)
    
    com_distances_from_origin = np.linalg.norm(centers_of_mass, axis=1)
    central_molecule_index = np.argmin(com_distances_from_origin)
    
    ref_indices = grouped_indices[central_molecule_index]
    ref_positions = positions_ref_unwrapped[ref_indices]
    min_distances = np.zeros(n_molecules_unit_cell)
    max_distances = np.zeros(n_molecules_unit_cell)

    for i in range(n_molecules_unit_cell):
        if i == central_molecule_index:
            continue
        neighbor_indices = grouped_indices[i]
        neighbor_positions = positions_ref_unwrapped[neighbor_indices]
        pairwise_distances = scipy.spatial.distance.cdist(ref_positions, neighbor_positions)
        min_distances[i] = np.min(pairwise_distances) 
        max_distances[i] = np.max(pairwise_distances)

    for i in range(system.n_frames):
        if system.positions.ndim == 3:
            positions_a = system.positions[i]
            positions_b = system_unwrapped.positions[i]
        else:
            positions_a = system.positions
            positions_b = system_unwrapped.positions
        if system.cell_vectors.ndim == 3:
            cell_a = system.cell_vectors[i]
            cell_b = system_unwrapped.cell_vectors[i]
        else:
            cell_a = system.cell_vectors
            cell_b = system_unwrapped.cell_vectors

        atomic_numbers_a = system.atomic_numbers
        atomic_numbers_b = system_unwrapped.atomic_numbers

        if validate_pbc_structure:
            #
            # For debugging
            #
            rmsd = mbe_automation.structure.crystal.match(
                positions_a, atomic_numbers_a, cell_a,
                positions_b, atomic_numbers_b, cell_b
            )
            assert rmsd is not None
            assert rmsd < 1.0E-8

    return mbe_automation.storage.MolecularCrystal(
        supercell=system_unwrapped,
        index_map=grouped_indices,
        centers_of_mass=centers_of_mass,
        identical_composition=identical,
        n_molecules=n_molecules_unit_cell,
        central_molecule_index=central_molecule_index,
        min_distances_to_central_molecule=min_distances,
        max_distances_to_central_molecule=max_distances
    )


def _extract_finite_subsystem(
        system: mbe_automation.storage.MolecularCrystal,
        selection_rule: str,
        n_molecules: int | None,
        distance: float | None
) -> mbe_automation.storage.FiniteSubsystem:
    """
    Extract a finite molecular cluster from a periodic structure.

    Molecules are selected based on specified distance criteria. For
    trajectories, assumes constant covalent bonds and no atom
    permutation between frames.
    """

    if selection_rule == "closest_to_center_of_mass":
        com_distances_from_origin = np.linalg.norm(system.centers_of_mass, axis=1)
        sorted_indices = np.argsort(com_distances_from_origin, stable=True)
        filtered_molecule_indices = sorted_indices[0:n_molecules]
        
    elif selection_rule == "closest_to_central_molecule":
        sorted_indices = np.argsort(system.min_distances_to_central_molecule, stable=True)
        filtered_molecule_indices = sorted_indices[0:n_molecules]
        
    elif selection_rule == "max_max_distance_to_central_molecule":
        mask = system.max_distances_to_central_molecule < distance
        filtered_molecule_indices = np.where(mask)[0]
        
    elif selection_rule == "max_min_distance_to_central_molecule":
        mask = system.min_distances_to_central_molecule < distance            
        filtered_molecule_indices = np.where(mask)[0]
        
    else:        
        raise ValueError(f"Invalid selection_rule: {selection_rule}")

    filtered_atom_indices = np.concatenate(
        [system.index_map[i] for i in filtered_molecule_indices]
    )
    if system.supercell.positions.ndim == 3:
        subsystem_pos = system.supercell.positions[:, filtered_atom_indices, :]
    elif system.supercell.positions.ndim == 2:
        subsystem_pos = system.supercell.positions[filtered_atom_indices, :]
    else:
        raise ValueError(f"Invalid rank of system.supercell.positions: {system.supercell.positions.ndim}")
        
    finite_subsystem = mbe_automation.storage.FiniteSubsystem(
        cluster_of_molecules=mbe_automation.storage.Structure(
            positions=subsystem_pos,
            atomic_numbers=system.supercell.atomic_numbers[filtered_atom_indices],
            masses=system.supercell.masses[filtered_atom_indices], 
            cell_vectors=None,
            n_frames=system.supercell.n_frames,
            n_atoms=len(filtered_atom_indices),
            periodic=False
        ),
        molecule_indices=filtered_molecule_indices,
        n_molecules=len(filtered_molecule_indices)
    )

    return finite_subsystem


def extract_finite_subsystem(
        system: mbe_automation.storage.MolecularCrystal,
        filter: FiniteSubsystemFilter=FiniteSubsystemFilter()
) -> List[mbe_automation.storage.FiniteSubsystem]:

    mbe_automation.common.display.framed("Finite subsystem extraction")
    finite_subsystems = []
    
    if filter.selection_rule in NUMBER_SELECTION:
        
        print(f"selection_rule  {filter.selection_rule}")
        print(f"n_molecules     {np.array2string(filter.n_molecules)}", flush=True)
        
        for n_molecules in filter.n_molecules:
            finite_subsystems.append(
                _extract_finite_subsystem(
                    system=system,
                    selection_rule=filter.selection_rule,
                    n_molecules=n_molecules,
                    distance=None
                )
            )
            
    elif filter.selection_rule in DISTANCE_SELECTION:

        print(f"selection_rule  {filter.selection_rule}")
        print(f"distances       {np.array2string(f.distances, precision=1, separator=' ')}", flush=True)
        
        for distance in filter.distances:
            finite_subsystems.append(
                _extract_finite_subsystem(
                    system=system,
                    selection_rule=filter.selection_rule,
                    n_molecules=None,
                    distance=distance
                )
            )

    print(f"Subsystem extraction completed", flush=True)
            
    return finite_subsystems
