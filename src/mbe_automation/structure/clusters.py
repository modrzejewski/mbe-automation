from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Dict, Tuple, Iterator
import math
import itertools
from collections import deque
import time
import sys
import numpy as np
import numpy.typing as npt
import ase.geometry
import ase.io
import ase.build
import ase.spacegroup.symmetrize
import ase.spacegroup.utils
from ase import Atoms, neighborlist
from ase.calculators.calculator import Calculator as ASECalculator
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
import pymatgen.core
import pymatgen.core.operations
import pymatgen.analysis.molecule_matcher

import mbe_automation.storage
import mbe_automation.structure.crystal
import mbe_automation.structure.molecule
import mbe_automation.common.display
import mbe_automation.calculators.core
from mbe_automation.storage.core import MolecularCrystal, UniqueClusters
from mbe_automation.configs.clusters import NUMBER_SELECTION, DISTANCE_SELECTION
from mbe_automation.configs.clusters import FiniteSubsystemFilter, UniqueClustersFilter
from mbe_automation.configs.structure import Minimum, SYMMETRY_TOLERANCE_LOOSE

@dataclass
class SupercellMolecules:
    """
    Atomic coordinates of molecules propagated in a supercell. 
    
    Handles the case where there were multiple unique molecules present in 
    the initial unit cell.

    Attributes:
        n_molecules_nonunique: Total count of all molecules in the supercell.
        n_molecules_unique: Count of unique molecules.
        n_equivalent: Array where `n_equivalent[k]` specifies the number 
            of occurences of the k-th unique molecule.
        positions: List of atomic coordinates. `positions[k]` has shape 
            `(n_equivalent[k], n_atoms, 3)`.
        atomic_numbers: List of atomic numbers. `atomic_numbers[k]` has shape 
            `(n_equivalent[k], n_atoms)`.
        masses: List of atomic masses. `masses[k]` has shape 
            `(n_equivalent[k], n_atoms)`.
        centers_of_mass: List of centers of mass. `centers_of_mass[k]` has shape 
            `(n_equivalent[k], 3)`.
        min_distance_to_ref_molecule: List of arrays where the k-th array has shape 
            `(n_molecules_unique, n_equivalent[k])` representing the minimum 
            atom-atom distance to each reference molecule.
    """
    n_molecules_nonunique: int
    n_molecules_unique: int
    n_equivalent: npt.NDArray[np.int64]
    positions: List[npt.NDArray[np.float64]]
    atomic_numbers: List[npt.NDArray[np.int64]]
    masses: List[npt.NDArray[np.float64]]
    centers_of_mass: List[npt.NDArray[np.float64]]
    min_distance_to_ref_molecule: List[npt.NDArray[np.float64]]

    def symmetry_unique_clusters(
        self,
        unique_cluster_filter: UniqueClustersFilter,
        key: str | None = None,
    ) -> Dict[str, List[UniqueClusters]]:
        """
        Extract symmetry-unique molecular clusters from the supercell.
        """
        return _symmetry_unique_clusters(
            supercell_molecules=self,
            unique_cluster_filter=unique_cluster_filter,
            key=key,
        )


@dataclass
class ReducibleClusters:
    """
    A collection of symmetry-reducible clusters of a given composition
    that satisfy all intermolecular distance constraints.
    
    Attributes:
        n_clusters: The total number of valid clusters identified for this composition.
        composition: A sorted tuple defining the molecular types in the clusters.
        clusters: Array of shape `(n_clusters, cluster_size)` containing 
            the candidate indices `c_idx`. The molecule type for the k-th column 
            is given by `composition[k]`. For example, if `composition` is `(0, 0, 1)`,
            a row `[0, 5, 2]` represents a cluster formed by the 0th candidate of type 0, 
            the 5th candidate of type 0, and the 2nd candidate of type 1.
        candidate_to_supercell: List of arrays mapping candidate indices `c_idx` to supercell 
            molecule indices `eq_i` for each unique molecule type.
    """
    n_clusters: int
    composition: Tuple[int, ...]
    clusters: npt.NDArray[np.int64]
    candidate_to_supercell: List[npt.NDArray[np.int64]]

    def __len__(self) -> int:
        return self.n_clusters
        
    def __getitem__(self, cluster_idx: int) -> Tuple[int, ...]:
        """
        Returns the mapped supercell molecule indices `eq_i` for the requested cluster.
        
        Note: This implicitly translates the internal candidate indices (`c_idx`) 
        stored in `self.clusters` into the global supercell coordinate space.
        """
        return self.to_supercell_indices(cluster_idx)

    def to_supercell_indices(self, cluster_idx: int) -> Tuple[int, ...]:
        """
        Returns the tuple of supercell molecule indices `eq_i` for a specific cluster.
        """
        row = self.clusters[cluster_idx]
        return tuple(self.candidate_to_supercell[u][c_idx] for u, c_idx in zip(self.composition, row))
        
    def sorted_min_rij(
        self, 
        cluster_idx: int, 
        min_rij: List[List[npt.NDArray[np.float64]]]
    ) -> npt.NDArray[np.float64]:
        """
        Returns the sorted minimum intermolecular distances for a specific cluster.
        """
        row = self.clusters[cluster_idx].tolist()
        pairs = itertools.combinations(range(len(self.composition)), 2)
        
        return np.sort([
            min_rij[self.composition[i]][self.composition[j]][row[i], row[j]] 
            for i, j in pairs
        ])

    def sorted_max_rij(
        self, 
        cluster_idx: int, 
        max_rij: List[List[npt.NDArray[np.float64]]]
    ) -> npt.NDArray[np.float64]:
        """
        Returns the sorted maximum intermolecular distances for a specific cluster.
        """
        row = self.clusters[cluster_idx].tolist()
        pairs = itertools.combinations(range(len(self.composition)), 2)
        
        return np.sort([
            max_rij[self.composition[i]][self.composition[j]][row[i], row[j]] 
            for i, j in pairs
        ])


@dataclass(kw_only=True)
class MolecularComposition:
    """
    Molecular composition of a periodic structure.

    Attributes:
        molecular_crystal: Full periodic molecular crystal graph representation.
        molecules_nonunique: List containing all individual 
            molecules found within the periodic cell.
        n_molecules_nonunique: Total count of all molecules in the cell.
        molecules_unique: List of the representative unique molecules.
        n_molecules_unique: Count of unique molecules.
        n_equivalent: n_equivalent[k] specifies how many equivalent molecules
            correspond to k-th unique molecule.
        groups: Lists of equivalent molecules. Each item is an array of indices 
            pointing to the equivalent molecules in `molecules_nonunique`.
    """
    molecular_crystal: mbe_automation.storage.core.MolecularCrystal
    molecules_nonunique: List[mbe_automation.storage.Structure]
    n_molecules_nonunique: int
    molecules_unique: List[mbe_automation.storage.Structure]
    n_molecules_unique: int
    n_equivalent: npt.NDArray[np.int64]
    groups: list[npt.NDArray[np.int64]]

    @property
    def identical_composition(self) -> bool:
        return self.molecular_crystal.identical_composition

    def expand_to_supercell(
            self, 
            supercell_size: List[int] | Tuple[int, int, int] | npt.NDArray[np.int64],
            frame_index: int = 0
    ) -> SupercellMolecules:
        """
        Extract the properties of identical molecules propagated in an [nx, ny, nz] supercell.
        """
        return _expand_to_supercell(self, supercell_size, frame_index)

    def extract_relaxed_unique_molecules(
            self,
            dataset: str,
            key: str,
            calculator: ASECalculator,
            config: Minimum,
            work_dir: Path | str = Path("./")
    ) -> None:
        """
        Extract nonequivalent molecules from a periodic cell and save the corresponding
        structures to a dataset file.
        """
        _extract_relaxed_unique_molecules(
            composition=self,
            dataset=dataset,
            key=key,
            calculator=calculator,
            config=config,
            work_dir=work_dir,
        )


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
        print(f"Cutoff for ghosts ({Cutoffs['ghosts']} Å) must be smaller than cutoff for dimers ({Cutoffs['dimers']} Å)")
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


def _generate_covalent_bond_graph(
        system: mbe_automation.storage.Structure,
        bonding_algo: NearNeighbors,
        reference_frame_index: int = 0,
        assert_identical_composition: bool = False,
        validate_pbc_structure: bool = False
) -> mbe_automation.storage.MolecularCrystal:
    """
    Identify molecules in a periodic Structure.
    """

    if not system.periodic:
        raise ValueError("_generate_covalent_bond_graph is designed for periodic systems.")

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

    print("Computing covalent bonds graph...", end="", flush=True)
    start_time = time.time()
    structure_graph = pymatgen.analysis.graphs.StructureGraph.from_local_env_strategy(
        structure=supercell,
        strategy=bonding_algo
    )
    end_time = time.time()
    delta_tau = end_time - start_time
    print(f" (Δτ={delta_tau:.2f} s)", flush=True)
    
    components = list(networkx.weakly_connected_components(structure_graph.graph))
    masses = np.array([site.specie.atomic_mass for site in supercell.sites])
    scaled_positions = supercell.frac_coords

    contiguous_molecules = []
    distances_from_center = []
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

            contiguous_molecules.append(atom_indices)
            distances_from_center.append(np.linalg.norm(com_scaled-np.array([0.5, 0.5, 0.5])))
            
    distances_from_center = np.array(distances_from_center)
    supercell_subset = []
    n_atoms_found = 0
    #
    # We must keep track of the atoms claimed to prevent
    # the following scenario:
    #
    # If the geometry is such that a periodic image of Molecule A
    # (from a neighboring cell) is spatially closer to the supercell
    # center than Molecule B (which resides inside the central unit cell),
    # the algorithm would select two images of Molecule A and zero
    # images of Molecule B.
    #
    # By keeping track of claimed_atoms, we skip the image of Molecule A
    # and accept Molecule B.
    #
    claimed_atoms = np.zeros(n_atoms_unit_cell, dtype=bool)
    
    for i in np.argsort(distances_from_center):
        if n_atoms_found >= n_atoms_unit_cell:
            break
        
        atom_indices = contiguous_molecules[i]
        unit_cell_indices = supercell_to_unit_cell[atom_indices]

        if not np.any(claimed_atoms[unit_cell_indices]):
            claimed_atoms[unit_cell_indices] = True
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
            match_result = mbe_automation.structure.crystal.match(
                positions_a, atomic_numbers_a, cell_a,
                positions_b, atomic_numbers_b, cell_b
            )
            assert match_result is not None
            assert match_result.rmsd < 1.0E-8

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


def _group_molecules_by_energy(
        molecules: List[mbe_automation.storage.Structure],
        thresh: float = 1.0E-5, # eV/atom
) -> list[npt.NDArray[np.int64]]:
    """
    Group molecules based on energy similarity.
    """
    
    assert len(molecules) > 0
    assert thresh > 0.0
    for molecule in molecules:
        assert molecule.E_pot is not None
        assert len(molecule.E_pot) == 1

    n_molecules = len(molecules)
    energies = np.array([molecules[i].E_pot[0] for i in range(n_molecules)])
    
    sort_order = np.argsort(energies, kind="stable")
    sorted_energies = energies[sort_order]
    #
    # Compute differences between consecutive sorted energies
    # diffs[i] = sorted_energies[i+1] - sorted_energies[i]
    #
    diffs = np.diff(sorted_energies)
    #
    # Find where the 'gap' is larger than the threshold
    # np.flatnonzero returns indices where the condition is True.
    # We add 1 because diff[i] corresponds to the gap between
    # index i and i+1.
    #
    split_indices = np.flatnonzero(diffs > thresh) + 1
    #
    # Split the SORTED indices array at these break points
    # This creates a list of arrays, where each array contains the
    # original indices of molecules in that group.
    #
    equiv_molecule_indices = np.split(sort_order, split_indices)    
    return equiv_molecule_indices


def _extract_nonunique_molecules(
        molecular_crystal: mbe_automation.storage.MolecularCrystal,
        reference_frame_index: int = 0,
        calculator: ASECalculator | None = None,
) -> List[mbe_automation.storage.Structure]:
    """
    Extract all molecules present in a unit cell as a list of separate
    Structures.
    """
    molecules = []
    for atom_indices in molecular_crystal.index_map:

        if molecular_crystal.supercell.positions.ndim == 3:
            selected_positions = molecular_crystal.supercell.positions[reference_frame_index][atom_indices]
        else: # ndim == 2
            selected_positions = molecular_crystal.supercell.positions[atom_indices]
        
        molecule = mbe_automation.storage.Structure(
            positions=selected_positions,
            atomic_numbers=molecular_crystal.supercell.atomic_numbers[atom_indices],
            masses=molecular_crystal.supercell.masses[atom_indices],
            cell_vectors=None,
            n_frames=1,
            n_atoms=len(atom_indices),
        )
        molecules.append(molecule)

    if calculator is not None:
        for molecule in molecules:
            E_pot, _, _, _ = mbe_automation.calculators.core.run_model(
                structure=molecule,
                calculator=calculator,
                compute_energies=True,
                compute_forces=False,
                compute_feature_vectors=False,
                silent=True,
            )
            molecule.E_pot = E_pot

    return molecules


def _group_molecules_by_rmsd(
        molecules: List[mbe_automation.storage.Structure],
        thresh: float = SYMMETRY_TOLERANCE_LOOSE,
) -> list[npt.NDArray[np.int64]]:
    n_mols = len(molecules)
    if n_mols == 0:
        return []

    adjacency_matrix = np.zeros((n_mols, n_mols), dtype=bool)
    for i in range(n_mols):
        adjacency_matrix[i, i] = True
        for j in range(i + 1, n_mols):
            rmsd = mbe_automation.structure.molecule.match(
                positions_a=molecules[i].positions,
                atomic_numbers_a=molecules[i].atomic_numbers,
                positions_b=molecules[j].positions,
                atomic_numbers_b=molecules[j].atomic_numbers,
            )
            if not np.isnan(rmsd) and rmsd < thresh:
                adjacency_matrix[i, j] = True
                adjacency_matrix[j, i] = True

    n_components, labels = sparse.csgraph.connected_components(
        adjacency_matrix, directed=False, return_labels=True
    )

    grouped_indices = []
    for comp_idx in range(n_components):
        grouped_indices.append(np.where(labels == comp_idx)[0])

    return grouped_indices

def _split_groups_by_rmsd(
        molecules_nonunique: List[mbe_automation.storage.Structure],
        energy_groups: list[npt.NDArray[np.int64]],
        rmsd_thresh: float = SYMMETRY_TOLERANCE_LOOSE,
) -> list[npt.NDArray[np.int64]]:
    
    energy_rmsd_groups = []
    for group_indices in energy_groups:
        group_molecules = [molecules_nonunique[i] for i in group_indices]
        rmsd_groups = _group_molecules_by_rmsd(
            molecules=group_molecules,
            thresh=rmsd_thresh,
        )
        for rmsd_group_indices in rmsd_groups:
            energy_rmsd_groups.append(group_indices[rmsd_group_indices])

    return energy_rmsd_groups


def _display_unique_molecules(
        molecules_nonunique: List[mbe_automation.storage.Structure],
        groups: list[npt.NDArray[np.int64]],
        energies_available: bool,
):

    if energies_available:
        header = f"{'molecule':>8}   {'n_atoms':>8}   {'n_equivalent':>12}   {'energy (eV/atom)':<17}   {'formula':<15}"
    else:
        header = f"{'molecule':>8}   {'n_atoms':>8}   {'n_equivalent':>12}   {'formula':<15}"

    mbe_automation.common.display.dotted_separator(len(header))
    print(header)
    mbe_automation.common.display.dotted_separator(len(header))

    for i, group_indices in enumerate(groups):
        mol = molecules_nonunique[group_indices[0]]
        pmg_mol = mol.to_pymatgen()
        comp = pmg_mol.composition.alphabetical_formula
        n_atoms = len(pmg_mol)
        n_eq = len(group_indices)

        if energies_available:
            for j, idx in enumerate(group_indices):
                energy_val = molecules_nonunique[idx].E_pot[0]
                energy = f"{energy_val:.6f}"
                if j == 0:
                    print(f"{i:>8}   {n_atoms:>8}   {n_eq:>12}   {energy:>17}   {comp:<15}")
                else:
                    print(f"{'':>8}   {'':>8}   {'':>12}   {energy:>17}")
        else:
            print(f"{i:>8}   {n_atoms:>8}   {n_eq:>12}   {comp:<15}")

    mbe_automation.common.display.dotted_separator(len(header))


def identify_molecules(
        crystal: mbe_automation.storage.Structure,
        calculator: ASECalculator | None = None,
        energy_thresh: float | None = None, # eV/atom
        rmsd_thresh: float | None = None, # Angs
        assert_identical_composition: bool = False,
        bonding_algo: NearNeighbors | None = None,
        reference_frame_index: int = 0,
        match_mode: Literal["energy_only", "rmsd_only", "combined"] = "energy_only",
) -> MolecularComposition:
    """
    Identify and extract molecules from a periodic structure. Group nonunique molecules into symmetry-unique subsets based on structure and potential energy.

    Args:
        crystal: The periodic structure to process.
        calculator: An optional ASE calculator to compute the potential energy of individual molecules.
        energy_thresh: Threshold in eV/atom for considering two molecules to have the same potential energy.
        rmsd_thresh: Threshold in Å for considering two molecules structurally identical based on RMSD.
        assert_identical_composition: If True, raises an error if the identified molecules do not share identical atomic compositions.
        bonding_algo: Optional pymatgen bonding algorithm to determine connectivity.
        reference_frame_index: The index of the frame to use as reference.
        match_mode: The mechanism for clustering structurally identical molecules in the cell:
            * "energy_only": Groups molecules based purely on their potential energy (per atom) using 
              `energy_thresh`. Avoids structural alignment, but fails to differentiate 
              isomers with identical or nearly identical energies.
            * "rmsd_only": Groups molecules directly via optimal RMSD alignment using `rmsd_thresh`. 
            * "combined": A hierarchical algorithm that first bins molecules into energy groups, 
              then sub-clusters each individual energy grouping using the RMSD 
              criterion.

    Returns:
        MolecularComposition dataclass containing the grouped molecular representations.
    """
    assert crystal.periodic
    assert match_mode in ("energy_only", "rmsd_only", "combined")

    if match_mode in ("energy_only", "combined") and calculator is None:
        raise ValueError(f"Cannot use {match_mode} match_mode when calculator is None.")

    if energy_thresh is None:
        energy_thresh = 1.0E-5
    if rmsd_thresh is None:
        rmsd_thresh = 0.1

    if bonding_algo is None:
        bonding_algo = CutOffDictNN.from_preset("vesta_2019")

    mbe_automation.common.display.framed("Molecule detection")
    print(f"bonding_algo                {type(bonding_algo).__name__}")
    print(f"match_mode                  {match_mode}")
    if match_mode in ("energy_only", "combined"):
        print(f"energy_thresh               {energy_thresh} eV/atom")
    if match_mode in ("rmsd_only", "combined"):
        print(f"rmsd_thresh                 {rmsd_thresh} Å")
    print(f"assert_identical_comp       {assert_identical_composition}")
    if crystal.n_frames > 1:
        print(f"reference_frame_index       {reference_frame_index}")
    print()

    assert crystal.atomic_numbers.ndim == 1
    assert crystal.masses.ndim == 1

    molecular_crystal = _generate_covalent_bond_graph(
        system=crystal,
        reference_frame_index=reference_frame_index,
        assert_identical_composition=assert_identical_composition,
        bonding_algo=bonding_algo,
    )

    molecules_nonunique = _extract_nonunique_molecules(
        molecular_crystal=molecular_crystal,
        reference_frame_index=reference_frame_index,
        calculator=calculator if match_mode in ("energy_only", "combined") else None,
    )

    if match_mode == "energy_only":
        groups = _group_molecules_by_energy(
            molecules=molecules_nonunique,
            thresh=energy_thresh,
        )
    elif match_mode == "rmsd_only":
        groups = _group_molecules_by_rmsd(
            molecules=molecules_nonunique,
            thresh=rmsd_thresh,
        )
    elif match_mode == "combined":
        energy_groups = _group_molecules_by_energy(
            molecules=molecules_nonunique,
            thresh=energy_thresh,
        )
        groups = _split_groups_by_rmsd(
            molecules_nonunique=molecules_nonunique,
            energy_groups=energy_groups,
            rmsd_thresh=rmsd_thresh,
        )

    molecules_unique = [molecules_nonunique[group[0]] for group in groups]
    n_molecules_unique = len(groups)
    n_equivalent = np.array([len(group) for group in groups], dtype=np.int64)

    print(f"{'Nonunique molecules:':<32}{len(molecules_nonunique)}/unit cell")
    print(f"{'Unique molecules:':<32}{n_molecules_unique}/unit cell")

    _display_unique_molecules(
        molecules_nonunique=molecules_nonunique,
        groups=groups,
        energies_available=(match_mode in ("energy_only", "combined"))
    )

    return MolecularComposition(
        molecular_crystal=molecular_crystal,
        molecules_nonunique=molecules_nonunique,
        n_molecules_nonunique=len(molecules_nonunique),
        molecules_unique=molecules_unique,
        n_molecules_unique=n_molecules_unique,
        n_equivalent=n_equivalent,
        groups=groups,
    )


def _shortest_atom_atom_distance(
    ref_pos: npt.NDArray[np.float64], 
    target_positions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Computes the shortest atom-atom distance between a reference molecule 
    and a set of target molecules in a fully vectorized manner.
    
    Args:
        ref_pos: Shape (n_atoms_ref, 3)
        target_positions: Shape (n_targets, n_atoms_target, 3)
        
    Returns:
        1D array of shape (n_targets,) with the minimum distances.
    """
    n_targets, n_atoms_target, _ = target_positions.shape
    
    # Reshape targets to a flat list of coordinates (n_targets * n_atoms_target, 3)
    flat_targets = target_positions.reshape(-1, 3)
    
    # Compute distances from all target atoms to all reference atoms
    # Resulting shape: (n_targets * n_atoms_target, n_atoms_ref)
    dists = scipy.spatial.distance.cdist(flat_targets, ref_pos)
    
    # Find the minimum distance to ANY atom in the reference molecule for each target atom
    # Reshape back to group by target molecule: (n_targets, n_atoms_target)
    min_dists_per_target_atom = np.min(dists, axis=1).reshape(n_targets, n_atoms_target)
    
    # Find the minimum distance for each target molecule
    return np.min(min_dists_per_target_atom, axis=1)


def _expand_to_supercell(
        composition: MolecularComposition,
        supercell_size: List[int] | Tuple[int, int, int] | npt.NDArray[np.int64],
        frame_index: int = 0
) -> SupercellMolecules:
    """Generate arrays of coordinates, atomic numbers, and masses for the 
    molecules identified in the unit cell, propagated to the specified supercell.

    Args:
        composition: Base molecular composition.
        supercell_size: Dimensions of the supercell [nx, ny, nz].
        frame_index: Index of the reference frame to extract positions from.

    Returns:
        SupercellMolecules object containing propagated atomic properties.
    """
    nx, ny, nz = supercell_size
    
    unit_cell = composition.molecular_crystal.supercell
    variable_cell = unit_cell.variable_cell
    n_cells = nx * ny * nz
    
    if variable_cell:
        unit_cell_vectors = unit_cell.cell_vectors[frame_index]
    else:
        unit_cell_vectors = unit_cell.cell_vectors
        
    I, J, K = np.meshgrid(
        np.arange(-(nx // 2), nx - (nx // 2)),
        np.arange(-(ny // 2), ny - (ny // 2)),
        np.arange(-(nz // 2), nz - (nz // 2)),
        indexing='ij'
    )
    shifts_frac = np.column_stack((I.ravel(), J.ravel(), K.ravel()))
    shifts_cart = shifts_frac @ unit_cell_vectors
    
    positions_list = []
    atomic_numbers_list = []
    masses_list = []
    centers_of_mass_list = []
    is_multi_frame = (unit_cell.positions.ndim == 3)
    
    for u in range(composition.n_molecules_unique):
        mol_unique = composition.molecules_unique[u]
        
        # `atom_indices` is a rank-2 array (2D matrix) of shape:
        # (n_equivalent_molecules, n_atoms_per_molecule)
        molecule_indices = composition.groups[u]
        if composition.identical_composition:
            atom_indices = composition.molecular_crystal.index_map[molecule_indices]
        else:
            atom_indices = np.array([
                composition.molecular_crystal.index_map[idx] 
                for idx in molecule_indices
            ])
            
        if is_multi_frame:
            base_pos = unit_cell.positions[frame_index, atom_indices, :]
        else:
            base_pos = unit_cell.positions[atom_indices, :]
            
        base_an = unit_cell.atomic_numbers[atom_indices]
        base_masses = unit_cell.masses[atom_indices]
        
        pos_supercell = base_pos[np.newaxis, :, :, :] + shifts_cart[:, np.newaxis, np.newaxis, :]
        pos_supercell = pos_supercell.reshape(-1, mol_unique.n_atoms, 3)
        positions_list.append(pos_supercell)
        
        atomic_numbers_list.append(np.tile(base_an, (n_cells, 1)))
        masses_list.append(np.tile(base_masses, (n_cells, 1)))
        
    total_mass = 0.0
    total_mass_r = np.zeros(3)
    for u in range(composition.n_molecules_unique):
        p = positions_list[u]
        m = masses_list[u]
        total_mass += np.sum(m)
        total_mass_r += np.sum(p * m[:, :, np.newaxis], axis=(0, 1))
        
    total_com = total_mass_r / total_mass
    
    for u in range(composition.n_molecules_unique):
        positions_list[u] -= total_com
        p = positions_list[u]
        m = masses_list[u]
        coms = np.sum(p * m[:, :, np.newaxis], axis=1) / np.sum(m, axis=1)[:, np.newaxis]
        
        distances_to_origin = np.linalg.norm(coms, axis=1)
        sort_indices = np.argsort(distances_to_origin)
        
        positions_list[u] = positions_list[u][sort_indices]
        atomic_numbers_list[u] = atomic_numbers_list[u][sort_indices]
        masses_list[u] = masses_list[u][sort_indices]
        coms = coms[sort_indices]
        
        centers_of_mass_list.append(coms)
        
    min_distance_to_ref_molecule = []
    for u in range(composition.n_molecules_unique):
        dist_matrix = np.zeros((composition.n_molecules_unique, composition.n_equivalent[u] * n_cells))
        
        for v in range(composition.n_molecules_unique):
            dist_matrix[v, :] = _shortest_atom_atom_distance(
                positions_list[v][0], 
                positions_list[u]
            )
            
        min_distance_to_ref_molecule.append(dist_matrix)
        
    return SupercellMolecules(
        n_molecules_nonunique=composition.n_molecules_nonunique * n_cells,
        n_molecules_unique=composition.n_molecules_unique,
        n_equivalent=np.array(composition.n_equivalent * n_cells, dtype=np.int64),
        positions=positions_list,
        atomic_numbers=atomic_numbers_list,
        masses=masses_list,
        centers_of_mass=centers_of_mass_list,
        min_distance_to_ref_molecule=min_distance_to_ref_molecule
    )


def _extract_relaxed_unique_molecules(
        composition: MolecularComposition,
        dataset: str,
        key: str,
        calculator: ASECalculator,
        config: Minimum,
        work_dir: Path | str = Path("./")
) -> None:
    """
    Extract nonequivalent molecules from a periodic cell and save the corresponding
    structures to a dataset file.
    """
    unique_molecules = composition.molecules_unique
    n_unique_molecules = composition.n_molecules_unique
    
    relaxed_molecules = []
    relaxed_labels = []
    unique_labels = []
    
    for i, molecule in enumerate(unique_molecules):

        unique_labels.append(f"molecule[extracted,{i}]")
        relaxed_labels.append(f"molecule[extracted,{i},opt:atoms]")
        
        relaxed_molecule = mbe_automation.structure.relax.isolated_molecule(
            molecule=molecule,
            calculator=calculator,
            config=config,
            work_dir=Path(work_dir)/relaxed_labels[-1],
            key=f"{key}/{relaxed_labels[-1]}",
        )

        relaxed_molecules.append(relaxed_molecule)

    for molecule in relaxed_molecules:
        E_pot, _, _, _ = mbe_automation.calculators.core.run_model(
                structure=molecule,
                calculator=calculator,
                compute_energies=True,
                compute_forces=False,
                compute_feature_vectors=False,
                silent=True,
            )
        molecule.E_pot = E_pot

    mbe_automation.storage.save_attribute(
        dataset=dataset,
        key=key,
        attribute_name="n_unique_molecules",
        attribute_value=n_unique_molecules,
    )

    for i in range(n_unique_molecules):
        mbe_automation.storage.save_structure(
            dataset=dataset,
            key=f"{key}/{unique_labels[i]}",
            structure=unique_molecules[i],
        )
        mbe_automation.storage.save_structure(
            dataset=dataset,
            key=f"{key}/{relaxed_labels[i]}",
            structure=relaxed_molecules[i],
        )

    return
        

def extract_finite_subsystem(
        system: mbe_automation.storage.MolecularCrystal,
        filter: FiniteSubsystemFilter | None = None,
) -> List[mbe_automation.storage.FiniteSubsystem]:
    """
    Extract symmetry-unique clusters from a MolecularCrystal.
    """
    
    if filter is None:
        filter = FiniteSubsystemFilter()
    
    mbe_automation.common.display.framed("Finite subsystem")
    finite_subsystems = []
    
    if filter.selection_rule in NUMBER_SELECTION:
        
        print(f"selection_rule  {filter.selection_rule}")
        print(f"n_molecules     {np.array2string(filter.n_molecules)}", flush=True)
        
        for n_molecules in filter.n_molecules:
            #
            # Ignore request if n_molecules exceeds total
            # molecules in the unit cell.
            #
            if n_molecules > system.n_molecules:
                print(f"Skipping n_molecules={n_molecules} (exceeds system.n_molecules={system.n_molecules})", flush=True)
                continue
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
        print(f"distances       {np.array2string(filter.distances, precision=1, separator=' ')}", flush=True)

        last_n_molecules = 0
        for distance in np.sort(filter.distances):
            new_subsystem = _extract_finite_subsystem(
                system=system,
                selection_rule=filter.selection_rule,
                n_molecules=None,
                distance=distance
            )
            #
            # Add subsystem only if the number of
            # molecules is different from the previous one.
            #
            if new_subsystem.n_molecules > last_n_molecules:
                finite_subsystems.append(new_subsystem)
                last_n_molecules = new_subsystem.n_molecules
                
    print(f"Subsystem extraction completed", flush=True)
            
    return finite_subsystems


def _global_candidates(
    supercell_molecules: SupercellMolecules,
    unique_cluster_filter: UniqueClustersFilter,
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.float64]]]:
    """
    Filter supercell molecules to those within the distance cutoff from any central reference molecule.
    
    A candidate is a molecule that is close enough to the reference molecule that it might be 
    included in at least some of the molecular clusters according to the filtering criteria.
    
    Resolve candidates by their unique molecule type `u`. Always include central reference 
    molecules (instance index 0) in the candidate pools, regardless of the cutoff.
    
    Args:
        supercell_molecules: The propagated supercell data containing coordinates and distances.
        unique_cluster_filter: The filter containing cutoffs for inclusion in the candidate pool.
        
    Returns:
        candidate_to_supercell: List of arrays (length n_unique) containing the supercell 
                                molecule indices of the valid candidates.
        candidate_positions: List of arrays (length n_unique) containing the coordinates 
                             of the valid candidates.
    """
    max_cutoff = max(unique_cluster_filter.cutoffs.values())
    n_unique = supercell_molecules.n_molecules_unique
    candidate_to_supercell = []
    candidate_positions = []

    for u in range(n_unique):
        # min_distance_to_ref_molecule[u] has shape (n_molecules_unique, N_eq[u])
        dist_to_any_ref = np.min(supercell_molecules.min_distance_to_ref_molecule[u], axis=0)
        
        mask = dist_to_any_ref < max_cutoff
        # Ensure reference molecules (instance index 0) are strictly included
        mask[0] = True
        
        eq_indices = np.where(mask)[0]
        
        candidate_to_supercell.append(eq_indices)
        candidate_positions.append(supercell_molecules.positions[u][eq_indices])
        
    return candidate_to_supercell, candidate_positions


def _intermolecular_distances(
    positions_a: npt.NDArray[np.float64], 
    positions_b: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes the shortest and longest atom-atom distances between two sets of molecules.
    
    Args:
        positions_a: Shape (n_a, n_atoms_a, 3)
        positions_b: Shape (n_b, n_atoms_b, 3)
        
    Returns:
        min_dists, max_dists: Arrays of shape (n_a, n_b) containing the minimum and maximum 
                              distances between each pair of molecules.
    """
    n_a, n_atoms_a, _ = positions_a.shape
    n_b, n_atoms_b, _ = positions_b.shape
    
    flat_a = positions_a.reshape(-1, 3)
    flat_b = positions_b.reshape(-1, 3)
    
    # Shape: (n_a * n_atoms_a, n_b * n_atoms_b)
    dists = scipy.spatial.distance.cdist(flat_a, flat_b)
    
    # Reshape to (n_a, n_atoms_a, n_b, n_atoms_b)
    dists = dists.reshape(n_a, n_atoms_a, n_b, n_atoms_b)
    
    # Minimum and maximum distances between molecules
    min_dists = np.min(dists, axis=(1, 3)) # Shape: (n_a, n_b)
    max_dists = np.max(dists, axis=(1, 3)) # Shape: (n_a, n_b)
    
    return min_dists, max_dists


def _candidate_distances(
    candidate_positions: List[npt.NDArray[np.float64]],
) -> Tuple[List[List[npt.NDArray[np.float64]]], List[List[npt.NDArray[np.float64]]]]:
    """
    Calculate the minimum and maximum distance matrices between candidates 
    resolved into groups corresponding to unique molecules (u, v).
    
    Args:
        candidate_positions: List of arrays containing the coordinates of valid 
            candidates, grouped by unique molecule type. `candidate_positions[u]` 
            is a numpy array of shape `(N_u, N_atoms, 3)` representing all 
            candidate molecules of type `u`.
                             
    Returns:
        min_rij: A nested list where `min_rij[u][v]` is a numpy array of shape 
            `(N_u, N_v)` containing the minimum distances between candidate 
            instances of type `u` and type `v`.
            
            Example: `min_rij[0][1][i, j]` is the shortest atom-atom distance 
            between the i-th candidate of type 0 and the j-th candidate of type 1.
            
        max_rij: A nested list of the same structure, containing the maximum 
            distances.
    """
    n_unique = len(candidate_positions)
    min_rij = [[np.empty(0) for _ in range(n_unique)] for _ in range(n_unique)]
    max_rij = [[np.empty(0) for _ in range(n_unique)] for _ in range(n_unique)]
    
    for u in range(n_unique):
        for v in range(u, n_unique):
            min_d, max_d = _intermolecular_distances(
                candidate_positions[u], 
                candidate_positions[v]
            )
            min_rij[u][v] = min_d
            max_rij[u][v] = max_d
            if u != v:
                min_rij[v][u] = min_d.T
                max_rij[v][u] = max_d.T
                
    return min_rij, max_rij


def _canonical_cluster_types(n_unique: int, cluster_size: int) -> Iterator[Tuple[int, ...]]:
    """
    Generate all possible canonical cluster compositions.
    
    For a given number of unique molecules in the asymmetric unit (n_unique) and a target 
    cluster size, this function yields sorted tuples representing all valid molecular 
    compositions.
    
    Example: for n_unique=2 (types 0, 1) and cluster_size=3 (trimers), this yields:
    (0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1).
    """
    return itertools.combinations_with_replacement(range(n_unique), cluster_size)


def _filter_candidates_by_min_rij(
    composition: Tuple[int, ...],
    max_min_rij: float,
    min_rij: List[List[npt.NDArray[np.float64]]],
    candidate_to_supercell: List[npt.NDArray[np.int64]],
) -> ReducibleClusters:
    """
    Find all symmetry-reducible clusters for a given composition that satisfy all minimum intermolecular distance constraints.
    
    A cluster satisfies the constraints if the minimum intermolecular distance between ANY two molecules
    in the cluster is strictly less than the `max_min_rij`.
    
    Args:
        composition: A sorted tuple representing the cluster composition (e.g., `(0, 0, 1)`).
        max_min_rij: The maximum distance threshold.
        min_rij: Minimum distance matrices between candidate instances.
        candidate_to_supercell: Reference mapping table to attach to the output dataclass.
        
    Returns:
        ReducibleClusters: A dataclass containing the composition, total count, 
            and an array of valid cluster indices.
    """
    cluster_size = len(composition)
    n_candidates = [len(c) for c in candidate_to_supercell]
    unique_u, counts = np.unique(composition, return_counts=True)
    
    # `cands_per_u` generates combinations of candidate indices for each unique molecule type.
    # Example: If a cluster needs two molecules of type 0 (counts=2), and there are 4 candidates
    # of type 0 in the supercell, the generator yields: (0, 1), (0, 2), (0, 3), (1, 2), ...
    cands_per_u = (
        itertools.combinations(range(n_candidates[u]), n)
        for u, n in zip(unique_u, counts)
    )
    
    # itertools.product takes the Cartesian product of these type-specific combinations.
    # Flatten the grouped combinations: ((0, 1), (2,)) -> [0, 1, 2]
    all_clusters = np.array([
        [idx for group in sub for idx in group] 
        for sub in itertools.product(*cands_per_u)
    ], dtype=np.int64)
    
    if len(all_clusters) == 0:
        return ReducibleClusters(
            n_clusters=0,
            composition=composition,
            clusters=np.empty((0, cluster_size), dtype=np.int64),
            candidate_to_supercell=candidate_to_supercell,
        )
        
    within_cutoff = np.ones(len(all_clusters), dtype=bool)
    for i, j in itertools.combinations(range(cluster_size), 2):
        u1, u2 = composition[i], composition[j]
        c1, c2 = all_clusters[:, i], all_clusters[:, j]
        within_cutoff &= (min_rij[u1][u2][c1, c2] < max_min_rij)
            
    filtered_clusters = all_clusters[within_cutoff]
    
    return ReducibleClusters(
        n_clusters=len(filtered_clusters),
        composition=composition,
        clusters=filtered_clusters,
        candidate_to_supercell=candidate_to_supercell
    )


def _symmetry_unique_clusters(
    supercell_molecules: SupercellMolecules,
    unique_cluster_filter: UniqueClustersFilter,
    key: str | None = None, # dataset key (only to display a message)
) -> Dict[str, List[UniqueClusters]]:

    cluster_size_map = {"monomers": 1, "dimers": 2, "trimers": 3, "tetramers": 4}

    if key is not None:
        mbe_automation.common.display.framed([
            "Symmetry-unique molecular clusters",
            key
        ])
    else:
        mbe_automation.common.display.framed(
            "Symmetry-unique molecular clusters"
        )

    print(f"cluster_types        {unique_cluster_filter.cluster_types}")
    print(f"alignment_thresh     {unique_cluster_filter.alignment_thresh} Å")
    print(f"align_mirror_images  {unique_cluster_filter.align_mirror_images}")
    print(f"algorithm            {unique_cluster_filter.algorithm}")

    candidate_to_supercell, candidate_positions = _global_candidates(
        supercell_molecules=supercell_molecules,
        unique_cluster_filter=unique_cluster_filter,
    )
    n_unique = supercell_molecules.n_molecules_unique

    min_rij, max_rij = _candidate_distances(candidate_positions)

    results = {}

    for cluster_type in unique_cluster_filter.cluster_types:
        max_min_rij = unique_cluster_filter.cutoffs[cluster_type]
        cluster_size = cluster_size_map[cluster_type]
        print(f"{cluster_type} with max_min_rij < {max_min_rij:.2f} Å...")
        
        clusters_by_comp = {}
        
        for comp in _canonical_cluster_types(n_unique, cluster_size):
            reducible = _filter_candidates_by_min_rij(
                composition=comp,
                max_min_rij=max_min_rij,
                min_rij=min_rij,
                candidate_to_supercell=candidate_to_supercell,
            )
            
            if reducible.n_clusters == 0:
                continue
                
            clusters_by_comp[comp] = {
                "indices_list": [],
                "weights_list": [],
                "min_dists_list": [],
                "max_dists_list": [],
                "positions_list": [],
                "atomic_numbers_list": []
            }
            comp_data = clusters_by_comp[comp]
            
            comp_str = "-".join(str(c) for c in comp)
            progress = mbe_automation.common.display.Progress(
                iterable=reducible,
                n_total_steps=len(reducible),
                label=f"type {comp_str}",
            )

            # 3. Type-Aware Clustering Loop
            for cluster_idx, eq_indices in enumerate(progress):
                # Extract min/max distances
                min_dists_current = reducible.sorted_min_rij(cluster_idx, min_rij)
                max_dists_current = reducible.sorted_max_rij(cluster_idx, max_rij)
                
                # The canonical `comp` order is already sorted.
                
                # Extract positions and atomic numbers for the current combination
                positions_current_list = []
                atomic_numbers_current_list = []
                for u, eq_i in zip(comp, eq_indices):
                    positions_current_list.append(supercell_molecules.positions[u][eq_i])
                    atomic_numbers_current_list.append(supercell_molecules.atomic_numbers[u][eq_i])
                    
                positions_current = np.concatenate(positions_current_list, axis=0)
                atomic_numbers_current = np.concatenate(atomic_numbers_current_list, axis=0)
            
                is_unique = True
                for i in range(len(comp_data["indices_list"])):
                    min_dists_ref = comp_data["min_dists_list"][i]
                    if (np.max(np.abs(min_dists_current - min_dists_ref)) <
                        unique_cluster_filter.alignment_thresh):
                        
                        positions_ref = comp_data["positions_list"][i]
                        atomic_numbers_ref = comp_data["atomic_numbers_list"][i]
                        
                        rmsd = mbe_automation.structure.molecule.match(
                            positions_a=positions_current,
                            atomic_numbers_a=atomic_numbers_current,
                            positions_b=positions_ref,
                            atomic_numbers_b=atomic_numbers_ref,
                            align_mirror_images=unique_cluster_filter.align_mirror_images,
                            algorithm=unique_cluster_filter.algorithm,
                        )
                        if rmsd < unique_cluster_filter.alignment_thresh:
                            is_unique = False
                            comp_data["weights_list"][i] += 1
                            break

                if is_unique:
                    comp_data["indices_list"].append(eq_indices)
                    comp_data["weights_list"].append(1)
                    comp_data["min_dists_list"].append(min_dists_current)
                    comp_data["max_dists_list"].append(max_dists_current)
                    comp_data["positions_list"].append(positions_current)
                    comp_data["atomic_numbers_list"].append(atomic_numbers_current)

        if not clusters_by_comp:
            continue
            
        results[cluster_type] = []
        for composition, comp_data in clusters_by_comp.items():
            if not comp_data["indices_list"]:
                continue
                
            molecule_indices_arr = np.array(comp_data["indices_list"], dtype=np.int64)
            weights_arr = np.array(comp_data["weights_list"], dtype=np.int64)
            
            n_clusters_comp = len(molecule_indices_arr)
            
            results[cluster_type].append(
                UniqueClusters(
                    n_clusters=n_clusters_comp,
                    cluster_composition=composition,
                    molecule_indices=molecule_indices_arr,
                    weights=weights_arr,
                    min_distances=(np.array(comp_data["min_dists_list"]) if cluster_size > 1 else np.array([])),
                    max_distances=(np.array(comp_data["max_dists_list"]) if cluster_size > 1 else np.array([])),
                )
            )
            print(f"Found {n_clusters_comp} symmetry-unique {cluster_type} of composition {composition}")

    return results
