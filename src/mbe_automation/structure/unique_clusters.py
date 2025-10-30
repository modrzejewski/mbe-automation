from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Dict
import numpy as np
import numpy.typing as npt
import itertools
from ase import Atoms
import scipy
from . import crystal
from . import compare
from ..storage import MolecularCrystal


@dataclass(kw_only=True)
class UniqueClusterFilter:
    """
    Filter for finding unique molecular clusters.
    """
    cluster_types: List[str] = field(default_factory=lambda: ["dimers"])
    cutoffs: Dict[str, float] = field(default_factory=lambda: {"dimers": 5.0})
    match_algo: Literal["RMSD", "MBTR"] = "RMSD"
    alignment_thresh: float = 1.0e-4
    align_mirror_images: bool = True

@dataclass(kw_only=True)
class UniqueCluster:
    """
    Represents a single unique cluster.
    """
    molecule_indices: npt.NDArray[np.integer]
    weight: int

@dataclass(kw_only=True)
class UniqueClusters:
    """
    Contains a list of unique molecular clusters.
    """
    unique_clusters: List[UniqueCluster]

def unique_clusters(
    molecular_crystal: MolecularCrystal,
    unique_cluster_filter: UniqueClusterFilter,
) -> UniqueClusters:

    def _get_cluster_atoms(molecular_crystal: MolecularCrystal, indices: npt.NDArray[np.integer]) -> Atoms:
        atom_indices = np.concatenate([molecular_crystal.index_map[i] for i in indices])

        if molecular_crystal.supercell.positions.ndim == 3:
            positions = molecular_crystal.supercell.positions[0, atom_indices, :]
        else:
            positions = molecular_crystal.supercell.positions[atom_indices, :]

        atomic_numbers = molecular_crystal.supercell.atomic_numbers[atom_indices]
        return Atoms(positions=positions, numbers=atomic_numbers)

    cluster_size_map = {"monomers": 1, "dimers": 2, "trimers": 3, "tetramers": 4}

    max_cutoff = 0.0
    if unique_cluster_filter.cutoffs:
        max_cutoff = max(unique_cluster_filter.cutoffs.values())

    central_molecule_idx = molecular_crystal.central_molecule_index

    # Pre-filter molecules based on the largest cutoff radius.
    candidate_molecule_indices = np.where(
        molecular_crystal.min_distances_to_central_molecule < max_cutoff
    )[0]

    # Ensure the central molecule is included.
    if central_molecule_idx not in candidate_molecule_indices:
        candidate_molecule_indices = np.append(candidate_molecule_indices, central_molecule_idx)

    # Efficiently calculate the distance matrix for the candidate molecules.
    min_rij = np.full((molecular_crystal.n_molecules, molecular_crystal.n_molecules), np.inf)

    if len(candidate_molecule_indices) > 1:

        # Extract all atomic positions for candidate molecules into a single array
        all_positions = []
        # Keep track of which atoms belong to which molecule
        mol_slices = [0]

        for mol_idx in candidate_molecule_indices:
            atom_indices = molecular_crystal.index_map[mol_idx]
            if molecular_crystal.supercell.positions.ndim == 3:
                positions = molecular_crystal.supercell.positions[0, atom_indices, :]
            else:
                positions = molecular_crystal.supercell.positions[atom_indices, :]
            all_positions.append(positions)
            mol_slices.append(mol_slices[-1] + len(atom_indices))

        all_positions = np.concatenate(all_positions, axis=0)

        # Calculate all atom-atom distances at once
        dist_matrix = scipy.spatial.distance.cdist(all_positions, all_positions)

        # Now, extract the minimum distance for each pair of molecules
        for i, mol_idx1 in enumerate(candidate_molecule_indices):
            for j, mol_idx2 in enumerate(candidate_molecule_indices):
                if i >= j:
                    continue

                start1, end1 = mol_slices[i], mol_slices[i+1]
                start2, end2 = mol_slices[j], mol_slices[j+1]

                sub_matrix = dist_matrix[start1:end1, start2:end2]
                min_dist = np.min(sub_matrix)

                min_rij[mol_idx1, mol_idx2] = min_dist
                min_rij[mol_idx2, mol_idx1] = min_dist

    all_unique_clusters = []

    for cluster_type in unique_cluster_filter.cluster_types:
        cutoff = unique_cluster_filter.cutoffs[cluster_type]
        n = cluster_size_map[cluster_type]

        #
        # Create a list of all possible clusters of size n using only the pre-filtered candidate molecules.
        #
        central_molecule = molecular_crystal.central_molecule_index

        other_candidate_molecules = [i for i in candidate_molecule_indices if i != central_molecule]

        all_clusters_indices = []
        for combo in itertools.combinations(other_candidate_molecules, n - 1):
            all_clusters_indices.append(np.array([central_molecule] + list(combo)))

        #
        # Filter clusters based on the cutoff distance
        #
        filtered_clusters_indices = []
        for indices in all_clusters_indices:
            within_cutoff = True
            for i, j in itertools.combinations(indices, 2):
                if min_rij[i, j] >= cutoff:
                    within_cutoff = False
                    break
            if within_cutoff:
                filtered_clusters_indices.append(indices)

        unique_clusters_for_type = []

        for indices in filtered_clusters_indices:

            cluster_atoms = _get_cluster_atoms(molecular_crystal, indices)

            is_unique = True
            for unique_cluster in unique_clusters_for_type:

                unique_cluster_atoms = _get_cluster_atoms(molecular_crystal, unique_cluster.molecule_indices)

                if unique_cluster_filter.match_algo == "RMSD":
                    dist = compare.AlignMolecules_RMSD(cluster_atoms, unique_cluster_atoms.copy())

                    if unique_cluster_filter.align_mirror_images:
                        mirrored_cluster_atoms = unique_cluster_atoms.copy()
                        coords2 = mirrored_cluster_atoms.get_positions()
                        coords2[:, 1] *= -1
                        mirrored_cluster_atoms.set_positions(coords2)
                        dist2 = compare.AlignMolecules_RMSD(cluster_atoms, mirrored_cluster_atoms)
                        dist = min(dist, dist2)

                    if dist < unique_cluster_filter.alignment_thresh:
                        is_unique = False
                        unique_cluster.weight += 1
                        break

                elif unique_cluster_filter.match_algo == "MBTR":
                    raise NotImplementedError("MBTR matching is not yet implemented.")

            if is_unique:
                unique_clusters_for_type.append(
                    UniqueCluster(
                        molecule_indices=indices,
                        weight=1,
                    )
                )

        all_unique_clusters.extend(unique_clusters_for_type)

    return UniqueClusters(unique_clusters=all_unique_clusters)
