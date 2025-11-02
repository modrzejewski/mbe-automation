from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Dict
import numpy as np
import numpy.typing as npt
import itertools
import time
import scipy
import pymatgen.core
import pymatgen.core.operations
import pymatgen.analysis.molecule_matcher
from . import crystal
from ..storage import MolecularCrystal
from ..common.display import Progress


@dataclass(kw_only=True)
class UniqueClusterFilter:
    """
    Filter for finding unique molecular clusters.
    """
    cluster_types: List[str] = field(default_factory=lambda: ["dimers"])
    cutoffs: Dict[str, float] = field(default_factory=lambda: {"dimers": 5.0})
    alignment_thresh: float = 1.0e-4
    align_mirror_images: bool = True

@dataclass(kw_only=True)
class UniqueClusters:
    """
    Contains arrays of unique molecular clusters and their properties.
    """
    molecule_indices: npt.NDArray[np.integer]
    weights: npt.NDArray[np.integer]
    min_distances: npt.NDArray[np.floating]  # Shape: (n_unique_clusters, n_pairs)
    max_distances: npt.NDArray[np.floating]  # Shape: (n_unique_clusters, n_pairs)

def unique_clusters(
    molecular_crystal: MolecularCrystal,
    unique_cluster_filter: UniqueClusterFilter,
    frame_index: int = 0,
) -> Dict[str, UniqueClusters]:

    if molecular_crystal.supercell.positions.ndim == 3:
        positions_supercell = molecular_crystal.supercell.positions[frame_index]
    else:
        positions_supercell = molecular_crystal.supercell.positions

    def _get_cluster_molecule(indices: npt.NDArray[np.integer]) -> pymatgen.core.Molecule:
        atom_indices = np.concatenate([molecular_crystal.index_map[i] for i in indices])
        positions = positions_supercell[atom_indices, :]
        atomic_numbers = molecular_crystal.supercell.atomic_numbers[atom_indices]
        return pymatgen.core.Molecule(species=atomic_numbers, coords=positions)

    def _are_distances_similar(dists1: np.ndarray, dists2: np.ndarray, threshold: float) -> bool:
        if len(dists1) != len(dists2):
            return False
        return np.max(np.abs(dists1 - dists2)) < threshold

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

    # Hybrid distance calculation
    n_mols = molecular_crystal.n_molecules
    min_rij = np.full((n_mols, n_mols), np.inf)
    max_rij = np.full((n_mols, n_mols), np.inf)

    if len(candidate_molecule_indices) > 1:
        # Extract all atomic positions for candidate molecules into a single array
        all_positions = []
        mol_slices = [0]

        for mol_idx in candidate_molecule_indices:
            atom_indices = molecular_crystal.index_map[mol_idx]
            positions = positions_supercell[atom_indices, :]
            all_positions.append(positions)
            mol_slices.append(mol_slices[-1] + len(atom_indices))

        mol_slices = np.array(mol_slices)
        all_positions = np.concatenate(all_positions, axis=0)

        # Calculate all atom-atom distances at once
        dist_matrix = scipy.spatial.distance.cdist(all_positions, all_positions)

        # Now, extract the min and max distances for each pair of molecules using a simple loop
        for i in range(len(candidate_molecule_indices)):
            for j in range(i + 1, len(candidate_molecule_indices)):
                mol_idx1 = candidate_molecule_indices[i]
                mol_idx2 = candidate_molecule_indices[j]

                start1, end1 = mol_slices[i], mol_slices[i+1]
                start2, end2 = mol_slices[j], mol_slices[j+1]

                sub_matrix = dist_matrix[start1:end1, start2:end2]
                min_dist = np.min(sub_matrix)
                max_dist = np.max(sub_matrix)

                min_rij[mol_idx1, mol_idx2] = min_dist
                min_rij[mol_idx2, mol_idx1] = min_dist
                max_rij[mol_idx1, mol_idx2] = max_dist
                max_rij[mol_idx2, mol_idx1] = max_dist

    results = {}

    for cluster_type in unique_cluster_filter.cluster_types:
        cutoff = unique_cluster_filter.cutoffs[cluster_type]
        n = cluster_size_map[cluster_type]

        #
        # Create a list of all possible clusters of size n using only the pre-filtered candidate molecules.
        #
        central_molecule = molecular_crystal.central_molecule_index
        other_candidate_molecules = [i for i in candidate_molecule_indices if i != central_molecule]

        total_combinations = scipy.special.comb(len(other_candidate_molecules), n - 1, exact=True)

        combinations_iterator = itertools.combinations(other_candidate_molecules, n - 1)

        print(f"Computing unique {cluster_type}s with cutoff < {cutoff:.2f} Å")
        progress_iterable = Progress(
            combinations_iterator,
            n_total_steps=total_combinations,
            label=f"{cluster_type}s"
        )

        unique_indices_list = []
        unique_weights_list = []
        unique_matchers_list = []
        unique_min_dists_list = []
        unique_max_dists_list = []

        for combo in progress_iterable:
            indices = np.array([central_molecule] + list(combo))

            # Filter clusters based on the cutoff distance
            within_cutoff = True
            for i, j in itertools.combinations(indices, 2):
                if min_rij[i, j] >= cutoff:
                    within_cutoff = False
                    break
            if not within_cutoff:
                continue

            min_dists = np.sort([min_rij[p1, p2] for p1, p2 in itertools.combinations(indices, 2)])
            max_dists = np.sort([max_rij[p1, p2] for p1, p2 in itertools.combinations(indices, 2)])

            is_unique = True
            for i, unique_matcher in enumerate(unique_matchers_list):
                if _are_distances_similar(min_dists, unique_min_dists_list[i], unique_cluster_filter.alignment_thresh) and \
                   _are_distances_similar(max_dists, unique_max_dists_list[i], unique_cluster_filter.alignment_thresh):

                    cluster_mol = _get_cluster_molecule(indices)
                    _, dist = unique_matcher.fit(cluster_mol)

                    if dist > unique_cluster_filter.alignment_thresh and unique_cluster_filter.align_mirror_images:
                        mirrored_cluster_mol = cluster_mol.copy()
                        mirrored_cluster_mol.apply_operation(
                            pymatgen.core.operations.SymmOp.from_reflection((0, 1, 0))
                        )
                        _, dist = unique_matcher.fit(mirrored_cluster_mol)

                    if dist < unique_cluster_filter.alignment_thresh:
                        is_unique = False
                        unique_weights_list[i] += 1
                        break

            if is_unique:
                unique_indices_list.append(indices)
                unique_weights_list.append(1)
                unique_min_dists_list.append(min_dists)
                unique_max_dists_list.append(max_dists)
                cluster_mol = _get_cluster_molecule(indices)
                unique_matchers_list.append(
                    pymatgen.analysis.molecule_matcher.HungarianOrderMatcher(cluster_mol)
                )

        if not unique_indices_list:
            continue

        # Convert lists to NumPy arrays
        molecule_indices_arr = np.array(unique_indices_list)
        weights_arr = np.array(unique_weights_list)

        # Handle cases with no internal distances (monomers)
        if n == 1:
            unique_min_dists_list = [[0]] * len(unique_indices_list)
            unique_max_dists_list = [[0]] * len(unique_indices_list)

        results[cluster_type] = UniqueClusters(
            molecule_indices=molecule_indices_arr,
            weights=weights_arr,
            min_distances=np.array(unique_min_dists_list),
            max_distances=np.array(unique_max_dists_list)
        )

    return results
