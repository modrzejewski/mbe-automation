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

    all_unique_clusters = []

    for cluster_type in unique_cluster_filter.cluster_types:
        cutoff = unique_cluster_filter.cutoffs[cluster_type]
        n = cluster_size_map[cluster_type]

        #
        # Create a list of all possible clusters of size n
        #
        molecules_in_supercell = molecular_crystal.n_molecules
        central_molecule = molecular_crystal.central_molecule_index

        other_molecules = [i for i in range(molecules_in_supercell) if i != central_molecule]

        all_clusters_indices = []
        for combo in itertools.combinations(other_molecules, n - 1):
            all_clusters_indices.append(np.array([central_molecule] + list(combo)))

        #
        # Filter clusters based on the cutoff distance
        #
        filtered_clusters_indices = []
        for indices in all_clusters_indices:
            within_cutoff = True
            for i, j in itertools.combinations(indices, 2):
                if molecular_crystal.min_distances_to_central_molecule[j] >= cutoff:
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

                    if dist < unique_cluster_filter.alignment_thresh:
                        is_unique = False
                        unique_cluster.weight += 1
                        break

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
