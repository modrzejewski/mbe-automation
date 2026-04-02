"""
Topology construction for AZ-FF.

This module detects:
    - Bonds (1–2 pairs)
    - Angles (1–3 pairs)
    - Dihedrals (1–4 paths)
    - 1–3 exclusions
    - 1–4 pairs (scaled interactions)

The logic is purely geometric and connectivity-based. It does NOT infer
chemistry beyond covalent connectivity.

We rely on:
    * ASE Atoms interface
    * Covalent radii with a tolerance factor

This keeps the topology general and stable for CIF-based unit-cell
structures, exactly as needed for the AZ-FF solid-state energy model.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


# Default bond expansion factor used in many FF tools (OpenMM, RDKit, ASE).
# Safe for CIF crystals, where bond lengths may vary slightly.
BOND_TOL = 1.25


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_topology(
    atoms: Atoms,
    bond_factor: float = BOND_TOL,
) -> Dict:
    """
    Build bonded topology for an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure with positions, cell, and (optional) PBC.
    bond_factor : float
        Scaling factor multiplying covalent radius sums.

    Returns
    -------
    dict with keys:
        "bonds"     : set of (i, j)
        "angles"    : list of (i, j, k)
        "dihedrals" : list of (i, j, k, l)
        "pairs12"   : set of (i, j)
        "pairs13"   : set of (i, j)
        "pairs14"   : set of (i, j)
        "neighbors" : adjacency list {i: [j1, j2, ...]}
    """
    n = len(atoms)
    bonds = detect_bonds(atoms, factor=bond_factor)
    neighbors = _build_adjacency(n, bonds)

    angles = generate_angles(neighbors)
    dihedrals = generate_dihedrals(neighbors)

    pairs12 = bonds.copy()
    pairs13 = _pairs_13(neighbors)
    pairs14 = _pairs_14(neighbors)

    return {
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "pairs12": pairs12,
        "pairs13": pairs13,
        "pairs14": pairs14,
        "neighbors": neighbors,
    }


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------
def detect_bonds(atoms: Atoms, factor: float = BOND_TOL) -> Set[Tuple[int, int]]:
    """
    Determine covalent bonds using covalent radii and a scale factor.

    Parameters
    ----------
    atoms : ase.Atoms
    factor : float
        Scaling factor applied to (r_i + r_j).

    Returns
    -------
    set of (i, j)
    """
    n = len(atoms)
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    # ASE neighbor search: using brute-force minimum-image distances
    bonds = set()

    for i in range(n):
        Ri = covalent_radii[atoms.numbers[i]]
        for j in range(i + 1, n):
            Rj = covalent_radii[atoms.numbers[j]]

            cutoff = factor * (Ri + Rj)
            rij = pos[j] - pos[i]

            # Apply minimum-image if PBC is on
            if any(pbc):
                rij = rij - np.dot(np.round(np.linalg.solve(cell.T, rij)), cell)

            dist = np.linalg.norm(rij)
            if dist <= cutoff:
                bonds.add((i, j))

    return bonds


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------
def _build_adjacency(n: int, bonds: Set[Tuple[int, int]]) -> Dict[int, List[int]]:
    neighbors = {i: [] for i in range(n)}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return neighbors


# ---------------------------------------------------------------------------
# Angle construction
# ---------------------------------------------------------------------------
def generate_angles(neighbors: Dict[int, List[int]]) -> List[Tuple[int, int, int]]:
    """
    Generate angles i–j–k where j is the central atom.

    Returns
    -------
    list of (i, j, k)
    """
    angles = []
    for j, nb in neighbors.items():
        ln = len(nb)
        for a in range(ln):
            for b in range(a + 1, ln):
                i = nb[a]
                k = nb[b]
                angles.append((i, j, k))
                angles.append((k, j, i))
    return angles


# ---------------------------------------------------------------------------
# Dihedral construction
# ---------------------------------------------------------------------------
def generate_dihedrals(
    neighbors: Dict[int, List[int]],
) -> List[Tuple[int, int, int, int]]:
    """
    Generate simple 1–2–3–4 torsions by scanning adjacency lists.

    Conditions applied:
    - 2–3 must be bonded
    - 1 must be neighbor of 2 (but not 3)
    - 4 must be neighbor of 3 (but not 2)
    - No repeated indices

    Returns
    -------
    list of (i, j, k, l)
    """
    dihedrals = []
    visited = set()

    for j, nb_j in neighbors.items():
        for k in nb_j:
            # Ensure j<k to avoid double-processing
            if j == k or (k, j) in visited:
                continue
            visited.add((j, k))

            # i are neighbors of j except k
            for i in neighbors[j]:
                if i == k:
                    continue

                # l are neighbors of k except j
                for l in neighbors[k]:
                    if l == j or l == i:
                        continue

                    dihedrals.append((i, j, k, l))

    return dihedrals


# ---------------------------------------------------------------------------
# 1–3 pairs (connected by two bonds)
# ---------------------------------------------------------------------------
def _pairs_13(neighbors: Dict[int, List[int]]) -> Set[Tuple[int, int]]:
    """
    Collect 1–3 pairs: atoms that share a central atom.

    Returns
    -------
    set of (i, k)
    """
    out = set()
    for j, nb in neighbors.items():
        ln = len(nb)
        for a in range(ln):
            for b in range(a + 1, ln):
                i = nb[a]
                k = nb[b]
                pair = tuple(sorted((i, k)))
                out.add(pair)
    return out


# ---------------------------------------------------------------------------
# 1–4 pairs (dihedral neighbors)
# ---------------------------------------------------------------------------
def _pairs_14(neighbors: Dict[int, List[int]]) -> Set[Tuple[int, int]]:
    """
    Collect 1–4 pairs from the dihedral list.

    1–4 are atoms connected i–j–k–l with no shortcut bond (i.e., not 1–2 or 1–3).

    Returns
    -------
    set of (i, l)
    """
    pairs = set()
    dihed = generate_dihedrals(neighbors)

    # Exclusion logic: i/l cannot be 1–2 or 1–3 pairs
    pairs12 = set()
    for j, nb in neighbors.items():
        for i in nb:
            pairs12.add(tuple(sorted((i, j))))

    pairs13 = _pairs_13(neighbors)

    for i, j, k, l in dihed:
        pair = tuple(sorted((i, l)))
        if pair in pairs12:
            continue
        if pair in pairs13:
            continue
        pairs.add(pair)

    return pairs
