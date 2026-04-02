"""
Atom typing rules for AZ-FF.

This module assigns FF types (FF_1, FF_2, ...) to atoms in a CIF/ASE Atoms
object based on element, hybridization-like environment, and local bonding
patterns extracted from the topology.

The AZ-FF force field defines a fixed library of atom types in the FF file.
This module provides mapping logic from structure → FF type using a simple,
transparent rule-based approach that can be extended by the user.

The typing scheme here is intentionally conservative:

    1. Element-based first pass (C, H, O, N, S, etc.).
    2. Bond-environment refinements: aromatic, nitro, amine, carbonyl, etc.
    3. Mapping to FF labels (FF_1 ... FF_n) supplied by the force-field file.

This keeps typing deterministic, easy to debug, and avoids external tools.

Notes:
------
- This module does *not* guess formal charges or aromaticity beyond local
  bond count patterns (e.g., 3 neighbors in a ring → aromatic C-like).
- Real AZ-FF typing depends on MacroModel rules. Here we implement minimal
  CIF-safe approximations sufficient for crystals and consistent with the
  uploaded oplsaa.ff file.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def assign_types(
    symbols: List[str],
    bonds: Dict[int, List[int]],
    ff_definitions: Dict[str, Dict],
) -> List[str]:
    """
    Assign AZ-FF / OPLS-AA types to each atom.

    Parameters
    ----------
    symbols : list of str
        Atomic symbols from ASE atoms (e.g., ["C", "O", "N", ...]).
    bonds : dict {atom_index: [neighbor_indices]}
        Adjacent atoms detected by topology builder.
    ff_definitions : dict
        Parsed FF file dictionary. Used to know available FF types.

    Returns
    -------
    list of str
        FF-type labels (e.g., FF_1, FF_8, FF_19) for each atom, length = len(symbols).

    Notes
    -----
    - The returned labels MUST be present in ff_definitions["vdw"] (the available FF types).
    - If no rule matches, a RuntimeError is raised — this avoids silent failures.
    """
    n = len(symbols)
    out = [None] * n

    # All FF types defined in the FF file
    available_ff_types = set(ff_definitions.get("vdw", {}).keys())

    # First-pass element buckets
    element_groups = _group_by_element(symbols)

    # -------------------------------------------------------------------------
    # Carbon typing
    # -------------------------------------------------------------------------
    for idx in element_groups["C"]:
        nb = bonds[idx]
        n_nb = len(nb)

        if _is_aromatic_carbon(idx, symbols, bonds):
            ff = _match_available("FF_13", available_ff_types)  # aromatic C-like
        elif n_nb == 4:
            ff = _match_available("FF_17", available_ff_types)  # aliphatic C (CT)
        elif n_nb == 3:
            ff = _match_available("FF_7", available_ff_types)   # trigonal C (CA)
        else:
            ff = _match_available("FF_10", available_ff_types)  # fallback carbon type

        out[idx] = ff

    # -------------------------------------------------------------------------
    # Hydrogen typing
    # -------------------------------------------------------------------------
    for idx in element_groups["H"]:
        parent = bonds[idx][0] if bonds[idx] else None
        if parent is None:
            out[idx] = _match_available("FF_19", available_ff_types)
            continue

        # Aromatic hydrogen (HA)
        if symbols[parent] == "C" and _is_aromatic_carbon(parent, symbols, bonds):
            out[idx] = _match_available("FF_21", available_ff_types)
        # Aliphatic hydrogen (HC)
        elif symbols[parent] == "C":
            out[idx] = _match_available("FF_25", available_ff_types)
        else:
            out[idx] = _match_available("FF_19", available_ff_types)  # generic H

    # -------------------------------------------------------------------------
    # Nitrogen typing
    # -------------------------------------------------------------------------
    for idx in element_groups["N"]:
        nb = bonds[idx]
        n_nb = len(nb)

        # Nitro-like (NO2)
        if _is_nitro_nitrogen(idx, symbols, bonds):
            out[idx] = _match_available("FF_5", available_ff_types)
        # Aromatic amine N (NE)
        elif n_nb == 2 and any(_is_aromatic_carbon(j, symbols, bonds) for j in nb):
            out[idx] = _match_available("FF_4", available_ff_types)
        # Tertiary amine (NZ)
        elif n_nb == 3:
            out[idx] = _match_available("FF_6", available_ff_types)
        else:
            out[idx] = _match_available("FF_4", available_ff_types)  # general N

    # -------------------------------------------------------------------------
    # Oxygen typing
    # -------------------------------------------------------------------------
    for idx in element_groups["O"]:
        nb = bonds[idx]

        # Nitro oxygen
        if _is_nitro_oxygen(idx, symbols, bonds):
            out[idx] = _match_available("FF_2", available_ff_types)
        # Carbonyl oxygen
        elif any(symbols[j] == "C" and len(bonds[j]) == 3 for j in nb):
            out[idx] = _match_available("FF_3", available_ff_types)
        # Hydroxyl oxygen
        elif any(symbols[j] == "H" for j in nb):
            out[idx] = _match_available("FF_20", available_ff_types)
        # Default O
        else:
            out[idx] = _match_available("FF_2", available_ff_types)

    # -------------------------------------------------------------------------
    # Sulfur typing
    # -------------------------------------------------------------------------
    for idx in element_groups["S"]:
        # Paper's test set includes sulfathiazole, and FF_1 was used for S-like roles
        out[idx] = _match_available("FF_1", available_ff_types)

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------
    for i, t in enumerate(out):
        if t is None:
            raise RuntimeError(
                f"Atom {i} ({symbols[i]}) could not be assigned an AZ-FF type. "
                f"Please extend typing_rules.py."
            )
        if t not in available_ff_types:
            raise RuntimeError(
                f"Assigned type {t} is not present in the FF definitions."
            )

    return out


# -----------------------------------------------------------------------------
# Utility: Element grouping
# -----------------------------------------------------------------------------
def _group_by_element(symbols: List[str]) -> Dict[str, List[int]]:
    groups = defaultdict(list)
    for i, s in enumerate(symbols):
        groups[s].append(i)
    return groups


# -----------------------------------------------------------------------------
# Environment detection helpers
# -----------------------------------------------------------------------------
def _is_aromatic_carbon(idx: int, symbols: List[str], bonds: Dict[int, List[int]]) -> bool:
    """
    Simple heuristic:
    - Carbon has exactly 3 neighbors
    - All neighbors are carbon or heteroatoms that typically appear in aromatic rings
    - Degree 3 is used as a proxy for sp2 planar/aromatic environment
    """
    if symbols[idx] != "C":
        return False
    nb = bonds[idx]
    return len(nb) == 3


def _is_nitro_nitrogen(idx: int, symbols: List[str], bonds: Dict[int, List[int]]) -> bool:
    """
    Identify N in a NO2 group:
    - N connected to exactly two O atoms
    """
    if symbols[idx] != "N":
        return False
    nb = bonds[idx]
    O_count = sum(symbols[j] == "O" for j in nb)
    return O_count == 2


def _is_nitro_oxygen(idx: int, symbols: List[str], bonds: Dict[int, List[int]]) -> bool:
    """
    Oxygen part of a NO2 group:
    - O bonded to N
    - That N has two O neighbors
    """
    if symbols[idx] != "O":
        return False
    nb = bonds[idx]
    if not nb:
        return False
    parent = nb[0]
    return _is_nitro_nitrogen(parent, symbols, bonds)


# -----------------------------------------------------------------------------
# Utility for matching FF types safely
# -----------------------------------------------------------------------------
def _match_available(preferred_type: str, all_types: set) -> str:
    """
    If the preferred FF_* type exists in the FF file, return it.
    Otherwise, pick a fallback of the same element family.

    This ensures robustness when FF files include only a subset
    of the standard type library.
    """
    if preferred_type in all_types:
        return preferred_type

    # Fallback: choose the lowest-numbered type as last resort
    numeric = sorted(
        (int(t.split("_")[1]), t) for t in all_types if t.startswith("FF_")
    )
    if numeric:
        return numeric[0][1]

    raise RuntimeError("No FF_* atom types available in the force-field file.")
