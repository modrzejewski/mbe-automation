from __future__ import annotations
import numpy as np
import pymatgen.core
import pymatgen.analysis.molecule_matcher

def get_rmsd(
    mol1: pymatgen.core.Molecule,
    mol2: pymatgen.core.Molecule,
    threshold: float = 0.5
) -> float:
    """
    Calculate the minimum RMSD between two molecules, accounting for atom reordering.

    Uses GeneticOrderMatcher with a fallback to HungarianOrderMatcher for complex cases.

    Args:
        mol1: The first pymatgen Molecule object.
        mol2: The second pymatgen Molecule object.
        threshold: The RMSD threshold for the GeneticOrderMatcher.

    Returns:
        The minimum RMSD between the two molecules.
    """
    gom = pymatgen.analysis.molecule_matcher.GeneticOrderMatcher(mol1, threshold=threshold)
    pairs = gom.fit(mol2)
    
    rmsd = threshold
    if pairs:
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
    else:
        hom = pymatgen.analysis.molecule_matcher.HungarianOrderMatcher(mol1)
        _, rmsd = hom.fit(mol2)

    return rmsd
