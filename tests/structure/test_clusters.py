"""Tests for molecular cluster extraction."""

from pathlib import Path
import pytest
from mbe_automation.api.classes import MolecularComposition, Structure
from mbe_automation.configs.clusters import UniqueClustersFilter
from mbe_automation import MACE

def extract_unique_clusters(cif_path: str | Path):
    """Extract symmetry-unique clusters from a 2x2x2 supercell."""
    cif_path = Path(cif_path)
    crystal = Structure.from_xyz_file(cif_path)
    calc = MACE(
        model_path="~/models/mace/mace-mh-1.model", 
        head="omol"
    )
    
    comp = MolecularComposition(
        crystal=crystal,
        calculator=calc,
    )
    supercell = comp.expand_to_supercell([2, 2, 2])
    
    unique_cluster_filter = UniqueClustersFilter(
        cluster_types=["monomers", "dimers", "trimers"],
        cutoffs={"monomers": 30.0, "dimers": 15.0, "trimers": 10.0}
    )
    
    clusters = supercell.symmetry_unique_clusters(
        unique_cluster_filter=unique_cluster_filter,
    )
    
    assert clusters is not None
    assert isinstance(clusters, dict)
    assert len(clusters) > 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target_cif = Path(sys.argv[1])
    else:
        target_cif = Path(__file__).parent / "helicene.cif"
        
    extract_unique_clusters(target_cif)
