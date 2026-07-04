"""Tests for molecular cluster extraction."""

from pathlib import Path
import pytest
from mbe_automation.api.classes import MolecularComposition
from mbe_automation.configs.clusters import UniqueClustersFilter

def test_extract_unique_clusters_helicene():
    """Extract symmetry-unique clusters from a 2x2x2 supercell of helicene."""
    cif_path = Path(__file__).parent / "helicene.cif"
    comp = MolecularComposition.from_xyz_file(cif_path, match_mode="rmsd_only", rmsd_thresh=0.1)
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
