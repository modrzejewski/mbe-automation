"""Tests for molecular cluster extraction."""

import sys
import tempfile
from pathlib import Path

# Ensure the project root is in sys.path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest

import mbe_automation
from mbe_automation import MACE, Structure
from mbe_automation.configs.many_body_expansion import MBE
from mbe_automation.configs.clusters import UniqueClustersFilter
from mbe_automation.common.display import dotted_separator

TESTS_DIR = Path(__file__).parent.parent

DIMER_CUTOFF = 30.0
TRIMER_CUTOFF = 10.0

from tests.reference_data.test_cases import TEST_CASES
TEST_CASES_WITH_MODELS = [c for c in TEST_CASES if c.get("general_model_path") is not None]

def _find_safe_cutoff(distances: np.ndarray, t: float, tolerance: float = 1e-3) -> float:
    """Find a safe cutoff near `t` by locating a gap in the distances of at least `tolerance`.
    Places the cutoff in the middle of the closest valid gap.
    
    Args:
        distances: Array of cluster distances to analyze for gaps.
        t: The target cutoff distance we want to evaluate.
        tolerance: Minimum gap size required to safely place a cutoff (default: 1e-3).
    """
    assert len(distances) >= 2, "Need at least 2 distances to find gaps."
    
    distances = np.sort(distances)
    gaps = np.diff(distances)
    centers = distances[:-1][gaps >= tolerance] + gaps[gaps >= tolerance] / 2.0
    
    # Prefer a gap slightly above t to include clusters exactly on the boundary
    return centers[np.argmin(np.abs(centers - (t + tolerance)))] if len(centers) else t

def _assert_cumulative_weights_equal(
    df_reference: pd.DataFrame, 
    df_generated: pd.DataFrame, 
    max_cutoff: float, 
    cluster_type: str, 
    system_name: str,
    distance_column: str
) -> None:
    """Assert cumulative physical sums match and unique clusters are equal or better.
    
    Args:
        df_reference: Reference dataframe containing 'MaxMinRij' and 'Weight'.
        df_generated: Generated dataframe containing the distance column and 'symmetry_weight (1∕A)'.
        max_cutoff: Maximum distance threshold in angstroms.
        cluster_type: Label for cluster type (e.g., 'dimers').
        system_name: Name of the chemical system being evaluated.
        distance_column: Name of the distance column in df_generated (e.g., 'min_r (Å)').
    """
    
    thresholds = np.arange(0.0, max_cutoff + 0.5, 0.5)
    
    print(f"\nSystem: {system_name} | Cluster: {cluster_type}")
    dotted_separator(120)
    print(f"{'cutoff (Å)':<12} | {'Physical Count':<36} | {'Unique Clusters':<36}")
    print(f"{'':<12} | {'Reference':<17} {'Generated':<16} | {'Reference':<17} {'Generated':<16} {'Status':<8} {'Efficiency':<10}")
    dotted_separator(120)
    
    ref_distances = df_reference['MaxMinRij'].to_numpy()
    
    for t in thresholds:
        safe_t = _find_safe_cutoff(ref_distances, t, tolerance=1e-3)
        
        ref_subset = df_reference[df_reference['MaxMinRij'] <= safe_t]
        gen_subset = df_generated[df_generated[distance_column] <= safe_t]
        
        ref_phys = int(ref_subset['Weight'].sum())
        gen_phys = int(gen_subset['symmetry_weight (1∕A)'].sum())
        
        ref_uniq = len(ref_subset)
        gen_uniq = len(gen_subset)
        
        phys_match = ref_phys == gen_phys
        status = "PASS" if phys_match else "FAIL"
        
        if gen_uniq < ref_uniq:
            efficiency = "BETTER"
        elif gen_uniq == ref_uniq:
            efficiency = "EQUAL"
        else:
            efficiency = "WORSE"
            
        print(f"{t:<12.1f} | {ref_phys:<17} {gen_phys:<16} | {ref_uniq:<17} {gen_uniq:<16} {status:<8} {efficiency:<10}")
        
        assert phys_match, f"Physical count mismatch at {t} A: {ref_phys} vs {gen_phys}"
        assert gen_uniq <= ref_uniq, f"Modern pipeline generated worse symmetry (more unique clusters) at {t} A: {gen_uniq} vs {ref_uniq}"
        
    dotted_separator(120)



def _verify_symmetry_weights(case: dict, work_dir: Path) -> None:
    """Verify generated symmetry weights against reference data.
    
    Args:
        case: Dictionary containing system paths and reference paths.
        work_dir: Directory containing the generated output CSVs.
    """
    csv_dir = work_dir / "csv"
    assert csv_dir.exists(), f"CSV directory missing: {csv_dir}"
    
    generated_dimers_path = csv_dir / "dimers[AA].csv"
    if generated_dimers_path.exists():
        df_generated = pd.read_csv(generated_dimers_path)
        df_reference = pd.read_csv(case["symmetry_weights"]["dimers"], skipinitialspace=True)
        _assert_cumulative_weights_equal(df_reference, df_generated, DIMER_CUTOFF, "dimers", case["name"], "min_r (Å)")
        
    generated_trimers_path = csv_dir / "trimers[AAA].csv"
    if generated_trimers_path.exists():
        df_generated = pd.read_csv(generated_trimers_path)
        df_reference = pd.read_csv(case["symmetry_weights"]["trimers"], skipinitialspace=True)
        _assert_cumulative_weights_equal(df_reference, df_generated, TRIMER_CUTOFF, "trimers", case["name"], "max_min_r (Å)")

def _run_cluster_extraction(case: dict, work_dir: Path) -> None:
    """Execute MBE cluster extraction pipeline for a given test case.
    
    Args:
        case: Dictionary containing system paths and reference paths.
        work_dir: Output directory for the generated clusters.
    """
    crystal = Structure.from_xyz_file(case["crystal_path"])
    calc = MACE(model_path=str(case["general_model_path"])) 

    unique_cluster_filter = UniqueClustersFilter(
        cluster_types=["monomers", "dimers", "trimers"],
        cutoffs={"dimers": DIMER_CUTOFF, "trimers": TRIMER_CUTOFF},
    )

    config = MBE(
        crystal=crystal,
        calculator=calc,
        filter=unique_cluster_filter,
        work_dir=work_dir,
        dataset=work_dir / "properties.hdf5",
        save_xyz=True,
        save_csv=True,
    )

    mbe_automation.run(config)
    _verify_symmetry_weights(case, work_dir)

@pytest.mark.parametrize("case", TEST_CASES_WITH_MODELS, ids=[c["name"] for c in TEST_CASES_WITH_MODELS])
def test_extract_unique_clusters(case: dict, tmp_path: Path):
    """Test full unique cluster extraction pipeline against golden references."""
    _run_cluster_extraction(case, tmp_path)

if __name__ == "__main__":
    for case in TEST_CASES_WITH_MODELS:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                _run_cluster_extraction(case, Path(temp_dir))
        except AssertionError as e:
            print(f"\n{e}")
            sys.exit(1)
