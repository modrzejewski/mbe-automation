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
MATCH_ALGO = "irmsd"

from tests.reference_data.test_cases import TEST_CASES
TEST_CASES_WITH_MODELS = [c for c in TEST_CASES if c.get("general_model_path") is not None]

def _assert_cumulative_weights_equal(
    df_reference: pd.DataFrame, 
    df_generated: pd.DataFrame, 
    max_cutoff: float, 
    cluster_type: str, 
    system_name: str,
    distance_column: str
) -> None:
    """Assert cumulative cluster sums match perfectly and print diagnostic tables.
    
    Args:
        df_reference: Reference dataframe containing 'MaxMinRij' and 'Weight'.
        df_generated: Generated dataframe containing the distance column and 'symmetry_weight (1∕A)'.
        max_cutoff: Maximum distance threshold in angstroms.
        cluster_type: Label for cluster type (e.g. 'dimers').
        system_name: Name of the chemical system being evaluated.
        distance_column: The name of the distance column in df_generated (e.g. 'min_r (Å)').
    """
    thresholds = np.arange(0.0, max_cutoff + 0.5, 0.5)
    
    ref_cumulative = []
    gen_cumulative = []
    
    for t in thresholds:
        ref_cumulative.append(df_reference.loc[df_reference['MaxMinRij'] <= t, 'Weight'].sum())
        gen_cumulative.append(df_generated.loc[df_generated[distance_column] <= t, 'symmetry_weight (1∕A)'].sum())
        
    print(f"\nSystem: {system_name} | Cluster: {cluster_type}")
    dotted_separator(71)
    print(f"{'cutoff (Å)':<12} {'reference count':<17} {'generated count':<17} {'status':<8}")
    dotted_separator(71)
    
    for t, ref_val, gen_val in zip(thresholds, ref_cumulative, gen_cumulative):
        status = "PASS" if ref_val == gen_val else "FAIL"
        print(f"{t:<12.1f} {int(ref_val):<17} {int(gen_val):<17} {status:<8}")
        
    dotted_separator(71)
    
    npt.assert_array_equal(
        gen_cumulative, 
        ref_cumulative, 
        err_msg="The cumulative cluster sums up to the given cutoffs do not match between the reference and generated pipelines!"
    )

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
        _assert_cumulative_weights_equal(df_reference, df_generated, 15.0, "dimers", case["name"], "min_r (Å)")
        
    generated_trimers_path = csv_dir / "trimers[AAA].csv"
    if generated_trimers_path.exists():
        df_generated = pd.read_csv(generated_trimers_path)
        df_reference = pd.read_csv(case["symmetry_weights"]["trimers"], skipinitialspace=True)
        _assert_cumulative_weights_equal(df_reference, df_generated, 10.0, "trimers", case["name"], "max_min_r (Å)")

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
        cutoffs={"dimers": 15.0, "trimers": 10.0},
        algorithm=MATCH_ALGO,
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
