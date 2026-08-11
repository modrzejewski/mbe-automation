"""Tests for molecular cluster extraction."""

import os
from pathlib import Path
import tempfile
import pytest
import pandas as pd

import mbe_automation
from mbe_automation import MACE, Structure
from mbe_automation.configs.many_body_expansion import MBE
from mbe_automation.configs.clusters import UniqueClustersFilter

TESTS_DIR = Path(__file__).parent.parent
MATCH_ALGO = "irmsd"

from tests.reference_data.test_cases import TEST_CASES
TEST_CASES_WITH_MODELS = [c for c in TEST_CASES if c["model_path"] is not None]

@pytest.mark.parametrize("case", TEST_CASES_WITH_MODELS, ids=[c["name"] for c in TEST_CASES_WITH_MODELS])
def test_extract_unique_clusters(case):
    _run_cluster_extraction(case)

def _verify_xyz_files(work_dir: Path):
    csv_dir = work_dir / "csv"
    xyz_dir = work_dir / "xyz"
    
    assert csv_dir.exists(), f"CSV directory missing: {csv_dir}"
    assert xyz_dir.exists(), f"XYZ directory missing: {xyz_dir}"
    
    for csv_file in csv_dir.glob("*.csv"):
        cluster_type = csv_file.stem
        df = pd.read_csv(csv_file)
        
        assert "system" in df.columns, f"'system' column missing in {csv_file}"
        
        for system_label in df["system"]:
            xyz_file = xyz_dir / cluster_type / f"{system_label}.xyz"
            assert xyz_file.exists(), f"Missing XYZ file for {system_label} in {cluster_type}"

def _run_cluster_extraction(case, work_dir=None):
    crystal = Structure.from_xyz_file(case["crystal_path"])
    calc = MACE(model_path=str(case["model_path"])) 

    unique_cluster_filter = UniqueClustersFilter(
        cluster_types=["monomers", "dimers", "trimers"],
        cutoffs={"dimers": 15.0, "trimers": 10.0},
        algorithm=MATCH_ALGO,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(work_dir) if work_dir else Path(temp_dir)
        
        config = MBE(
            crystal=crystal,
            calculator=calc,
            filter=unique_cluster_filter,
            work_dir=output_dir,
            dataset=output_dir / "properties.hdf5",
            save_xyz=True,
            save_csv=True,
        )

        mbe_automation.run(config)
        _verify_xyz_files(output_dir)

if __name__ == "__main__":
    results = []
    
    for case in TEST_CASES:
        try:
            _run_cluster_extraction(case)
            results.append((case['name'], "PASS"))
        except Exception as e:
            results.append((case['name'], f"FAIL ({type(e).__name__})"))
            
    print("\nTest Summary:")
    for name, status in results:
        print(f"{name:<25} {status}")
