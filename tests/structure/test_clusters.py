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
X23_DIR = TESTS_DIR / "xyz" / "X23"
MACE_MODELS_DIR = TESTS_DIR / "mace_models" / "Pia_et_al_Chem_Sci_2025"

TEST_CASES = [
    {
        "name": "1,4-cyclohexanedione",
        "crystal_path": X23_DIR / "01_1,4-cyclohexanedione" / "solid.xyz",
        "model_path": MACE_MODELS_DIR / "01_cyclohexanedione" / "MACE_model_swa.model"
    },
    {
        "name": "acetic_acid",
        "crystal_path": X23_DIR / "02_acetic_acid" / "solid.xyz",
        "model_path": MACE_MODELS_DIR / "02_acetic_acid" / "MACE_model_swa.model"
    },
    {
        "name": "adamantane",
        "crystal_path": X23_DIR / "03_adamantane" / "solid.xyz",
        "model_path": MACE_MODELS_DIR / "03_adamantane" / "MACE_model_swa.model"
    },
    {
        "name": "ammonia",
        "crystal_path": X23_DIR / "04_ammonia" / "solid.xyz",
        "model_path": MACE_MODELS_DIR / "04_ammonia" / "MACE_model_swa.model"
    },
    {
        "name": "anthracene",
        "crystal_path": X23_DIR / "05_anthracene" / "solid.xyz",
        "model_path": MACE_MODELS_DIR / "05_anthracene" / "MACE_model_swa.model"
    },
    {
        "name": "benzene",
        "crystal_path": X23_DIR / "06_benzene" / "solid.xyz",
        "model_path": MACE_MODELS_DIR / "06_benzene" / "MACE_model_swa.model"
    }
]

@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
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
        cutoffs={"dimers": 15.0, "trimers": 10.0} 
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
