import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to sys.path so we can import modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from test_cases import TEST_CASES
import mbe_automation.directory_structure as directory_structure
import mbe_automation.mbe_legacy as mbe

DIMERS_CUTOFF = 30.0
TRIMERS_CUTOFF = 10.0

def main():
    SystemTypes = ["dimers", "trimers"]
    Cutoffs = {"dimers": DIMERS_CUTOFF, "trimers": TRIMERS_CUTOFF, "tetramers": 0.0, "ghosts": 4.0}
    Ordering = "MaxMinRij"
    ClusterComparisonAlgorithm = "RMSD"
    SymmetrizeUnitCell = True
    
    # We pass a dummy method so that mbe.Make can resolve directory_structure.CSV_DIRS["DUMMY"]
    MethodsMBE = ["DUMMY"]
    
    reference_data_dir = Path(__file__).resolve().parent

    for case in TEST_CASES:
        system_name = case['name']
        print(f"\nGenerating reference clusters for {system_name}...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            ProjectDirectory = Path(temp_dir) / system_name
            # We source the input XYZ directly from the TEST_CASES dictionary
            # e.g., tests/xyz/X23/01_1,4-cyclohexanedione/solid.xyz
            UnitCellFile = str(case['crystal_path'])
            
            directory_structure.SetUp(
                ProjectDirectory,
                MethodsMBE,
                []
            )
            
            mbe.Make(
                UnitCellFile,
                Cutoffs,
                SystemTypes,
                False, # MonomerRelaxation
                False, # PBCEmbedding
                "",    # RelaxedMonomerXYZ
                Ordering,
                ProjectDirectory,
                directory_structure.XYZ_DIRS,
                directory_structure.CSV_DIRS,
                MethodsMBE,
                SymmetrizeUnitCell,
                ClusterComparisonAlgorithm
            )

            for cluster_type in SystemTypes:
                # The CSV output for a specific cluster type
                csv_file = Path(directory_structure.CSV_DIRS["DUMMY"][cluster_type]) / "systems.csv"
                
                # Destination reference CSV (path sourced from test_cases.py which now has the proper prefix)
                dest_file = case["symmetry_weights"][cluster_type]
                
                # Ensure the destination directory exists
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                if csv_file.exists():
                    shutil.copy2(csv_file, dest_file)
                    print(f"  Copied reference for {cluster_type} -> {dest_file}")
                else:
                    print(f"  WARNING: Output CSV for {cluster_type} not found at {csv_file}")
                
                # Archive the XYZ geometries
                src_xyz_dir = directory_structure.XYZ_DIRS[cluster_type]
                dest_zip_base = dest_file.parent.parent / cluster_type
                
                if Path(src_xyz_dir).exists() and any(Path(src_xyz_dir).iterdir()):
                    shutil.make_archive(str(dest_zip_base), 'zip', src_xyz_dir)
                    print(f"  Archived XYZ geometries -> {dest_zip_base}.zip")

if __name__ == "__main__":
    main()
