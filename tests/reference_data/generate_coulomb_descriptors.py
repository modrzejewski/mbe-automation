import sys
import zipfile
import ase.io
from io import StringIO
from pathlib import Path
import numpy as np

try:
    from dscribe.descriptors import CoulombMatrix
except ImportError:
    print("Error: dscribe is required to run this script. Please install it.", file=sys.stderr)
    sys.exit(1)

from tqdm import tqdm

# Ensure the project root is in sys.path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.reference_data.test_cases import TEST_CASES, REFERENCE_DATA_DIR

def process_clusters(case: dict, cluster_type: str):
    sym_weights_csv = case["symmetry_weights"][cluster_type]
    case_dir = Path(sym_weights_csv).parent.parent
    zip_path = case_dir / f"{cluster_type}.zip"
    
    if not zip_path.exists():
        return
        
    out_npz = Path(case["descriptors"][cluster_type])
    out_npz.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"  [RUNNING] {cluster_type} -> {out_npz.relative_to(PROJECT_ROOT / 'tests')}")
    
    descriptors_dict = {}
    cm_calculator = None
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        sorted_namelist = sorted([n for n in z.namelist() if n.endswith('.xyz')])
        
        for filename in tqdm(sorted_namelist, desc=f"  Processing {cluster_type}"):
            try:
                system_id = filename.split('-')[0] # e.g. "000"
                
                xyz_data = z.read(filename).decode('utf-8').strip()
                atoms = ase.io.read(StringIO(xyz_data), format='xyz')
                
                if cm_calculator is None:
                    # Initialize CoulombMatrix for the exact size of the cluster
                    cm_calculator = CoulombMatrix(
                        n_atoms_max=len(atoms), 
                        permutation="eigenspectrum"
                    )
                
                # Create descriptor and explicitly flatten
                descriptor = cm_calculator.create(atoms).flatten()
                
                descriptors_dict[system_id] = descriptor
                
            except Exception as e:
                print(f"    Error processing {filename}: {e}", file=sys.stderr)
                
    if descriptors_dict:
        np.savez_compressed(out_npz, **descriptors_dict)
        print(f"  [SUCCESS] Wrote {len(descriptors_dict)} descriptors to {out_npz.relative_to(PROJECT_ROOT / 'tests')}")

def main():
    print("Starting Coulomb descriptor generation script...")
    
    for case in TEST_CASES:
        system_name = case["name"]
        print(f"\nProcessing system: {system_name}")
        
        process_clusters(case, "dimers")
        process_clusters(case, "trimers")

if __name__ == "__main__":
    main()
