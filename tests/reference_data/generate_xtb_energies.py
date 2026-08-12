import pandas as pd
import zipfile
import ase.io
from io import StringIO
from pathlib import Path
import sys

try:
    from tblite.ase import TBLite
except ImportError:
    print("Error: tblite is required to run this script. Please install it.", file=sys.stderr)
    sys.exit(1)

# Import TEST_CASES from the test_cases module
from tests.reference_data.test_cases import TEST_CASES, REFERENCE_DATA_DIR

def main():
    print("Starting energy generation script...")
    
    for case in TEST_CASES:
        system_name = case["name"]
        print(f"\nProcessing system: {system_name}")
        
        # Test cases might have dimers and trimers
        for cluster_type in ["dimers", "trimers"]:
            csv_path = case["symmetry_weights"].get(cluster_type)
            if not csv_path:
                continue
                
            # The zip file should be in the same directory as the csv, named dimers.zip or trimers.zip
            zip_path = csv_path.parent.parent / f"{cluster_type}.zip"
            
            if not zip_path.exists():
                print(f"  [SKIPPED] {cluster_type} - Zip file missing: {zip_path}")
                continue
                
            out_csv = csv_path.parent.parent / "energies" / f"{cluster_type}_gfn2-xtb.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"  [RUNNING] {cluster_type} -> {out_csv.name}")
            
            results = []
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Sort to process systematically
                sorted_namelist = sorted([n for n in z.namelist() if n.endswith('.xyz')])
                total_files = len(sorted_namelist)
                
                for i, filename in enumerate(sorted_namelist):
                    if i > 0 and i % 10 == 0:
                        print(f"    Processed {i}/{total_files} clusters...", flush=True)
                        
                    try:
                        # Extract the System ID from the prefix (e.g. '000-dimer-0000-0001.xyz')
                        system_id = int(filename.split('-')[0])
                        
                        xyz_data = z.read(filename).decode('utf-8')
                        atoms = ase.io.read(StringIO(xyz_data), format='xyz')
                        
                        # Assign TBLite GFN2-xTB calculator with high accuracy
                        atoms.calc = TBLite(method="GFN2-xTB", accuracy=30)
                        
                        # Get total energy in eV
                        total_energy = atoms.get_potential_energy()
                        
                        # Convert to energy per atom
                        energy_per_atom = total_energy / len(atoms)
                        
                        results.append({
                            "System": f"{system_id:03d}",
                            "Energy (eV/atom)": energy_per_atom
                        })
                    except Exception as e:
                        print(f"    Error processing {filename}: {e}", file=sys.stderr)
            
            if results:
                df_energies = pd.DataFrame(results)
                df_energies.to_csv(out_csv, index=False)
                print(f"  [SUCCESS] Wrote {len(results)} energies to {out_csv.name}")

if __name__ == "__main__":
    main()
