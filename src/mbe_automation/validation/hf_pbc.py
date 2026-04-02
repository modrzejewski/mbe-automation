
import os
import os.path
import glob
from pathlib import Path

def check_HF_PBC(project_dir):

    base_dir = os.path.join(project_dir, "PBC", "HF")
    base_dir = Path(base_dir)
    results = []
    
    # Find all X directories
    x_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for x_dir in x_dirs:
        # Find all Y directories within each X directory
        y_dirs = [d for d in x_dir.iterdir() if d.is_dir()]
        
        for y_dir in y_dirs:
            # Check if the required files exist
            molecule_log_exists = (y_dir / "molecule.log").exists()
            solid_log_exists = (y_dir / "solid.log").exists()
            
            # Record the results
            results.append((
                x_dir.name,
                y_dir.name,
                molecule_log_exists,
                solid_log_exists
            ))
            
    return results






            

