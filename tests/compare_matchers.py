import numpy as np
from ase.build import molecule, minimize_rotation_and_translation
import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mbe_automation.structure.molecule import match

def run_comparison():
    # Define systems
    systems = [
        "H2O", "NH3", "CH4", "C6H6",
        "H2O dimer", "NH3 dimer", "CH4 dimer", "C6H6 dimer",
        "H2O + NH3", "CH4 + C6H6"
    ]
    
    # Print header
    print(f"| {'System':<20} | {'RMSD (ASE)':<15} | {'RMSD (Pymatgen)':<15} |")
    print(f"|{'-'*22}|{'-'*17}|{'-'*17}|")
    
    np.random.seed(42) # For reproducibility
    
    for system_name in systems:
        # Create system A
        if " + " in system_name:
            parts = system_name.split(" + ")
            m1 = molecule(parts[0])
            m2 = molecule(parts[1])
            m2.translate([5, 0, 0])
            mol_a = m1 + m2
        elif " dimer" in system_name:
            base = system_name.replace(" dimer", "")
            m1 = molecule(base)
            m2 = m1.copy()
            m2.translate([5, 0, 0])
            mol_a = m1 + m2
        else:
            mol_a = molecule(system_name)
            
        # Create system B as a permuted and noisy copy of A
        mol_b = mol_a.copy()
        
        # Uniform noise [-0.1, 0.1]
        noise = np.random.uniform(-0.1, 0.1, mol_b.positions.shape)
        mol_b.positions += noise
        
        # Permute atoms of mol_b
        indices = np.random.permutation(len(mol_b))
        mol_b = mol_b[indices]
        
        rmsd_ase = match(
            mol_a.positions, mol_a.numbers,
            mol_b.positions, mol_b.numbers,
            algorithm="ase"
        )
        
        rmsd_pmg = match(
            mol_a.positions, mol_a.numbers,
            mol_b.positions, mol_b.numbers,
            algorithm="pymatgen"
        )
        
        print(f"| {system_name:<20} | {rmsd_ase:<15.6f} | {rmsd_pmg:<15.6f} |")

if __name__ == "__main__":
    run_comparison()
