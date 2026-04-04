import numpy as np
from ase.build import molecule
import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mbe_automation.structure.molecule import match

def create_system(system_name):
    if " + " in system_name:
        parts = system_name.split(" + ")
        mol_a = None
        for i, part in enumerate(parts):
            m = molecule(part.strip())
            # Translate molecules so they don't overlap
            # A simple grid placement: translate based on index
            if i > 0:
                # e.g., i=1 -> [5, 0, 0], i=2 -> [0, 5, 0], i=3 -> [5, 5, 0]
                tx = (i % 2) * 5
                ty = (i // 2) * 5
                m.translate([tx, ty, 0])
            if mol_a is None:
                mol_a = m
            else:
                mol_a += m
        return mol_a
    else:
        return molecule(system_name)

def run_comparison():
    # Define systems using "+" notation
    systems = [
        "H2O", "NH3", "CH4", "C6H6",
        "H2O + H2O", "NH3 + NH3", "CH4 + CH4", "C6H6 + C6H6",
        "H2O + H2O + H2O", "NH3 + NH3 + NH3", "CH4 + CH4 + CH4", "C6H6 + C6H6 + C6H6",
        "H2O + NH3", "CH4 + C6H6"
    ]
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    np.random.seed(42) # For reproducibility
    
    total_tests = 0
    ase_better = 0
    pmg_better = 0

    def evaluate(mol_a, mol_b, align_mirror_images=False):
        nonlocal total_tests, ase_better, pmg_better
        
        # Permute atoms of mol_b
        indices = np.random.permutation(len(mol_b))
        mol_b = mol_b[indices]
        
        rmsd_ase = match(
            mol_a.positions, mol_a.numbers,
            mol_b.positions, mol_b.numbers,
            align_mirror_images=align_mirror_images,
            algorithm="ase"
        )
        
        rmsd_pmg = match(
            mol_a.positions, mol_a.numbers,
            mol_b.positions, mol_b.numbers,
            align_mirror_images=align_mirror_images,
            algorithm="pymatgen"
        )
        
        total_tests += 1
        if rmsd_ase < rmsd_pmg - 1e-6:
            ase_better += 1
        elif rmsd_pmg < rmsd_ase - 1e-6:
            pmg_better += 1

        return rmsd_ase, rmsd_pmg

    # 1. Tests with different noise levels (no mirror images)
    for noise_level in noise_levels:
        print(f"\n### Test: Noise level {noise_level} Angs")
        print(f"| {'System':<20} | {'RMSD (ASE)':<15} | {'RMSD (Pymatgen)':<15} |")
        print(f"|{'-'*22}|{'-'*17}|{'-'*17}|")

        for system_name in systems:
            mol_a = create_system(system_name)
            mol_b = mol_a.copy()

            if noise_level > 0.0:
                noise = np.random.uniform(-noise_level, noise_level, mol_b.positions.shape)
                mol_b.positions += noise

            rmsd_ase, rmsd_pmg = evaluate(mol_a, mol_b, align_mirror_images=False)
            print(f"| {system_name:<20} | {rmsd_ase:<15.6f} | {rmsd_pmg:<15.6f} |")

    # 2. Tests with mirror images
    print(f"\n### Test: Mirror Images (align_mirror_images=True)")
    print(f"| {'System':<20} | {'RMSD (ASE)':<15} | {'RMSD (Pymatgen)':<15} |")
    print(f"|{'-'*22}|{'-'*17}|{'-'*17}|")

    for system_name in systems:
        mol_a = create_system(system_name)
        mol_b = mol_a.copy()

        # Create mirror image by inverting y-axis
        mol_b.positions *= [1, -1, 1]

        # Add a small uniform noise to make it realistic
        noise = np.random.uniform(-0.1, 0.1, mol_b.positions.shape)
        mol_b.positions += noise

        rmsd_ase, rmsd_pmg = evaluate(mol_a, mol_b, align_mirror_images=True)
        print(f"| {system_name:<20} | {rmsd_ase:<15.6f} | {rmsd_pmg:<15.6f} |")

    print("\nSummary:")
    print(f"In {ase_better} tests out of {total_tests} algorithm ase is better than algorithm pymatgen.")
    print(f"In {pmg_better} tests out of {total_tests} algorithm pymatgen is better than algorithm ase.")

if __name__ == "__main__":
    run_comparison()
