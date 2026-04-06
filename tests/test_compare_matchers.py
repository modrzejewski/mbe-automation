import numpy as np
from ase.build import molecule
from scipy.spatial.transform import Rotation
import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mbe_automation.structure.molecule import match

def create_system(system_name, R=5.0):
    if " + " in system_name:
        parts = system_name.split(" + ")
        mol_a = None
        for i, part in enumerate(parts):
            m = molecule(part.strip())
            # Translate molecules so they don't overlap using radius R
            if i > 0:
                tx = (i % 2) * R
                ty = (i // 2) * R
                m.translate([tx, ty, 0])
            if mol_a is None:
                mol_a = m
            else:
                mol_a += m
        return mol_a
    else:
        return molecule(system_name)

def _optimal_rmsd(pos_a, pos_b):
    # Center positions
    center_a = np.mean(pos_a, axis=0)
    center_b = np.mean(pos_b, axis=0)

    pos_a_centered = pos_a - center_a
    pos_b_centered = pos_b - center_b

    # Find optimal rotation using scipy
    rot, _ = Rotation.align_vectors(pos_a_centered, pos_b_centered)
    pos_b_aligned = rot.apply(pos_b_centered)

    # Calculate RMSD
    diff = pos_a_centered - pos_b_aligned
    dist_sq = np.sum(diff**2, axis=1)
    return np.sqrt(np.mean(dist_sq))

def compute_exact_rmsd(mol_a, mol_b, align_mirror_images=False):
    """
    Compute exact RMSD between mol_a and mol_b without permutations,
    using scipy to find optimal translation and rotation.
    """
    pos_a = mol_a.positions
    pos_b = mol_b.positions

    rmsd = _optimal_rmsd(pos_a, pos_b)

    if align_mirror_images:
        pos_b_mirror = pos_b * [1, -1, 1]
        rmsd_mirror = _optimal_rmsd(pos_a, pos_b_mirror)
        rmsd = min(rmsd, rmsd_mirror)

    return rmsd

def run_comparison(verbose=True):
    # Define systems using "+" notation
    systems = [
        "H2O", "NH3", "CH4", "C6H6",
        "H2O + H2O", "NH3 + NH3", "CH4 + CH4", "C6H6 + C6H6",
        "H2O + H2O + H2O", "NH3 + NH3 + NH3", "CH4 + CH4 + CH4", "C6H6 + C6H6 + C6H6",
        "H2O + NH3", "CH4 + C6H6"
    ]
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    R_values = [5.0, 10.0]
    
    np.random.seed(42) # For reproducibility
    
    total_tests = 0
    ase_better = 0
    pmg_better = 0

    total_ase_deviation = 0.0
    total_pmg_deviation = 0.0

    def evaluate(mol_a, mol_b, exact_rmsd, align_mirror_images=False):
        nonlocal total_tests, ase_better, pmg_better, total_ase_deviation, total_pmg_deviation
        
        # ALWAYS permute atoms of mol_b before calling match
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
        
        # Calculate deviations
        ase_dev = abs(rmsd_ase - exact_rmsd)
        pmg_dev = abs(rmsd_pmg - exact_rmsd)

        total_ase_deviation += ase_dev
        total_pmg_deviation += pmg_dev

        total_tests += 1

        # Comparison logic based on deviation from exact result
        if ase_dev < pmg_dev - 1e-6:
            ase_better += 1
        elif pmg_dev < ase_dev - 1e-6:
            pmg_better += 1

        return rmsd_ase, rmsd_pmg

    for R in R_values:
        if verbose:
            print(f"\n=======================================================")
            print(f"================== Radius R = {R} Angs ==================")
            print(f"=======================================================")

        # 1. Tests with different noise levels (no mirror images)
        for noise_level in noise_levels:
            if verbose:
                print(f"\n### Test: Noise level {noise_level} Angs (R={R})")
                print(f"| {'System':<20} | {'Exact RMSD':<15} | {'RMSD (ASE)':<15} | {'RMSD (Pymatgen)':<15} |")
                print(f"|{'-'*22}|{'-'*17}|{'-'*17}|{'-'*17}|")

            for system_name in systems:
                mol_a = create_system(system_name, R=R)
                mol_b = mol_a.copy()

                if noise_level > 0.0:
                    noise = np.random.uniform(-noise_level, noise_level, mol_b.positions.shape)
                    mol_b.positions += noise

                # Compute exact RMSD before permutation
                exact_rmsd = compute_exact_rmsd(mol_a, mol_b, align_mirror_images=False)

                # evaluate function takes care of permutation and matching
                rmsd_ase, rmsd_pmg = evaluate(mol_a, mol_b, exact_rmsd, align_mirror_images=False)

                if verbose:
                    print(f"| {system_name:<20} | {exact_rmsd:<15.6f} | {rmsd_ase:<15.6f} | {rmsd_pmg:<15.6f} |")

        # 2. Tests with mirror images
        if verbose:
            print(f"\n### Test: Mirror Images (align_mirror_images=True, R={R})")
            print(f"| {'System':<20} | {'Exact RMSD':<15} | {'RMSD (ASE)':<15} | {'RMSD (Pymatgen)':<15} |")
            print(f"|{'-'*22}|{'-'*17}|{'-'*17}|{'-'*17}|")

        for system_name in systems:
            mol_a = create_system(system_name, R=R)
            mol_b = mol_a.copy()

            # Create mirror image by inverting y-axis
            mol_b.positions *= [1, -1, 1]

            # Add a small uniform noise to make it realistic
            noise = np.random.uniform(-0.1, 0.1, mol_b.positions.shape)
            mol_b.positions += noise

            # Compute exact RMSD before permutation
            exact_rmsd = compute_exact_rmsd(mol_a, mol_b, align_mirror_images=True)

            # evaluate function takes care of permutation and matching
            rmsd_ase, rmsd_pmg = evaluate(mol_a, mol_b, exact_rmsd, align_mirror_images=True)

            if verbose:
                print(f"| {system_name:<20} | {exact_rmsd:<15.6f} | {rmsd_ase:<15.6f} | {rmsd_pmg:<15.6f} |")

    avg_ase_dev = total_ase_deviation / total_tests
    avg_pmg_dev = total_pmg_deviation / total_tests

    # Summary
    if verbose:
        print("\n" + "="*55)
        print("Summary:")
        print("="*55)
        print(f"In {ase_better} tests out of {total_tests} algorithm ase was closer to the exact result than algorithm pymatgen.")
        print(f"In {pmg_better} tests out of {total_tests} algorithm pymatgen was closer to the exact result than algorithm ase.")

        print(f"\nAverage deviation against exact RMSD:")
        print(f"  - ASE:      {avg_ase_dev:.6f}")
        print(f"  - Pymatgen: {avg_pmg_dev:.6f}")
    
    return avg_ase_dev, avg_pmg_dev

def test_compare_matchers_accuracy():
    """
    Test that the average error of pymatgen and ASE matchers is within limits.
    Success criteria:
    - Pymatgen average error < 0.02 Angstrom
    - ASE average error < 0.2 Angstrom
    """
    avg_ase_dev, avg_pmg_dev = run_comparison(verbose=False)
    
    assert avg_pmg_dev < 0.02, f"Pymatgen average deviation {avg_pmg_dev:.6f} exceeds 0.02!"
    assert avg_ase_dev < 0.2, f"ASE average deviation {avg_ase_dev:.6f} exceeds 0.2!"

if __name__ == "__main__":
    run_comparison(verbose=True)
