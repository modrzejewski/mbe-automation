import re
import os
import os.path
import mbe_automation.validation.hf_pbc

def extract_molecule_energies(filename):
    energies = {}
    with open(filename) as f:
        lines = f.readlines()
    
    jobs = [
        "crystal geometry with ghosts",
        "crystal geometry without ghosts",
        "relaxed geometry"
    ]
    
    # Modified pattern to account for multiple spaces between job name and value
    pattern = re.compile(r"^(.*?)\s+(-?\d+\.\d+)$")
    in_energy_block = False
    
    for line in lines:
        if line.strip() == "Calculations completed":
            in_energy_block = True
            continue
        
        if in_energy_block:
            stripped = line.strip()
            
            # Skip the "Energies (a.u.):" line
            if "Energies" in stripped:
                continue
                
            match = pattern.match(stripped)
            if match:
                job, energy = match.groups()
                job = job.strip()  # Remove any extra whitespace
                
                # Check if this job is one we're interested in
                if job in jobs:
                    energies[job] = float(energy)
            
            # Exit if we encounter an empty line after finding energies
            if not stripped and energies:
                break
    
    return energies


def extract_solid_energy(filename):
    with open(filename) as f:
        lines = f.readlines()

    pattern = re.compile(r"^crystal lattice energy per molecule\s+(-?\d+\.\d+)$")

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            return {"crystal lattice energy per molecule": float(match.group(1))}
    return {}


def lattice_energy(jobdir):
    molecule_file = os.path.join(jobdir, "molecule.log")
    solid_file = os.path.join(jobdir, "solid.log")
    
    molecule_energies = extract_molecule_energies(os.path.join(jobdir, "molecule.log"))
    solid_energy = extract_solid_energy(os.path.join(jobdir, "solid.log"))
    print(molecule_energies)
    print(solid_energy)
    if len(molecule_energies) == 3 and len(solid_energy) == 1:
        LatticeEnergy_au = (solid_energy["crystal lattice energy per molecule"]
                            - molecule_energies["crystal geometry with ghosts"]
                            + molecule_energies["crystal geometry without ghosts"]
                            - molecule_energies["relaxed geometry"]
                            )
        au_to_kcal_per_mol = 627.5094688043 # CODATA 2006
        cal_to_joule = 4.184
        au_to_kJ_per_mol = au_to_kcal_per_mol * cal_to_joule
        LatticeEnergy_kJ_per_mol = LatticeEnergy_au * au_to_kJ_per_mol
        return LatticeEnergy_kJ_per_mol
    else:
        return None

    
def lattice_energies(project_dir):
    status = mbe_automation.validation.hf_pbc.check_HF_PBC(project_dir)

    print(f"{'Rmin [Å]':10}{'basis':20}{'k-points':20}{'Elatt (kJ/mol)':20}")
    for x, y, MoleculeDone, SolidDone in status:
        if MoleculeDone and SolidDone:
            job_dir = os.path.join(project_dir, "PBC", "HF", x, y)
            Elatt = lattice_energy(job_dir)
            
            parts = x.split('-')
            Basis = y
            R = float(parts[0])
            KPoints = f"{parts[1]}×{parts[2]}×{parts[3]}"
            print(f"{R:<10.1f}{Basis:20}{KPoints:20}{Elatt:20.3f}")
            
    

