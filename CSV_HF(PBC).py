import re
import os
import os.path

def extract_molecule_energies(filename):
    energies = {}
    with open(filename) as f:
        lines = f.readlines()

    Jobs = [
        "crystal geometry with ghosts",
        "crystal geometry without ghosts",
        "relaxed geometry"
    ]
    
    # Create a dynamic pattern for the JOBS list
    job_pattern = "|".join(re.escape(job) for job in Jobs)
    pattern = re.compile(rf"^({job_pattern})\s+(-?\d+\.\d+)$")

    in_energy_block = False
    for line in lines:
        if "Calculations completed" in line:
            in_energy_block = True
            continue
        if in_energy_block:
            stripped = line.strip()
            match = pattern.match(stripped)
            if match:
                job, energy = match.groups()
                energies[job] = float(energy)
            if not stripped:
                break  # end of block
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
    
    if len(molecule_energies) == 3 and len(solid_energy) == 0:
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





    
