import numpy as np
import ase
import pyscf
import pyscf.hessian.thermo
from typing import Tuple, Optional

import mbe_automation.common

def from_file(path):
    return mbe_automation.common.io.read(path)


def to_pyscf(atoms: ase.Atoms, charge: int = 0, spin: int = 0) -> pyscf.gto.Mole:
    """
    Convert an ASE Atoms object to a PySCF Mole object.
    """
    mol = pyscf.gto.Mole()
    mol.atom = [(atom.symbol, atom.position) for atom in atoms]
    mol.unit = 'Angstrom'
    mol.charge = charge
    mol.spin = spin
    mol.build()
    return mol


def analyze_geometry(atoms: ase.Atoms) -> Tuple[str, int]:
    """
    Analyze molecular geometry to determine if it's linear and get symmetry number.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        ASE Atoms object
        
    Returns:
    --------
    tuple : (rotor_type, symmetry_number)
        - rotor_type: 'monatomic', 'linear', 'nonlinear'
        - symmetry_number: rotational symmetry number
    """
    #
    # This code is copied from pyscf.hessian.thermo.harmonic_analysis
    #
    mol = to_pyscf(atoms)
    mass = mol.atom_mass_list(isotope_avg=True)
    coords = mol.atom_coords()
    rot_const = pyscf.hessian.thermo.rotation_const(mass, coords, unit='GHz')
    if np.all(rot_const > 1e8):
        rotor_type = 'monatomic'
    elif rot_const[0] > 1e8 and (rot_const[1] - rot_const[2] < 1e-3):
        rotor_type = 'linear'
    else:
        rotor_type = 'nonlinear'
        
    sym_number = pyscf.hessian.thermo.rotational_symmetry_number(mol)
    
    return rotor_type, sym_number


def n_rotational_degrees_of_freedom(molecule):
    rotor_type, _ = analyze_geometry(molecule)
    
    if rotor_type == "monatomic":
        n_rot_dof = 0
    elif rotor_type == "linear":
        n_rot_dof = 2
    elif rotor_type == "nonlinear":
        n_rot_dof = 3

    return n_rot_dof
