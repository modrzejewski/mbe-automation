import ase
import pyscf
from ase.io import read
import numpy as np
import ase.geometry
import ase.io
import ase.build
from ase import Atoms
import os.path
import time
from pyscf import scf
import pyscf.lib
from pyscf import scf, gto
from pyscf import scf as mol_scf
import sys

BasisSet = "{BASIS_SET}"
NAtoms = {NATOMS} # Number of atoms of the reference molecule
XYZ_MoleculeWithGhosts = "molecule_supercell.xyz"
XYZ_RelaxedMolecule = "molecule_relaxed.xyz"
MaxMemory = 170 * 10**3 # memory in megabytes
Verbosity = 4
AlwaysCheckLinDeps = True
LinDepThresh = 1.0E-6
ScratchDir = "scratch_molecule"

WorkDir = os.path.abspath(os.path.dirname(__file__))
XYZ_MoleculeWithGhosts = os.path.join(WorkDir, XYZ_MoleculeWithGhosts)
XYZ_RelaxedMolecule = os.path.join(WorkDir, XYZ_RelaxedMolecule)
ScratchDir = os.path.join(WorkDir, ScratchDir)

print("Finite system Hartree-Fock calculation")
print("(1) Molecule (crystal geometry, with ghosts)")
print("(2) Molecule (crystal geometry, without ghosts)")
print("(3) Molecule (relaxed geometry)")
print(f"Coordinates 1, 2: {{XYZ_MoleculeWithGhosts}}")
print(f"Coordinated 3: {{XYZ_RelaxedMolecule}}")
print(f"Basis set: {{BasisSet}}")
print(f"Atoms: {{NAtoms}}")

os.makedirs(ScratchDir, exist_ok=True)
os.environ["PYSCF_TMPDIR"] = os.path.realpath(ScratchDir)
os.environ["TMPDIR"] = ScratchDir

MoleculeWithGhosts  = read(XYZ_MoleculeWithGhosts)
RelaxedMolecule = read(XYZ_RelaxedMolecule)
NCenters = len(MoleculeWithGhosts)
if NCenters < NAtoms:
    print("Invalid xyz file for molecule with ghosts")
    sys.exit(1)
NGhosts = NCenters - NAtoms
print(f"Ghosts: {{NGhosts}}")
Molecule = MoleculeWithGhosts[:NAtoms]
Ghosts = MoleculeWithGhosts[NAtoms:]

s1 = "\n".join(
    f"{{symbol}} {{pos[0]:.8f}} {{pos[1]:.8f}} {{pos[2]:.8f}}"
    for symbol, pos in zip(Molecule.get_chemical_symbols(), Molecule.get_positions())
)
if NGhosts > 0:
    s2 = "\n".join(
        f"ghost-{{symbol}} {{pos[0]:.8f}} {{pos[1]:.8f}} {{pos[2]:.8f}}"
        for symbol, pos in zip(Ghosts.get_chemical_symbols(), Ghosts.get_positions())
    )
else:
    s2 = ""

s3 = "\n".join(
    f"{{symbol}} {{pos[0]:.8f}} {{pos[1]:.8f}} {{pos[2]:.8f}}"
    for symbol, pos in zip(RelaxedMolecule.get_chemical_symbols(), RelaxedMolecule.get_positions())
)
    
M1 = gto.M(
    unit="angstrom",
    atom = "\n".join([s1, s2]),
    basis = BasisSet,
    verbose=Verbosity,
    max_memory=MaxMemory
)
M1.build()

M2 = gto.M(
    unit="angstrom",
    atom = s1,
    basis = BasisSet,
    verbose=Verbosity,
    max_memory=MaxMemory
)
M2.build()

M3 = gto.M(
    unit="angstrom",
    atom = s3,
    basis = BasisSet,
    verbose=Verbosity,
    max_memory=MaxMemory
)
M3.build()

Jobs = ["crystal geometry with ghosts",
        "crystal geometry without ghosts",
        "relaxed geometry"]    
M = [M2, M2, M3]
Energies = {{}}
for k, Mk in enumerate(M):
    print(f"Molecule: {{Jobs[k]}}")
    print(f'Atomic orbitals: {{Mk.nao}}')
    print(f'Occupied orbitals: {{Mk.nelectron//2}}')
    mean_field = scf.RHF(Mk).density_fit()
    #
    # Elimination of linear dependencies
    #
    # mean_field = mol_scf.addons.remove_linear_dep_(mean_field,
    #                                                threshold=LinDepThresh,
    #                                                cholesky_threshold=1.0E-8,
    #                                                force_pivoted_cholesky=AlwaysCheckLinDeps)
    if AlwaysCheckLinDeps:
        mean_field._eigh = _eigh_with_canonical_orth(LinDepThresh)
    mean_field.chkfile = os.path.join(ScratchDir, f"molecule_{{k}}.chk")
    Energies[Jobs[k]] = mean_field.run()
    

print(f"Calculations completed")
for J in Jobs:
    print(f"Energy of single molecule ({{J}}; a.u.): {{Energies[J]:.8f}}")


