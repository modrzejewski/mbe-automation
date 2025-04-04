import ase
import pyscf
from ase.io import read
import numpy as np
import ase.geometry
import ase.io
import ase.build
from ase import Atoms
import scipy
from scipy import sparse
import os
import os.path
import time
from pyscf import scf
import pyscf.lib
from pyscf.pbc import gto
from pyscf import scf
from pyscf.pbc import gto, scf
from pyscf import scf as mol_scf
import sys

BasisSet = "{BASIS_SET}"
#
# k-point grid adjusted to satisfy 
# Nk * LatticeVectorLength > {KPOINT_RADIUS:.0f} A
#
KPointGrid = [{NX},{NY},{NZ}] 
XYZFile = "solid.xyz"
NAtoms = {NATOMS} # Number of atoms of the reference molecule
MaxMemory = 170 * 10**3 # memory in megabytes
Verbosity = 4
AlwaysCheckLinDeps = True
LinDepThresh = 1.0E-6
ScratchDir = "scratch_solid"

XYZFile = os.path.join(os.path.abspath(__file__), XYZFile)
ScratchDir = os.path.join(os.path.abspath(__file__), ScratchDir)

print("PBC Hartree-Fock calculation with PySCF")
print(f"Coordinates: {{XYZFile}}")
print(f"Basis set: {{BasisSet}}")
print(f"K-points: ({{KPointGrid[0]}},{{KPointGrid[1]}},{{KPointGrid[2]}}")

os.makedirs(ScratchDir, exist_ok=True)
os.environ["PYSCF_TMPDIR"] = ScratchDir
os.environ["TMPDIR"] = ScratchDir

system  = read(XYZFile)
cell_ase = system.get_cell()
if cell_ase.handedness == -1:
    print("The input lattice vectors (row by row) should form a right-handed coordinate system.")
    print("Otherwise some integrals may be computed incorrectly in PySCF.")
    sys.exit(1)
if len(system) % NAtoms != 0:
    print("Invalid number of atoms in the unit cell: cannot determine the number of molecules")
    sys.exit(1)
NMoleculesPerCell = len(system) / NAtoms
print(f"Molecules in the unit cell: {NMoleculesPerCell}")
    
geometry_string = "\n".join(
    f"{{symbol}} {{pos[0]:.8f}} {{pos[1]:.8f}} {{pos[2]:.8f}}"
    for symbol, pos in zip(system.get_chemical_symbols(), system.get_positions())
)
cell = gto.Cell()
cell.build(
    unit = "angstrom",
    atom = geometry_string,
    basis = BasisSet,
    a = system.get_cell(),
    max_memory=MaxMemory,
    verbose = Verbosity
)

print(f"AOs per cell: {{cell.nao:5d}}")
print(f"Occupied orbitals per cell: {{cell.nelectron//2:5d}}")

kpts = cell.make_kpts(KPointGrid)
#
# Enable range-separation algorithm for Coulomb integral evaluation.
# According to the pySCF manual, this variant has the smallest
# memory requirements for large systems.
#
mean_field = scf.KRHF(cell, kpts=kpts, exxdiv="ewald").jk_method("RS")
#
# Elimination of linear dependencies
#
# mean_field = mol_scf.addons.remove_linear_dep_(mean_field,
#                                                threshold=LinDepThresh,
#                                                cholesky_threshold=1.0E-8,
#                                                force_pivoted_cholesky=AlwaysCheckLinDeps)
if AlwaysCheckLinDeps:
    mean_field._eigh = _eigh_with_canonical_orth(LinDepThresh)
mean_field.chkfile = os.path.join(ScratchDir, "solid.chk")
CellEnergy = mean_field.run()
CellEnergyPerMolecule = CellEnergy / NMoleculesPerCell
print(f"Calculations completed")
print(f"Cell energy per molecule (a.u.): {{CellEnergyPerMolecule:.8f}}")
