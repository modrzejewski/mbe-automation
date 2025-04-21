import ase
from ase.atoms import Atoms
from ase.io import read
from ase.units import kJ, mol
import numpy as np
import os
import os.path
import time
import mace.calculators
import sys
import mbe_automation.kpoints
import mbe_automation.structure.optimize
import mbe_automation.vibrations.harmonic
import torch
torch.set_default_dtype(torch.float64)

Calculator = mace.calculators.mace_off(model="medium")
SupercellRadius = 20.0 # Minimum point-periodic image distance in the supercell (Angstrom)
Displacement = 0.03
XYZ_Solid = "{XYZ_Solid}"
XYZ_Molecule = "{XYZ_Molecule}"
PhononPlot = "phonon_band_structure.png"
DOSPlot = "phonon_DOS.png"

print("Calculations with MACE")
print(f"Coordinates (crystal structure): {{XYZ_Solid}}")
print(f"Coordinates (relaxed molecule): {{XYZ_Molecule}}")

WorkDir = os.path.abspath(os.path.dirname(__file__))
XYZ_Solid = os.path.join(WorkDir, XYZ_Solid)
XYZ_Molecule = os.path.join(WorkDir, XYZ_Molecule)
PhononPlot = os.path.join(WorkDir, PhononPlot)

UnitCell = read(XYZ_Solid)
UnitCell.calc = Calculator
RelaxedMolecule = read(XYZ_Molecule)
RelaxedMolecule.calc = Calculator

NAtoms = len(RelaxedMolecule)
if len(UnitCell) % NAtoms != 0:
    print("Invalid number of atoms in the unit cell: cannot determine the number of molecules")
    sys.exit(1)
NMoleculesPerCell = len(UnitCell) // NAtoms
print(f"Molecules per cell: {{NMoleculesPerCell}}")

RelaxedMoleculeEnergy = RelaxedMolecule.get_potential_energy()
CellEnergy = UnitCell.get_potential_energy()
CellEnergyPerMolecule = CellEnergy / NMoleculesPerCell

LatticeEnergy_kJmol = (CellEnergyPerMolecule - RelaxedMoleculeEnergy) / (kJ/mol)

print(f"Calculations completed")
print(f"Energies (kJ/mol):")
J = "crystal lattice energy per molecule"
print(f"{{J}} {{LatticeEnergy_kJmol:.6f}}")

OptUnitCell = mbe_automation.structure.optimize.atoms_and_cell(UnitCell, Calculator)
properties = mbe_automation.vibrations.harmonic.phonopy(
    OptUnitCell,
    Calculator,
    SupercellRadius=SupercellRadius,
    Mesh=SupercellDims,
    SupercellDisplacement=0.03,
    MinTemperature=0,
    MaxTemperature=1000,
    TemperatureStep=10,
    Sigma=0.1,
    DOSPlot=DOSPlot):

print(properties)

