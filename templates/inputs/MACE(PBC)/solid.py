import ase
from ase.atoms import Atoms
from ase.io import read
import numpy as np
import os
import os.path
import time
import mace.calculators
import sys
import mbe_automation.properties
import torch
torch.set_default_dtype(torch.float64)

Calc = mace.calculators.mace_off(model="medium")
SupercellRadius = 30.0 # Minimum point-periodic image distance in the supercell (Angstrom)
SupercellDisplacement = 0.01 # Displacement in Cartesian coordinated to compute numerical derivatives
XYZ_Solid = "{XYZ_Solid}"
XYZ_Molecule = "{XYZ_Molecule}"
CSV_Dir = "{CSV_Dir}"
Plots_Dir = "{Plot_Dir}"
Temperatures=np.arange(0, 401, 1)
ConstantVolume = True

print("Calculations with MACE")
print(f"Coordinates (crystal structure): {{XYZ_Solid}}")
print(f"Coordinates (relaxed molecule): {{XYZ_Molecule}}")

WorkDir = os.path.abspath(os.path.dirname(__file__))
XYZ_Solid = os.path.join(WorkDir, XYZ_Solid)
XYZ_Molecule = os.path.join(WorkDir, XYZ_Molecule)
CSV_Dir = os.path.join(WorkDir, CSV_Dir)
Plots_Dir = os.path.join(WorkDir, Plots_Dir)

UnitCell = read(XYZ_Solid)
Molecule = read(XYZ_Molecule)

mbe_automation.properties.thermodynamic(
    UnitCell,
    Molecule,
    Calc,
    Calc,
    Calc,
    Temperatures,
    ConstantVolume,
    CSV_Dir,
    Plots_Dir,
    SupercellRadius,
    SupercellDisplacement)

