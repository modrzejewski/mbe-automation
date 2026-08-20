import ase
from ase.atoms import Atoms
from ase.io import read
import numpy as np
import os
import os.path
import time
import sys
import mbe_automation.kpoints
import mbe_automation.properties
import mbe_automation.calculators.dftb

SupercellRadius = 30.0 # Minimum point-periodic image distance in the supercell (Angstrom)
SupercellDisplacement = 0.01 # Displacement in Cartesian coordinated to compute numerical derivatives
XYZ_Solid = "{XYZ_Solid}"
XYZ_Molecule = "{XYZ_Molecule}"
Params_Dir = "{Params_Dir}"
CSV_Dir = "{CSV_Dir}"
Plots_Dir = "{Plot_Dir}"
Temperatures=np.arange(0, 401, 1)
ConstantVolume = True

print("Calculations with DFTB")
print(f"Coordinates (crystal structure): {{XYZ_Solid}}")
print(f"Coordinates (relaxed molecule): {{XYZ_Molecule}}")

WorkDir = os.path.abspath(os.path.dirname(__file__))
XYZ_Solid = os.path.join(WorkDir, XYZ_Solid)
XYZ_Molecule = os.path.join(WorkDir, XYZ_Molecule)
CSV_Dir = os.path.join(WorkDir, CSV_Dir)
Plots_Dir = os.path.join(WorkDir, Plots_Dir)

UnitCell = read(XYZ_Solid)
Molecule = read(XYZ_Molecule)

Nk = mbe_automation.kpoints.RminSupercell(UnitCell, SupercellRadius)
ScaledKPoints = mbe_automation.kpoints.ScaledKPoints(Nk, GammaCentered=True)
Calc_UnitCell = mbe_automation.calculators.dftb.DFTB3_D4(elements=UnitCell.get_chemical_symbols(),
                                                         params_dir=Params_Dir, kpts=ScaledKPoints)
Calc_SuperCell = mbe_automation.calculators.dftb.DFTB3_D4(elements=UnitCell.get_chemical_symbols(),
                                                          params_dir=Params_Dir, kpts=[1,1,1])
Calc_Molecule = mbe_automation.calculators.dftb.DFTB3_D4(elements=UnitCell.get_chemical_symbols(),
                                                          params_dir=Params_Dir)

mbe_automation.properties.thermodynamic(
    UnitCell,
    Molecule,
    Calc_Molecule,
    Calc_UnitCell,
    Calc_SuperCell,
    Temperatures,
    ConstantVolume,
    CSV_Dir,
    Plots_Dir,
    SupercellRadius,
    SupercellDisplacement)

