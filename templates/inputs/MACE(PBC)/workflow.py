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
import mbe_automation.ml.training_data
import torch
torch.set_default_dtype(torch.float64)
#
# ============================== User-defined parameters =================================
#
# Unviersal parametrers for all enabled
# types of calculations
#
Calc = mace.calculators.mace_off(model="small")
SupercellRadius = 10.0 # Minimum point-periodic image distance in the supercell (Angstrom)
XYZ_Solid = "{XYZ_Solid}"
XYZ_Molecule = "{XYZ_Molecule}"
CSV_Dir = "{CSV_Dir}"
Plots_Dir = "{Plot_Dir}"
Training_Dir = "{Training_Dir}"
#
# Molecular dynamics
#
Training_Crystal_MD = True
Training_Molecule_MD = True
                                   #
                                   # Thermostat temperature. Equilibrium is detected
                                   # when the standard deviation sigma(T) within the
                                   # averaging window falls below predefined
                                   # threshold.
                                   #
temperature_K = 298.15
time_total_fs = 50000
time_step_fs = 0.5
sampling_interval_fs = 50
time_equilibration_fs = 5000
                                   #
                                   # Time window for the plot of running average T and E.
                                   # Should be long enough to average over characteristic
                                   # motions present in the system.
                                   #
averaging_window_fs = 5000
#
# Parameters for thermodynamic properties
# in the harmonic approximation
#
                                   #
                                   # Enable computational path for harmonic properties.
                                   # Second derivatives evaluated using finite
                                   # differences with phonopy.
                                   #
HarmonicProperties = True
                                   #
                                   # Range of temperatures where thermodynamic properties
                                   # are calculated, e.g., T's where Cv is evaluated from
                                   # the phonon data
                                   #
Temperatures=np.arange(0, 401, 1) 
ConstantVolume = True
SupercellDisplacement = 0.01 # Displacement in Cartesian coordinated to compute numerical derivatives
#
# ========================= End of user-defined parameters ===============================
#

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

if Training_Molecule_MD:
    mbe_automation.ml.training_data.molecule_md(
        Molecule,
        Calc,
        Training_Dir,
        temperature_K,
        time_total_fs,
        time_step_fs,
        sampling_interval_fs,
        averaging_window_fs,
        time_equilibration_fs
    )
    
if Training_Crystal_MD:
    mbe_automation.ml.training_data.supercell_md(
        UnitCell,
        Calc,
        SupercellRadius,
        Training_Dir,
        temperature_K,
        time_total_fs,
        time_step_fs,
        sampling_interval_fs,
        averaging_window_fs,
        time_equilibration_fs
    )

if HarmonicProperties:
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



