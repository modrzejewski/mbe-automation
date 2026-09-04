import numpy as np
import os.path

import mbe_automation
from mbe_automation import Structure
from mbe_automation.configs.execution import Resources
from mbe_automation.configs.structure import Minimum

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))

env_config = Resources.auto_detect(model_name="dftb3-d4")
env_config.set()

crystal = Structure.from_file(xyz_solid)
molecule = Structure.from_file(xyz_molecule)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name = "dftb3-d4",
    crystal = crystal,
    molecule = molecule,
    thermal_expansion = False,
    calculator = mbe_automation.calculators.DFTB3_D4(),
    supercell_radius = 10.0,
    work_dir = work_dir,
    dataset = os.path.join(work_dir, "properties.hdf5"),
)

mbe_automation.run(properties_config)
