import numpy as np
import os.path
import mbe_automation
from mbe_automation.storage import from_xyz_file
from mbe_automation.configs.execution import ParallelCPU

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))

env_config = ParallelCPU.recommended(model_name="gfn2-xtb")
env_config.set()

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name = "gfn2-xtb",
    crystal = from_xyz_file(os.path.join(work_dir, xyz_solid)),
    molecule = from_xyz_file(os.path.join(work_dir, xyz_molecule)),
    thermal_expansion = False,
    calculator = mbe_automation.calculators.GFN2_xTB(verbose=True),
    supercell_radius = 10.0,
    work_dir = os.path.join(work_dir, "properties"),
    dataset = os.path.join(work_dir, "properties.hdf5")
)

mbe_automation.workflows.quasi_harmonic.run(properties_config)



