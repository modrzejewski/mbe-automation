import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation.storage import from_xyz_file

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("{mlip_parameters}"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.from_template(
    model_name = "MACE",
    crystal = from_xyz_file(os.path.join(work_dir, xyz_solid)),
    molecule = from_xyz_file(os.path.join(work_dir, xyz_molecule)),
    temperatures_K = np.array([5.0, 200.0, 300.0]),
    calculator = mace_calc,
    supercell_radius = 25.0,
    work_dir = os.path.join(work_dir, "properties"),
    dataset = os.path.join(work_dir, "properties.hdf5")
)

mbe_automation.workflows.quasi_harmonic.run(properties_config)



