import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation
from mbe_automation.configs.md import ClassicalMD
from mbe_automation.storage import from_xyz_file

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("{mlip_parameters}"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

config = mbe_automation.configs.training.TrainingSet(
    crystal = from_xyz_file(os.path.join(work_dir, xyz_solid)),
    calculator = mace_calc,
    temperature_K = 298.15,
    pressure_GPa = 1.0E-4,
    filter = "closest_to_central_molecule",
    n_molecules=5,
    md_crystal = ClassicalMD(
        ensemble = "NPT",
        time_total_fs=10000.0, 
        time_equilibration_fs=1000.0,
        sampling_interval_fs=1000.0,
        supercell_radius = 10.0,
    ),
    work_dir = os.path.join(work_dir, "properties"),
    dataset = os.path.join(work_dir, "properties.hdf5")
)

mbe_automation.workflows.training.run(config)



