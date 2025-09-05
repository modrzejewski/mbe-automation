import numpy as np
import os.path
import mace.calculators
from mbe_automation.configs.md import Sublimation, ClassicalMD
import mbe_automation.workflows.md
import mbe_automation.io
import torch

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("{mlip_parameters}"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

md_config = Sublimation(
    molecule = mbe_automation.io.read(os.path.join(work_dir, xyz_molecule)),
    crystal = mbe_automation.io.read(os.path.join(work_dir, xyz_solid)),
    calculator = mace_calc,
    temperature_K = 298.15,
    pressure_GPa = 1.0E-4,
    work_dir = os.path.join(work_dir, "properties"),
    dataset = os.path.join(work_dir, "properties.hdf5"),
    
    md_molecule = ClassicalMD(
        ensemble = "NVT",
        time_total_fs = 50000.0,
        time_step_fs = 1.0,
        sampling_interval_fs = 50.0,
        time_equilibration_fs = 5000.0
    ),
    
    md_crystal = ClassicalMD(
        ensemble = "NPT",
        time_total_fs = 50000.0,
        time_step_fs = 1.0,
        sampling_interval_fs = 50.0,
        time_equilibration_fs = 5000.0,
        supercell_radius = 15.0,
        supercell_diagonal = True
    )
)

mbe_automation.workflows.md.run(md_config)



