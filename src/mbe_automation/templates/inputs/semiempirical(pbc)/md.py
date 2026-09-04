import numpy as np
import os.path

import mbe_automation
import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation.configs.execution import Resources
from mbe_automation import Structure

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))
dataset = os.path.join(work_dir, "properties.hdf5")

env_config = Resources.auto_detect(model_name="dftb3-d4")
env_config.set()

molecule = Structure.from_file(xyz_molecule)
crystal = Structure.from_file(xyz_solid)

md_config = mbe_automation.configs.md.Enthalpy(
    molecule = molecule,
    crystal = crystal,
    calculator = mbe_automation.calculators.DFTB3_D4(),
    temperature_K = 298.15,
    pressure_GPa = 1.0E-4,
    work_dir = work_dir,
    dataset = dataset,
    
    md_molecule = mbe_automation.configs.md.ClassicalMD(
        ensemble = "NVT",
        time_total_fs = 10000.0,
        time_step_fs = 1.0,
        sampling_interval_fs = 50.0,
        time_equilibration_fs = 5000.0,
    ),
    
    md_crystal = mbe_automation.configs.md.ClassicalMD(
        ensemble = "NPT",
        time_total_fs = 10000.0,
        time_step_fs = 1.0,
        sampling_interval_fs = 50.0,
        time_equilibration_fs = 5000.0,
        supercell_radius = 15.0,
    )
)

mbe_automation.run(md_config)
