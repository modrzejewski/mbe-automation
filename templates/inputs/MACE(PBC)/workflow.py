import numpy as np
import os.path
import mace.calculators
from mbe_automation.configs.training import TrainingConfig
from mbe_automation.configs.properties import PropertiesConfig
import mbe_automation.workflows
import mbe_automation.io
import torch
torch.set_default_dtype(torch.float64)

Training = True
HarmonicProperties = True

work_dir = os.path.abspath(os.path.dirname(__file__))

training_config = TrainingConfig(
    calculator = mace.calculators.mace_off(model="small"),
    unit_cell = mbe_automation.io.read(os.path.join(work_dir, "{XYZ_Solid}")),
    molecule = mbe_automation.io.read(os.path.join(work_dir, "{XYZ_Molecule}")),
    supercell_radius = 15.0,
    training_dir = "{Training_Dir}",
    hdf5_dataset = os.path.join("{Training_Dir}", "training.hdf5"),
    temperature_K = 298.15,
    time_equilibration_fs = 5000,
    time_total_fs = 50000,
    time_step_fs = 0.5,
    sampling_interval_fs = 50,
    averaging_window_fs = 5000    
    )

properties_config = PropertiesConfig(
    unit_cell = mbe_automation.io.read(os.path.join(work_dir, "{XYZ_Solid}")),
    molecule = mbe_automation.io.read(os.path.join(work_dir, "{XYZ_Molecule}")),
    calculator = mace.calculators.mace_off(model="small"),
    optimize_lattice_vectors = False,
    optimize_volume = False,
    preserve_space_group = True,
    supercell_radius = 30.0,
    supercell_displacement = 0.01,
    properties_dir = work_dir,
    hdf5_dataset = os.path.join(work_dir, "properties.hdf5")
)

if Training:
    mbe_automation.workflows.create_training_dataset_mace(training_config)
    
if HarmonicProperties:
    mbe_automation.workflows.compute_harmonic_properties(properties_config)



