import numpy as np
import os.path
import mace.calculators
from mbe_automation.configs.training import TrainingConfig
from mbe_automation.configs.properties import PropertiesConfig
import mbe_automation.workflows
import mbe_automation.io

training_dir = "{Training_Dir}"
xyz_solid = "{XYZ_Solid}"
xyz_molecule = "{XYZ_Molecule}"

Training = False
HarmonicProperties = True

work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("~/models/mace/MACE-OFF24_medium.model"),
    compute_atomic_stresses=False,
    device="cuda"
)

# training_config = TrainingConfig(
#     calculator = mace.calculators.mace_off(model="small"),
#     unit_cell = mbe_automation.io.read(os.path.join(work_dir, xyz_solid)),
#     molecule = mbe_automation.io.read(os.path.join(work_dir, xyz_molecule)),
#     supercell_radius = 15.0,
#     training_dir = training_dir,
#     hdf5_dataset = os.path.join(training_dir, "training.hdf5"),
#     temperature_K = 298.15,
#     time_equilibration_fs = 5000,
#     time_total_fs = 50000,
#     time_step_fs = 0.5,
#     sampling_interval_fs = 50,
#     averaging_window_fs = 5000    
#     )

properties_config = PropertiesConfig(
    unit_cell = mbe_automation.io.read(os.path.join(work_dir, xyz_solid)),
    molecule = mbe_automation.io.read(os.path.join(work_dir, xyz_molecule)),
    calculator = mace_calc,
    optimize_lattice_vectors = False,
    optimize_volume = False,
    preserve_space_group = True,
    supercell_radius = 20.0,
    supercell_displacement = 0.01,
    properties_dir = work_dir,
    hdf5_dataset = os.path.join(work_dir, "properties.hdf5")
)

# if Training:
#     mbe_automation.workflows.create_training_dataset_mace(training_config)
    
if HarmonicProperties:
    mbe_automation.workflows.compute_harmonic_properties(properties_config)



