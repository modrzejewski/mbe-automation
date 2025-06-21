import numpy as np
import os.path
import mace.calculators
from mbe_automation.configs.training import TrainingConfig
import mbe_automation.workflows
import mbe_automation.io

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))
training_dir = os.path.join(work_dir, "training")

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("~/models/mace/MACE-OFF24_medium.model"),
    compute_atomic_stresses=False,
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

training_config = TrainingConfig(
    calculator = mace_calc,
    unit_cell = mbe_automation.io.read(os.path.join(work_dir, xyz_solid)),
    molecule = mbe_automation.io.read(os.path.join(work_dir, xyz_molecule)),
    supercell_radius = 20.0,
    training_dir = training_dir,
    hdf5_dataset = os.path.join(training_dir, "training.hdf5"),
    temperature_K = 298.15,
    time_equilibration_fs = 5000,
    time_total_fs = 50000,
    time_step_fs = 0.5,
    sampling_interval_fs = 50,
    averaging_window_fs = 5000    
    )

mbe_automation.workflows.create_training_dataset_mace(training_config)
    


