import numpy as np
import os.path
import mace.calculators
from mbe_automation.configs.quasi_harmonic import QuasiHarmonicConfig
import mbe_automation.workflows.quasi_harmonic
import mbe_automation.io
import torch

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))
properties_dir = os.path.join(work_dir, "properties")

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("{mlip_parameters}"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

properties_config = QuasiHarmonicConfig(
    unit_cell = mbe_automation.io.read(os.path.join(work_dir, xyz_solid)),
    molecule = mbe_automation.io.read(os.path.join(work_dir, xyz_molecule)),
    temperatures = np.array([5.0, 200.0, 300.0]),
    calculator = mace_calc,
    supercell_radius = 25.0,
    properties_dir = properties_dir,
    hdf5_dataset = os.path.join(properties_dir, "properties.hdf5")
)

mbe_automation.workflows.quasi_harmonic.run(properties_config)



