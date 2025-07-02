import numpy as np
import os.path
import mace.calculators
from mbe_automation.configs.properties import PropertiesConfig
import mbe_automation.workflows
import mbe_automation.io
import torch

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))
properties_dir = os.path.join(work_dir, "properties")

mace_calc = mace.calculators.MACECalculator(
    model_paths="{mace_model}",
    compute_atomic_stresses=False,
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

properties_config = PropertiesConfig(
    unit_cell = mbe_automation.io.read(os.path.join(work_dir, xyz_solid)),
    molecule = mbe_automation.io.read(os.path.join(work_dir, xyz_molecule)),
    calculator = mace_calc,
    optimize_lattice_vectors = False,
    optimize_volume = False,
    preserve_space_group = True,
    supercell_radius = 20.0,
    supercell_displacement = 0.01,
    properties_dir = properties_dir,
    hdf5_dataset = os.path.join(properties_dir, "properties.hdf5")
)

mbe_automation.workflows.compute_harmonic_properties(properties_config)



