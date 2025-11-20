import numpy as np
import os.path
import torch
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation.storage import from_xyz_file

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))
properties_dir = 
model_from_file = False
#
# Meta's Universal Model for Atoms
#
if model_from_file: # For some reason loading a model using explicit file path doesn't work
    uma_model = load_predict_unit(
        "{mlip_parameters}",
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )
else:
    uma_model = pretrained_mlip.get_predict_unit(
        "uma-m-1p1",
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )
#
# Select your specific task:
# oc20: catalysis
# omat: inorganic materials
# omol: molecules
# odac: MOFs
# omc: molecular crystals
#
uma_calc = FAIRChemCalculator(uma_model, task_name="omc")

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name = "uma",
    crystal = from_xyz_file(os.path.join(work_dir, xyz_solid)),
    molecule = from_xyz_file(os.path.join(work_dir, xyz_molecule)),
    temperatures_K = np.array([5.0, 200.0, 300.0]),
    calculator = uma_calc,
    supercell_radius = 25.0,
    work_dir = os.path.join(work_dir, "properties"),
    dataset = os.path.join(work_dir, "properties.hdf5")
)

mbe_automation.workflows.quasi_harmonic.run(properties_config)
