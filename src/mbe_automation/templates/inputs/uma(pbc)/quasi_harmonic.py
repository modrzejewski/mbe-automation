import os.path
import numpy as np

import mbe_automation.calculators
import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation import Structure

xyz_solid = "{xyz_solid}"
xyz_molecule = "{xyz_molecule}"
work_dir = os.path.abspath(os.path.dirname(__file__))

#
# Meta's Universal Model for Atoms (UMA)
# Select your specific task:
# oc20: catalysis
# oc22: oxide catalysis
# oc25: solid-liquid catalysis
# omat: inorganic materials
# omol: molecules
# odac: MOFs
# omc: molecular crystals
#
uma_calc = mbe_automation.calculators.UMA(
    model_path="{mlip_parameters}",
    task_name="omc",
    inference_settings="float64",
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="uma",
    crystal=Structure.from_file(os.path.join(work_dir, xyz_solid)),
    molecule=Structure.from_file(os.path.join(work_dir, xyz_molecule)),
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=uma_calc,
    supercell_radius=25.0,
    work_dir=os.path.join(work_dir, "properties"),
    dataset=os.path.join(work_dir, "properties.hdf5"),
)

mbe_automation.workflows.quasi_harmonic.run(properties_config)
