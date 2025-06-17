import numpy as np
import os.path
from mbe_automation.configs.properties import PropertiesConfig
import mbe_automation.workflows
import mbe_automation.io
from fairchem.core import pretrained_mlip, FAIRChemCalculator
#
# Meta's Universal Model for Atoms
#
# Select your specific task:
# oc20: catalysis
# omat: inorganic materials
# omol: molecules
# odac: MOFs
# omc: molecular crystals
#
uma_model = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")
uma_calc = FAIRChemCalculator(uma_model, task_name="omc")
work_dir = os.path.abspath(os.path.dirname(__file__))
properties_config = PropertiesConfig(
    unit_cell = mbe_automation.io.read(os.path.join(work_dir, "{XYZ_Solid}")),
    molecule = mbe_automation.io.read(os.path.join(work_dir, "{XYZ_Molecule}")),
    calculator = uma_calc,
    optimize_lattice_vectors = False,
    optimize_volume = False,
    preserve_space_group = True,
    supercell_radius = 30.0,
    supercell_displacement = 0.01,
    properties_dir = work_dir,
    hdf5_dataset = os.path.join(work_dir, "properties.hdf5")
)

mbe_automation.workflows.compute_harmonic_properties(properties_config)



