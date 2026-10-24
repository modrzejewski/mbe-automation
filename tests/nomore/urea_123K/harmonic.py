import mbe_automation
import numpy as np
from mbe_automation.calculators import UMA

import mbe_automation.configs
from mbe_automation.configs.structure import Minimum
from mbe_automation import Structure

xyz_solid = "urea_123K_xray.cif"

uma_calc = UMA(model_path="./uma-m-1p1.pt", task_name="omc")

# Create custom relaxation settings (optional)
relaxation_config = Minimum(
    cell_relaxation="only_atoms",
    max_force_on_atom_eV_A=1.0E-4,
)

crystal = Structure.from_file(xyz_solid)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="uma",
    crystal=crystal,
    molecule=None,
    temperatures_K=np.array([123.0]),
    calculator=uma_calc,
    supercell_matrix=np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
    thermal_expansion=False,
    filter_out_imaginary_optical=False,
    dataset="urea_harmonic_cif.hdf5",
    relaxation=relaxation_config
)

mbe_automation.run(properties_config)


