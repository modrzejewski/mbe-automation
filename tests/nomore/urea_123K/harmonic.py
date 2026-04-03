import mbe_automation
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
from mbe_automation.configs.structure import Minimum
from mbe_automation import Structure

xyz_solid = "urea_123K_xray.cif"

#mace_calc = MACE(model_path="~/models/mace/mace-mh-1.model", head="omol")
mace_calc = MACE(model_path="./MACE_model_swa.model")

# Create custom relaxation settings (optional)
relaxation_config = Minimum(
    cell_relaxation="only_atoms",
    max_force_on_atom_eV_A=1.0E-4,
    symmetrize_final_structure=False,
)

crystal, u_cart_exp = mbe_automation.storage.cif.read_cif_with_apds("./urea_123K_xray.cif")
print(u_cart_exp)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=crystal,
    molecule=None,
    temperatures_K=np.array([123.0]),
    calculator=mace_calc,
    supercell_matrix=np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
    thermal_expansion=False,
    filter_out_imaginary_optical=False,
    dataset="urea_harmonic_cif.hdf5",
    relaxation=relaxation_config
)

mbe_automation.run(properties_config)


