#!/usr/bin/env python3

# ------------------------------------------- User's Input -----------------------------------------

ProjectDirectory            = "./"

# --------------------------------------- End of User's Input --------------------------------------

import mbe_automation.outputs.hf_pbc

mbe_automation.outputs.hf_pbc.lattice_energies(ProjectDirectory)
