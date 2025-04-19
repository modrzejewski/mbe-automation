#!/usr/bin/env python3

# ------------------------------------------- User's Input -----------------------------------------

ProjectDirectory            = "./"

# --------------------------------------- End of User's Input --------------------------------------

import mbe_automation.readout.hf_pbc

mbe_automation.readout.hf_pbc.lattice_energies(ProjectDirectory)
