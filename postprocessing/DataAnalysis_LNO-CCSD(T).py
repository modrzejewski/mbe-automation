#!/usr/bin/env python3

# ------------------------------------------- User's Input -----------------------------------------

ProjectDirectory            = "./"

                                                   #
                                                   # Available methods:
                                                   #
                                                   # LNO-CCSD(T)
                                                   # LNO-CCSD
                                                   #
                                                 
Method                      = "LNO-CCSD(T)"

SmallBasisXNumber           = 3

# --------------------------------------- End of User's Input --------------------------------------

import mbe_automation.readout.mrcc

mbe_automation.readout.mrcc.Make(ProjectDirectory, Method, SmallBasisXNumber)
