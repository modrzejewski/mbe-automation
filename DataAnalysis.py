#!/usr/bin/env python3

# ------------------------------------------- User's Input -----------------------------------------

ProjectDirectory            = "{PROJECT_DIR}"

                                                   #
                                                   # Available methods:
                                                   #
                                                   # "rPT2", "RPA+MBPT3", "MBPT3",
                                                   # "RPA", "RPA+ALL_CORRECTIONS", "MP3"
                                                   #
                                                 
Method                      = "RPA+MBPT3"

SmallBasisXNumber           = 3

# --------------------------------------- End of User's Input --------------------------------------

import sys
import os
sys.path.append(os.path.abspath("{ROOT_DIR}"))

import CSV

CSV.Make(ProjectDirectory, Method, SmallBasisXNumber)
