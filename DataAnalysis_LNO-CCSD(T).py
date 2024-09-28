#!/usr/bin/env python3

# ------------------------------------------- User's Input -----------------------------------------

ProjectDirectory            = "{PROJECT_DIR}"

                                                   #
                                                   # Available methods:
                                                   #
                                                   # "rPT2", "RPA+MBPT3", "MBPT3",
                                                   # "RPA", "RPA+ALL_CORRECTIONS", "MP3",
                                                   # "JCTC2024", "ph-RPA(3)"                                                   
                                                   #
                                                 
Method                      = "JCTC2024"

SmallBasisXNumber           = 3

# --------------------------------------- End of User's Input --------------------------------------

import sys
import os
sys.path.append(os.path.abspath("{ROOT_DIR}"))

import CSV_MRCC

CSV_MRCC.Make(ProjectDirectory, Method, SmallBasisXNumber)
