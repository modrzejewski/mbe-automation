#!/usr/bin/env python3

# ------------------------------------------- User's Input -----------------------------------------

ProjectDirectory            = "./"

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

import mbe_automation.outputs.rpa

mbe_automation.csv.rpa.Make(ProjectDirectory, Method, SmallBasisXNumber)
