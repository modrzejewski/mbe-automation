#!/usr/bin/env python3

# ------------------------------------------- User's Input -----------------------------------------

ProjectDirectory            = "./"

                                                   #
                                                   # Available methods:
                                                   #
                                                   # "rPT2", "RPA+MBPT3", "MBPT3",
                                                   # "RPA", "RPA+ALL_CORRECTIONS", "MP3"
                                                   #
                                                 
Method                      = "RPA+MBPT3"

SmallBasisXNumber           = 3

# --------------------------------------- End of User's Input --------------------------------------

import CSV

CSV.Make(ProjectDirectory, Method, SmallBasisXNumber)
