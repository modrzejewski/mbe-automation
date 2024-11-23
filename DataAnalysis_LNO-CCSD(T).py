#!/usr/bin/env python3

# ------------------------------------------- User's Input -----------------------------------------

ProjectDirectory            = "{PROJECT_DIR}"

                                                   #
                                                   # Available methods:
                                                   #
                                                   # LNO-CCSD(T)
                                                   # LNO-CCSD
                                                   #
                                                 
Method                      = "LNO-CCSD(T)"

SmallBasisXNumber           = 3

# --------------------------------------- End of User's Input --------------------------------------

import sys
import os
sys.path.append(os.path.abspath("{ROOT_DIR}"))

import CSV_MRCC

CSV_MRCC.Make(ProjectDirectory, Method, SmallBasisXNumber)
