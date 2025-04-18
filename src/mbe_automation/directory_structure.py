import sys
import os
from os import path
import shutil

SUBSYSTEM_LABELS = {
    "dimers":    ["AB", "A", "B"],
    "trimers":   ["ABC", "A", "B", "C", "AB", "BC", "AC"],
    "tetramers": ["ABCD", "A", "B", "C", "AB", "BC", "AC", "D", "AD", "BC", "CD", "ABC", "ABD", "ACD", "BCD"]
    }

def SetUp(ProjectDir, MethodsMBE, MethodsPBC):
    global ROOT_DIR, PROJECT_DIR, INP_DIRS, LOG_DIRS, CSV_DIRS
    global XYZ_DIRS, QUEUE_DIRS, QUEUE_MAIN_SCRIPT
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PROJECT_DIR = path.realpath(ProjectDir)
    INP_DIRS, LOG_DIRS, CSV_DIRS, XYZ_DIRS, QUEUE_DIRS = {}, {}, {}, {}, {}
    #
    # If the project directory already exists, move the existing contents
    # to a backup location
    #
    if os.path.exists(PROJECT_DIR):
        print(f"Project directory already exists")
        MaxCounter = 1000
        Success = False
        for i in range(MaxCounter):
            BackupPath = PROJECT_DIR + f"_{i}"
            if not os.path.exists(BackupPath):
                try:
                    shutil.move(src=PROJECT_DIR, dst=BackupPath)
                    Success = True
                    break
                except Exception as e:
                    print(f"{e}")
                    sys.exit()
        if Success:
            print(f"Moved existing contents to {BackupPath}")
        else:
            print(f"Clean up your directory manually before proceeding with a new project")
            sys.exit()
    
    os.makedirs(PROJECT_DIR)
    
    for SystemType in ("monomers-supercell", "monomers-relaxed", "dimers", "trimers", "tetramers", "supercell", "unitcell"):
        XYZ_DIRS[SystemType] = path.join(PROJECT_DIR, "xyz", SystemType)
        os.makedirs(XYZ_DIRS[SystemType], exist_ok=True)

    for Method in MethodsMBE:
        INP_DIRS[Method], LOG_DIRS[Method], QUEUE_DIRS[Method], CSV_DIRS[Method] = {}, {}, {}, {}
        for SystemType in ("monomers-supercell", "monomers-relaxed", "dimers", "trimers", "tetramers"):
            INP_DIRS[Method][SystemType] = {}
            LOG_DIRS[Method][SystemType] = {}
            QUEUE_DIRS[Method][SystemType] = {}
            if Method != "DLPNO-CCSD(T)":
                BasisTypes = ["small-basis", "large-basis"]
            else:
                #
                # DLPNO-CC calculations in Orca employ Orca's internal algorithm for basis set extrapolation.
                # As a result, a single-point result is already a CBS estimate and no separate small basis
                # and larger basis calculations are needed.
                #
                BasisTypes = ["orca-extrapolation"]
            for BasisType in BasisTypes:
                QUEUE_DIRS[Method][SystemType][BasisType] = path.join(PROJECT_DIR, "queue", Method, SystemType, BasisType)
                os.makedirs(QUEUE_DIRS[Method][SystemType][BasisType], exist_ok=True)
                INP_DIRS[Method][SystemType][BasisType] = path.join(PROJECT_DIR, "inputs", Method, SystemType, BasisType)
                os.makedirs(INP_DIRS[Method][SystemType][BasisType], exist_ok=True)
                LOG_DIRS[Method][SystemType][BasisType] = path.join(PROJECT_DIR, "logs", Method, SystemType, BasisType)
                os.makedirs(LOG_DIRS[Method][SystemType][BasisType], exist_ok=True)
                #
                # For the methods where we can't use a single input file for the cluster
                # and all of its subsystems, we need to create separate subdirectories
                # for each single point energy calculation
                #
                if Method in ["LNO-CCSD(T)"]:
                    if SystemType not in ["monomers-supercell", "monomers-relaxed"]:
                        for Subsystem in SUBSYSTEM_LABELS[SystemType]:
                            subsystem_inp = path.join(INP_DIRS[Method][SystemType][BasisType], Subsystem)
                            subsystem_log = path.join(LOG_DIRS[Method][SystemType][BasisType], Subsystem)
                            os.makedirs(subsystem_inp, exist_ok=True)                                
                            os.makedirs(subsystem_log, exist_ok=True)
                            
        for SystemType in ["monomers", "dimers", "trimers", "tetramers"]:
            CSV_DIRS[Method][SystemType] = path.join(PROJECT_DIR, "csv", Method, SystemType)
            os.makedirs(CSV_DIRS[Method][SystemType], exist_ok=True)

    for Method in MethodsPBC:
        INP_DIRS[Method+"(PBC)"] = path.join(PROJECT_DIR, "PBC", Method)
        LOG_DIRS[Method+"(PBC)"] = path.join(PROJECT_DIR, "PBC", Method)
        QUEUE_DIRS[Method+"(PBC)"] = path.join(PROJECT_DIR, "PBC", Method)
        os.makedirs(INP_DIRS[Method+"(PBC)"], exist_ok=True)
            
    QUEUE_MAIN_SCRIPT = {}
    for Method in MethodsMBE:
        QUEUE_MAIN_SCRIPT[Method] = path.join(PROJECT_DIR, f"RunTaskArray_{Method}.py")
        
    return


