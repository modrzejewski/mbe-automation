#!/usr/bin/env python3
#
# Coded by Marcin Modrzejewski (Charles Univ, Prague 2018)
#
#
import os
from os import path

SUBSYSTEM_LABELS = {
    "dimers":    ["AB", "A", "B"],
    "trimers":   ["ABC", "A", "B", "C", "AB", "BC", "AC"],
    "tetramers": ["ABCD", "A", "B", "C", "AB", "BC", "AC", "D", "AD", "BC", "CD", "ABC", "ABD", "ACD", "BCD"]
    }

def SetUp(ProjectDir, Methods):
    global ROOT_DIR, PROJECT_DIR, INP_DIRS, LOG_DIRS, CSV_DIRS
    global XYZ_DIRS, QUEUE_DIRS, QUEUE_MAIN_SCRIPT
    
    ROOT_DIR = path.dirname(path.realpath(__file__))
    PROJECT_DIR = path.join(ProjectDir)
    INP_DIRS, LOG_DIRS, CSV_DIRS, XYZ_DIRS, QUEUE_DIRS = {}, {}, {}, {}, {}
    for Method in Methods:
        INP_DIRS[Method], LOG_DIRS[Method], QUEUE_DIRS[Method], CSV_DIRS[Method] = {}, {}, {}, {}
    for SystemType in ("supercell", "monomers", "dimers", "trimers", "tetramers"):
        for Method in Methods:
            INP_DIRS[Method][SystemType] = {}
            LOG_DIRS[Method][SystemType] = {}
            QUEUE_DIRS[Method][SystemType] = {}
            CSV_DIRS[Method][SystemType] = path.join(PROJECT_DIR, "csv", Method, SystemType)
            if not path.exists(CSV_DIRS[Method][SystemType]):
                os.makedirs(CSV_DIRS[Method][SystemType])
        XYZ_DIRS[SystemType] = path.join(PROJECT_DIR, "xyz", SystemType)    
        if not path.exists(XYZ_DIRS[SystemType]):
            os.makedirs(XYZ_DIRS[SystemType])
        for BasisType in ("small-basis", "large-basis", "no-extrapolation"):
            for Method in Methods:
                INP_DIRS[Method][SystemType][BasisType] = path.join(PROJECT_DIR, "inputs", Method, SystemType, BasisType)
                LOG_DIRS[Method][SystemType][BasisType] = path.join(PROJECT_DIR, "logs", Method, SystemType, BasisType)
                if not path.exists(INP_DIRS[Method][SystemType][BasisType]):
                    os.makedirs(INP_DIRS[Method][SystemType][BasisType])
                if not path.exists(LOG_DIRS[Method][SystemType][BasisType]):
                    os.makedirs(LOG_DIRS[Method][SystemType][BasisType])
                if Method == "LNO-CCSD(T)":
                    if BasisType in ("small-basis", "large-basis") and SystemType in SUBSYSTEM_LABELS:
                        for Subsystem in SUBSYSTEM_LABELS[SystemType]:
                            subsystem_inp = path.join(INP_DIRS[Method][SystemType][BasisType], Subsystem)
                            subsystem_log = path.join(LOG_DIRS[Method][SystemType][BasisType], Subsystem)
                            if not path.exists(subsystem_inp):
                                os.makedirs(subsystem_inp)                                
                            if not path.exists(subsystem_log):
                                os.makedirs(subsystem_log)
                QUEUE_DIRS[Method][SystemType][BasisType] = path.join(PROJECT_DIR, "queue", Method, SystemType, BasisType)
                if not path.exists(QUEUE_DIRS[Method][SystemType][BasisType]):
                    os.makedirs(QUEUE_DIRS[Method][SystemType][BasisType])

    QUEUE_MAIN_SCRIPT = {}
    for Method in Methods:
        QUEUE_MAIN_SCRIPT[Method] = path.join(PROJECT_DIR, "queue", f"{Method}-RunTaskArray.py")
    CleanAllDirs(Methods)


def CleanDir(directory, extension):
    if os.path.exists(directory):
        ext = extension.upper()
        for f in sorted(os.listdir(directory)):
            if f.upper().endswith(ext):
                full_path = path.join(directory, f)
                os.remove(full_path)

def CleanAllDirs(Methods):
    for SystemType in ("supercell", "monomers", "dimers", "trimers", "tetramers"):
        CleanDir(XYZ_DIRS[SystemType], ".xyz")
    for Method in Methods:
        for SystemType in ("supercell", "monomers", "dimers", "trimers", "tetramers"):
            CleanDir(CSV_DIRS[Method][SystemType], ".csv")
            for BasisType in ("small-basis", "large-basis", "no-extrapolation"):
                CleanDir(INP_DIRS[Method][SystemType][BasisType], ".inp")
                CleanDir(INP_DIRS[Method][SystemType][BasisType], ".conf")
                CleanDir(LOG_DIRS[Method][SystemType][BasisType], ".log")            
                CleanDir(QUEUE_DIRS[Method][SystemType][BasisType], ".py")
            
        if path.exists(QUEUE_MAIN_SCRIPT[Method]):
            os.remove(QUEUE_MAIN_SCRIPT[Method])

        


