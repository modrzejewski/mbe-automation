#!/usr/bin/env python3
#
# Coded by Marcin Modrzejewski (Charles Univ, Prague 2018)
#
#
import os
from os import path

def SetUp(ProjectDir):
    global ROOT_DIR, PROJECT_DIR, INP_DIRS, LOG_DIRS, CSV_DIRS
    global XYZ_DIRS, QUEUE_DIRS, QUEUE_MAIN_SCRIPT
    
    ROOT_DIR = path.dirname(path.realpath(__file__))
    PROJECT_DIR = path.join(ProjectDir)
    INP_DIRS, LOG_DIRS, CSV_DIRS, XYZ_DIRS, QUEUE_DIRS = {}, {}, {}, {}, {}
    for SystemType in ("supercell", "monomers", "dimers", "trimers", "tetramers"):
        INP_DIRS[SystemType] = {}
        LOG_DIRS[SystemType] = {}
        QUEUE_DIRS[SystemType] = {}
        CSV_DIRS[SystemType] = path.join(PROJECT_DIR, "csv", SystemType)
        XYZ_DIRS[SystemType] = path.join(PROJECT_DIR, "xyz", SystemType)    
        if not path.exists(CSV_DIRS[SystemType]):
            os.makedirs(CSV_DIRS[SystemType])
        if not path.exists(XYZ_DIRS[SystemType]):
            os.makedirs(XYZ_DIRS[SystemType])
        for BasisType in ("small-basis", "large-basis"):
            INP_DIRS[SystemType][BasisType] = path.join(PROJECT_DIR, "inputs", SystemType, BasisType)
            LOG_DIRS[SystemType][BasisType] = path.join(PROJECT_DIR, "logs", SystemType, BasisType)
            QUEUE_DIRS[SystemType][BasisType] = path.join(PROJECT_DIR, "queue", SystemType, BasisType)
            if not path.exists(INP_DIRS[SystemType][BasisType]):
                os.makedirs(INP_DIRS[SystemType][BasisType])
            if not path.exists(LOG_DIRS[SystemType][BasisType]):
                os.makedirs(LOG_DIRS[SystemType][BasisType])
            if not path.exists(QUEUE_DIRS[SystemType][BasisType]):
                os.makedirs(QUEUE_DIRS[SystemType][BasisType])
    QUEUE_MAIN_SCRIPT = path.join(PROJECT_DIR, "queue", "RunTaskArray.py")
    CleanAllDirs()


def CleanDir(directory, extension):
    if os.path.exists(directory):
        ext = extension.upper()
        for f in sorted(os.listdir(directory)):
            if f.upper().endswith(ext):
                full_path = path.join(directory, f)
                os.remove(full_path)

def CleanAllDirs():
    for SystemType in ("supercell", "monomers", "dimers", "trimers", "tetramers"):
        CleanDir(XYZ_DIRS[SystemType], ".xyz")
        CleanDir(CSV_DIRS[SystemType], ".csv")
        for BasisType in ("small-basis", "large-basis"):
            CleanDir(INP_DIRS[SystemType][BasisType], ".inp")
            CleanDir(INP_DIRS[SystemType][BasisType], ".conf")
            CleanDir(LOG_DIRS[SystemType][BasisType], ".log")
            CleanDir(QUEUE_DIRS[SystemType][BasisType], ".py")
            
    if path.exists(QUEUE_MAIN_SCRIPT):
        os.remove(QUEUE_MAIN_SCRIPT)

        


