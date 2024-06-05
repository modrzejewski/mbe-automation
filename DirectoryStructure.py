#!/usr/bin/env python3
#
# Coded by Marcin Modrzejewski (Charles Univ, Prague 2018)
#
#
import os
from os import path

ROOT_DIR = path.dirname(path.realpath(__file__))
PROJECT_DIR = path.join(ROOT_DIR, "Project")

INP_DIRS = {}
LOG_DIRS = {}
CSV_DIRS = {}
XYZ_DIRS = {}
QUEUE_DIRS = {}
for SystemType in ("supercell", "monomers", "dimers", "trimers", "tetramers"):
    INP_DIRS[SystemType] = {}
    LOG_DIRS[SystemType] = {}
    QUEUE_DIRS[SystemType] = {}
    CSV_DIRS[SystemType] = path.join(ROOT_DIR, "Project", "csv", SystemType)
    XYZ_DIRS[SystemType] = path.join(ROOT_DIR, "Project", "xyz", SystemType)    
    if not path.exists(CSV_DIRS[SystemType]):
        os.makedirs(CSV_DIRS[SystemType])
    if not path.exists(XYZ_DIRS[SystemType]):
        os.makedirs(XYZ_DIRS[SystemType])
    for BasisType in ("small-basis", "large-basis"):
        INP_DIRS[SystemType][BasisType] = path.join(ROOT_DIR, "Project", "inputs", SystemType, BasisType)
        LOG_DIRS[SystemType][BasisType] = path.join(ROOT_DIR, "Project", "logs", SystemType, BasisType)
        QUEUE_DIRS[SystemType][BasisType] = path.join(ROOT_DIR, "Project", "queue", SystemType, BasisType)
        if not path.exists(INP_DIRS[SystemType][BasisType]):
            os.makedirs(INP_DIRS[SystemType][BasisType])
        if not path.exists(LOG_DIRS[SystemType][BasisType]):
            os.makedirs(LOG_DIRS[SystemType][BasisType])
        if not path.exists(QUEUE_DIRS[SystemType][BasisType]):
            os.makedirs(QUEUE_DIRS[SystemType][BasisType])
QUEUE_MAIN_SCRIPT = path.join(PROJECT_DIR, "queue", "RunTaskArray.py")
SUPERCELL_XYZ_DIR = path.join(ROOT_DIR, "Project", "xyz", "supercell")

MONOMER_XYZ_DIR = path.join(ROOT_DIR, "Project", "xyz", "monomers")
DIMER_XYZ_DIR = path.join(ROOT_DIR, "Project", "xyz", "dimers")
TRIMER_XYZ_DIR = path.join(ROOT_DIR, "Project", "xyz", "trimers")
TETRAMER_XYZ_DIR = path.join(ROOT_DIR, "Project", "xyz", "tetramers")
if not path.exists(SUPERCELL_XYZ_DIR):
   os.makedirs(SUPERCELL_XYZ_DIR)
if not path.exists(MONOMER_XYZ_DIR):
   os.makedirs(MONOMER_XYZ_DIR)
if not path.exists(DIMER_XYZ_DIR):
   os.makedirs(DIMER_XYZ_DIR)
if not path.exists(TRIMER_XYZ_DIR):
   os.makedirs(TRIMER_XYZ_DIR)
if not path.exists(TETRAMER_XYZ_DIR):
   os.makedirs(TETRAMER_XYZ_DIR)

TMP_DIR = path.join(ROOT_DIR, "Project", "tmp")
if not path.exists(TMP_DIR):
   os.makedirs(TMP_DIR)

def CleanDir(directory, extension):
    if os.path.exists(directory):
        ext = extension.upper()
        for f in sorted(os.listdir(directory)):
            if f.upper().endswith(ext):
                full_path = path.join(directory, f)
                os.remove(full_path)

def CleanAllDirs():
    for SystemType in ("monomers", "dimers", "trimers", "tetramers"):
        CleanDir(XYZ_DIRS[SystemType], ".xyz")
        CleanDir(CSV_DIRS[SystemType], ".csv")
        for BasisType in ("small-basis", "large-basis"):
            CleanDir(INP_DIRS[SystemType][BasisType], ".inp")
            CleanDir(INP_DIRS[SystemType][BasisType], ".conf")
            CleanDir(LOG_DIRS[SystemType][BasisType], ".log")
            CleanDir(QUEUE_DIRS[SystemType][BasisType], ".py")
    CleanDir(TMP_DIR, ".xyz")    
    CleanDir(SUPERCELL_XYZ_DIR, ".xyz")    
    if path.exists(QUEUE_MAIN_SCRIPT):
        os.remove(QUEUE_MAIN_SCRIPT)

        


