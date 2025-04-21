#!/usr/bin/env python3
#SBATCH --job-name="{FIRST_SYSTEM}-{LAST_SYSTEM}-{SYSTEM_TYPE}-{BASIS_TYPE}"
#SBATCH -A plgrpa2025-cpu
#SBATCH -p plgrid 
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=72:00:00 

import os
import shutil

SUBSYSTEM_LABELS = {{
    "monomers-relaxed": [""],
    "monomers-supercell": [""],
    "dimers":    ["AB", "A", "B"],
    "trimers":   ["ABC", "A", "B", "C", "AB", "BC", "AC"],
    "tetramers": ["ABCD", "A", "B", "C", "AB", "BC", "AC", "D", "AD", "BC", "CD", "ABC", "ABD", "ACD", "BCD"]
    }}

#
# Number of cores used by MRCC
#
os.environ["OMP_NUM_THREADS"] = "48"
os.environ["MKL_NUM_THREADS"] = "48"
#
# Total number of {SYSTEM_TYPE}/{BASIS_TYPE}: {NTASKS}
# This script is for {SYSTEM_TYPE}/{BASIS_TYPE} {FIRST_SYSTEM}-{LAST_SYSTEM}
#
ThisJob = {OFFSET} + int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
for Subsystem in SUBSYSTEM_LABELS["{SYSTEM_TYPE}"]:
    InpDir = os.path.join("{INP_DIR}", Subsystem)
    LogDir = os.path.join("{LOG_DIR}", Subsystem)
    AllFiles = sorted([x for x in os.listdir(InpDir) if x.endswith(".inp")])
    InpFile = AllFiles[ThisJob-1]
    LogFile = os.path.splitext(InpFile)[0] + ".log"
    InpPath = os.path.join(InpDir, InpFile)
    LogPath = os.path.join(LogDir, LogFile)
    #
    # Skip to the next system if the log file already exits. Thanks to this feature,
    # individual job restarts can be done simply by deleting the invalid log files
    # and restarting the whole job batch.
    #
    if os.path.exists(LogPath):
        continue
    ScratchDir = os.path.join(LogDir, os.path.splitext(InpFile)[0])
    if not os.path.exists(ScratchDir):
        os.makedirs(ScratchDir)
    #
    # Copy input file into the scratch directory.
    #
    source = InpPath
    destination = os.path.join(ScratchDir, "MINP")
    shutil.copy(source, destination)
    current_dir = os.getcwd()
    os.chdir(ScratchDir)
    os.system(f"module load mrcc; dmrcc >& '{{LogPath}}'")
    os.chdir(current_dir)
    #
    # Remove scratch
    #
    shutil.rmtree(ScratchDir)
