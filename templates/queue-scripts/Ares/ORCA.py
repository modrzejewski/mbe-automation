#!/usr/bin/env python3
#SBATCH --job-name="{FIRST_SYSTEM}-{LAST_SYSTEM}-{SYSTEM_TYPE}-{BASIS_TYPE}"
#SBATCH -A plgrpa2025-cpu
#SBATCH -p plgrid 
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00 

import os
import shutil

#
# Number of cores used by ORCA
#
os.environ["OMP_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
os.environ["MKL_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
#
# On the ares cluster, it is recommended to use the ORCA environmental
# variable instead of simply orca command.
#
OrcaExecutable = os.environ["ORCA"]
#
# Total number of {SYSTEM_TYPE}/{BASIS_TYPE}: {NTASKS}
# This script is for {SYSTEM_TYPE}/{BASIS_TYPE} {FIRST_SYSTEM}-{LAST_SYSTEM}
#
ThisJob = {OFFSET} + int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
InpDir = "{INP_DIR}"
LogDir = "{LOG_DIR}"
AllFiles = sorted([x for x in os.listdir(InpDir) if x.endswith(".inp")])
InpFile = AllFiles[ThisJob-1]
LogFile = os.path.splitext(InpFile)[0] + ".log"
InpPath = os.path.join(InpDir, InpFile)
LogPath = os.path.join(LogDir, LogFile)
ScratchDir = os.path.join(LogDir, os.path.splitext(InpFile)[0])
if not path.exists(ScratchDir):
    os.makedirs(ScratchDirs)
#
# Copy input file into the scratch directory.
#
source = InpFile
destination = os.path.join(ScratchDir, "input.inp")
shutils.copy(source, destination)
current_dir = os.getcwd()
os.chdir(ScratchDir)
os.system(f"module load orca; {{OrcaExecutable}} input.inp >& '{{LogPath}}'")
os.chdir(current_dir)
#
# Remove scratch
#
shutils.rmtree(ScratchDir)
