#!/usr/bin/env python3
#SBATCH --job-name="{FIRST}-{LAST}-{SYSTEM_TYPE}-{BASIS_TYPE}"
#SBATCH -A plgrpa2023-cpu
#SBATCH -p plgrid 
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00 

import os
#
# Total number of {SYSTEM_TYPE}/{BASIS_TYPE}: {NTASKS}
# This script is for {SYSTEM_TYPE}/{BASIS_TYPE} {FIRST}-{LAST}
#
ThisJob = {FIRST} + int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
InpDir = "{INP_DIR}"
LogDir = "{LOG_DIR}"
AllFiles = sorted(os.listdir(InpDir)) 
InpFile = AllFiles[ThisJob-1]
LogFile = os.path.splitext(InpFile)[0] + ".log"
InpPath = os.path.join(InpDir, InpFile)
LogPath = os.path.join(LogDir, LogFile)

os.system(f"module load intel/2021b; /net/people/plgrid/plgmodrzej/beyond-rpa/bin/run -np 1 -nt 48 '{{InpPath}}' >& '{{LogPath}}'")
