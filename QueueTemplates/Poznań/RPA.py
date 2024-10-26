#!/usr/bin/env python3
#SBATCH --job-name="{FIRST_SYSTEM}-{LAST_SYSTEM}-{SYSTEM_TYPE}-{BASIS_TYPE}"
#SBATCH -A pl0415-01
#SBATCH -p altair 
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00 

import os
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

os.system(f"module load python/3.10.7; module load ifort; module load impi; ~/beyond-rpa/bin/run -np 1 -nt 48 '{{InpPath}}' >& '{{LogPath}}'")
