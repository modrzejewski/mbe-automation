#!/usr/bin/env python3
#SBATCH --job-name="{FIRST_SYSTEM}-{LAST_SYSTEM}-{SYSTEM_TYPE}-{BASIS_TYPE}"
#SBATCH -A plgrpa2025-cpu
#SBATCH -p plgrid 
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00 

import os
import os.path
import sys
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
if os.path.exists(LogPath):
    sys.exit(0)

if "I_MPI_PMI_LIBRARY" in os.environ:
    del os.environ["I_MPI_PMI_LIBRARY"]
os.environ["I_MPI_HYDRA_BOOTSTRAP"] = "ssh"
os.environ["I_MPI_OFI_PROVIDER"] = "tcp"

# In case of MPI-related bugs, set the following parameters
# to get debugging data:
#
# export I_MPI_DEBUG=5
# export I_MPI_HYDRA_DEBUG=1
# export I_MPI_OFI_PROVIDER_DUMP=1

os.system(f"module load intel/2021b; /net/people/plgrid/plgmodrzej/beyond-rpa/bin/run -np 1 -nt 48 '{{InpPath}}' >& '{{LogPath}}'")
