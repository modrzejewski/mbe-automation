#!/usr/bin/env python3
#SBATCH --job-name="{TITLE}"
#SBATCH -A pl0458-01
#SBATCH -p altair 
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --mem=180gb

import os
import os.path
import sys
import subprocess

InpScript = "{INP_SCRIPT}"
LogFile = "{LOG_FILE}"

os.environ["OMP_NUM_THREADS"] = "48"
os.environ["MKL_NUM_THREADS"] = "48"

cmd = f"module load python; module load ifort; module load impi; module load mkl; python '{{InpScript}}'"
with open(LogFile, "w") as log_file:
    process = subprocess.Popen(cmd, shell=True, stdout=log_file,
                               stderr=subprocess.STDOUT, bufsize=1,
                               universal_newlines=True)
    process.communicate()


    



