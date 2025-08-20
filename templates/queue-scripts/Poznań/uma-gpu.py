#!/usr/bin/env python3
#SBATCH --job-name="UMA"
#SBATCH -A pl0415-02
#SBATCH --partition=tesla
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=180gb

import os
import os.path
import sys
import subprocess

InpScript = "{inp_script}"
LogFile = "{log_file}"

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
#
# Set up virtual environment
# Make sure that the environment works properly on the coputer nodes.
# The libaries seen from the login nodes and compute nodes can be
# different!
# It's a good idea to log on to a compute node via an interactive slurm
# session and create a venv from within this session.
#
virtual_environment = os.path.expanduser("~/.virtualenvs/uma-compute")
virtual_environment = os.path.realpath(virtual_environment)
activate_env = os.path.realpath(os.path.join(virtual_environment, "bin", "activate"))
cmd = f"module load python/3.11.9-gcc-11.5.0-5l7rvgy && . {{activate_env}} && python {{InpScript}}"

with open(LogFile, "w") as log_file:
    process = subprocess.Popen(cmd, shell=True, stdout=log_file,
                               stderr=subprocess.STDOUT, bufsize=1,
                               universal_newlines=True)
    process.communicate()
    

