#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -A plgrpa2025-cpu
#SBATCH -p plgrid 
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=72:00:00
#SBATCH --array=1-{n_tasks}

module load mrcc

export PATH=$PATH:~/mrcc

TARGET=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" tasks.txt)
cd $(dirname $TARGET)

dmrcc > dmrcc.log
