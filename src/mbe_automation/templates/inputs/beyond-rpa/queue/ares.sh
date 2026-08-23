#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -A plgrpa2025-cpu
#SBATCH -p plgrid 
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00

unset I_MPI_PMI_LIBRARY
export I_MPI_HYDRA_BOOTSTRAP=ssh
export I_MPI_OFI_PROVIDER=tcp

module load intel/2021b

TARGET=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" tasks.txt)
TARGET_BASE="${{TARGET%.*}}"

/net/people/plgrid/plgmodrzej/beyond-rpa/bin/run -np 1 -nt 48 "$TARGET" > "${{TARGET_BASE}}.log"
