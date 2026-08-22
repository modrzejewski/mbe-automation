#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -A pl0415-03
#SBATCH -p altair 
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --mem=180gb
#SBATCH --array=1-{n_tasks}

module load python
module load ifort
module load impi
module load mkl

TARGET=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" tasks.txt)
TARGET_BASE="${{TARGET%.*}}"

~/beyond-rpa/bin/run -np 1 -nt 48 "$TARGET" > "${{TARGET_BASE}}.log"
