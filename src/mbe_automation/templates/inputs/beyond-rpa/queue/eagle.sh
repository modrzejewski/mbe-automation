#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -A pl0415-03
#SBATCH -p altair 
#SBATCH --nodes 1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=32:00:00
#SBATCH --mem=180gb

module load python
module load ifort
module load impi
module load mkl

if [ -n "$I_MPI_PMI_LIBRARY" ]; then
    unset I_MPI_PMI_LIBRARY
fi
export I_MPI_HYDRA_BOOTSTRAP="ssh"
export I_MPI_OFI_PROVIDER="tcp"

TARGET=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" tasks.txt)
TARGET_BASE="${{TARGET%.*}}"

if [ -f "${{TARGET_BASE}}.log" ]; then
    echo "${{TARGET_BASE}}.log already exists, skipping."
    exit 0
fi

~/beyond-rpa/bin/run -np 1 -nt 48 "$TARGET" >& "${{TARGET_BASE}}.log"


