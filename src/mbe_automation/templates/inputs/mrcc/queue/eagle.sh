#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -A pl0415-03
#SBATCH -p altair 
#SBATCH --nodes 1
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=180gb

module load python
module load ifort
module load impi
module load mkl

export PATH=$PATH:~/mrcc

TARGET=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" tasks.txt)
cd $(dirname $TARGET)

if [ -f dmrcc.log ]; then
    echo "dmrcc.log already exists in $(pwd), skipping."
    exit 0
fi

dmrcc >& dmrcc.log
find . -maxdepth 1 -type f ! -name 'MINP' ! -name 'dmrcc.log' -delete

