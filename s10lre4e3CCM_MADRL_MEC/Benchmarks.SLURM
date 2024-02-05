#!/bin/sh
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --job-name=Benchmarks
#SBATCH --time 60:00:00
source activate vCMaDRLMEC
python Benchmarks_run.py $SLURM_ARRAY_TASK_ID