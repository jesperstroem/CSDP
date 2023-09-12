#!/bin/bash -l

#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=25G
#SBATCH --account=project_465000374

srun singularity exec --mount type=bind,src=/scratch/project_465000374,dst=/users/engholma/mnt /scratch/project_465000374/train.sif python Speciale2023/run_usleep.py
