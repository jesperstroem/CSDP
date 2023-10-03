#!/bin/bash -l

#SBATCH --partition=small-g  # Partition (queue) name
#SBATCH --ntasks=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-task=1      # Allocate one gpu per MPI rank   # Send email at begin and end of job
#SBATCH --account=project_465000374  # Project for billing

srun singularity exec --mount type=bind,src=/scratch/project_465000374,dst=/users/engholma/mnt /scratch/project_465000374/train.sif python Speciale2023/shared/performance_tests/dataset_details.py
