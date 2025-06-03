#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --output=dataset.log
#SBATCH --error=dataset.out
#SBATCH --time=72:00:00
#SBATCH --begin=now
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=370G
#SBATCH --nodes=1

srun --ntasks=16 python dataset.py -opt yml/dataset.yml
