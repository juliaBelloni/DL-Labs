#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Part3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:07:00
#SBATCH --output=slurm_ass_3_part1_%A.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=julia.belloni@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/DL-Labs/

# activate env
eval "$(conda shell.bash hook)"
conda activate dl2025

#run code
cd assignment3/part1
# srun python train_pl.py
srun python train_pl.py --z_dim=2 
