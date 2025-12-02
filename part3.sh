#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:30:00
#SBATCH --output=slurm_output_part3_%A.out
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
cd assignment2/part3
srun python train.py --model gcn
srun python train.py --model matrix-gcn
srun python train.py --model gat
