#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Exercise1_net2_parallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:40:00
#SBATCH --output=slurm_env_update_%A.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=julia.belloni@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/DL-Labs/
eval "$(conda shell.bash hook)"

conda env update -f assignment2/part3/dl2025_gpu.yml
