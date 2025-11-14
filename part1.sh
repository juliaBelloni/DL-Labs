#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:40:00
#SBATCH --output=slurm_output_exercise1_1bi.out
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
cd assignment2/part1/exercise1.1/
srun python net.py --net_type='Net2' --conv_type='valid'
srun python net.py --net_type='Net2' --conv_type='replicate'
srun python net.py --net_type='Net2' --conv_type='reflect'
srun python net.py --net_type='Net2' --conv_type='circular'
srun python net.py --net_type='Net2' --conv_type='sconv'
srun python net.py --net_type='Net2' --conv_type='fconv'
