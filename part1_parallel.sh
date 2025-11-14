#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=Exercise1_net2_parallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:40:00
#SBATCH --output=slurm_output_exercise1_%A_%a.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=julia.belloni@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/DL-Labs/
eval "$(conda shell.bash hook)"
conda activate dl2025

cd assignment2/part1/exercise1.1/

# Run all six experiments in parallel (background processes)
python net.py --net_type='Net2' --conv_type='valid'     &
python net.py --net_type='Net2' --conv_type='replicate' &
python net.py --net_type='Net2' --conv_type='reflect'   &
python net.py --net_type='Net2' --conv_type='circular'  &
python net.py --net_type='Net2' --conv_type='sconv'     &
python net.py --net_type='Net2' --conv_type='fconv'     &

# Wait for all background processes to finish
wait
