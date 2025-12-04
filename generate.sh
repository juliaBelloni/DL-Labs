#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=genereate_gpt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:20:00
#SBATCH --output=slurm_generate_new_%j.out
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
cd assignment2/part2
srun python generate.py --model_weights_folder ./logs/gpt-mini/version_0/checkpoints --prompt "Once upon a time there" --num_generated_tokens 200 --num_samples 5 --top_p 0.4


srun python generate.py --model_weights_folder ./logs/gpt-mini/version_0/checkpoints --prompt "A Multilayer Perceptron is a universal" --num_generated_tokens 200 --num_samples 5 --top_p 0.4