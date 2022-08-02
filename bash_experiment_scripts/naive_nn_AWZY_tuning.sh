#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 1-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                         # Partition to run in
#SBATCH --gres=gpu:a100:1                        # Number of gpus
#SBATCH --mem=20g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/naive_nn_awzy_tune_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/naive_nn_awzy_tune_%j.err                 # File to which STDERR will be written, including job ID (%j)

echo "Tuning naive NN Y ~ A + W + Z"
python main.py configs/naive_nn_AWZY_demand.json ate
