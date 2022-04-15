#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 2-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                        # Partition to run in
#SBATCH --gres gpu:a100:1                        # Number of gpus
#SBATCH --mem=24g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/Demand_GPU_NMMR.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/Demand_GPU_NMMR.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=beam_ab455

echo "Running NMMR U"
python main.py configs/nmmr_u.json ate

echo "Running NMMR V"
python main.py configs/nmmr_v.json ate
