#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 2-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                        # Partition to run in
#SBATCH --gres gpu:a100:1                        # Number of gpus
#SBATCH --mem=24g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/Dsprite_GPU.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/Dsprite_GPU.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=beam_ab455

echo "Running NMMR dsprite"
python main.py configs/nmmr_dsprite.json ate