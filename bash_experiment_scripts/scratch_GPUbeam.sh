#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 3:00:00                        # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                      # Partition to run in
#SBATCH --gres=gpu:titanx:1                        # Number of gpus
#SBATCH --mem=24g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/nmmr_u_demandnoise_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/nmmr_u_demandnoise_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=beam_ab455

python main.py configs/demand_noise_configs/naivenet_aw_demandnoise.json ate
