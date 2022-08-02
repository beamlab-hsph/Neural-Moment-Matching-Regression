#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 1-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                        # Partition to run in
#SBATCH --gres=gpu:1                        # Number of gpus
#SBATCH --mem=24g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/dfpv_dsprite_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/dfpv_dsprite_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=beam_ab455

python main.py configs/dfpv.json ate
