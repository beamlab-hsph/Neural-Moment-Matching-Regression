#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 1-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                       # Partition to run in
#SBATCH --gres gpu:a100:1                       # Number of gpus
#SBATCH --mem=40g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/kpv_demand_GPU.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/kpv_demand_GPU.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=beam_ab455

module load gcc/9.2.0
module load cuda/11.2

python main.py configs/kpv.json ate
