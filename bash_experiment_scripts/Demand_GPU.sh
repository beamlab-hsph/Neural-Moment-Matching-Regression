#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 2-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                        # Partition to run in
#SBATCH --gres gpu:a100:1                        # Number of gpus
#SBATCH --mem=24g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/Demand_GPU.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/Demand_GPU.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=beam_ab455

echo "Running naive linear regression Y ~ A + W + Z"
python main.py configs/linear_regression_AWZY.json ate

echo "Running naive linear regression Y ~ A"
python main.py configs/linear_regression_AY.json ate

echo "Running naive neural net Y ~ A"
python main.py configs/naive_nn_AY_demand.json ate

echo "Running naive neural net Y ~ A + W + Z"
python main.py configs/naive_nn_AWZY_demand.json ate

echo "Running NMMR"
python main.py configs/NMMR.json ate

echo "Running PMMR"
python main.py configs/pmmr.json ate
