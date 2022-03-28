#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 2-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                        # Partition to run in
#SBATCH --gres gpu:a100:1                        # Number of gpus
#SBATCH --mem=10g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/Demand_GPU.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/Demand_GPU.err                 # File to which STDERR will be written, including job ID (%j)

echo "Running CEVAE"
python main.py configs/cevae.json ate

echo "Running DFPV"
python main.py configs/dfpv.json ate

echo "Running KPV"
python main.py configs/kpv.json ate

echo "Running naive linear regression"
python main.py configs/linear_regression.json ate

echo "Running naive neural net"
python main.py configs/naive_nn_demand.json ate

echo "Running NMMR"
python main.py configs/NMMR.json ate

echo "Running PMMR"
python main.py configs/pmmr.json ate
