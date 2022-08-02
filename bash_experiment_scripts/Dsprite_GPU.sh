#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 7-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                        # Partition to run in
#SBATCH --gres gpu:a100:1                        # Number of gpus
#SBATCH --mem=24g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/Dsprite_GPU_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/Dsprite_GPU_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=beam_ab455

#echo "Running naive linear regression Y ~ A + W + Z"
#python main.py configs/linear_regression_AWZY.json ate
#
#echo "Running naive linear regression Y ~ A"
#python main.py configs/linear_regression_AY.json ate
#
#echo "Running naive neural net Y ~ A"
#python main.py configs/naive_nn_AY_demand.json ate
#
#echo "Running naive neural net Y ~ A + W + Z"
#python main.py configs/naive_nn_AWZY_demand.json ate
#
#echo "Running NMMR"
#python main.py configs/NMMR.json ate
#
#echo "Running PMMR"
#python main.py configs/pmmr.json ate

python main.py configs/figure3_config/naivenet_aw_figure3.json ate