#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 1-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_beam                        # Partition to run in
#SBATCH --gres=gpu:a100:1                        # Number of gpus
#SBATCH --mem=24g                       # Memory total in MiB (for all cores)
#SBATCH -o slurm_logs/dsprite_5k_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e slurm_logs/dsprite_5k_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --account=beam_ab455

#echo "CEVAE"
#python main.py configs/cevae.json ate

#echo "DFPV"
#python main.py configs/dfpv.json ate

#echo "KPV"
#python main.py configs/kpv.json ate

#echo "PMMR"
#python main.py configs/pmmr.json ate

echo "NMMR"
python main.py configs/test.json ate