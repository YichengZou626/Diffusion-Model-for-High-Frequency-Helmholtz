#!/bin/bash
#SBATCH --job-name=Hemholtz_train
#SBATCH --output=Hemholtz_train_%j.out
#SBATCH --error=Hemholtz_train_%j.err
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --mem=100G  # Increase memory if possible
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz886@duke.edu
#SBATCH --account=salahshoorlab

# Load CUDA module
module load CUDA/12.4

# Verify GPU is available
nvidia-smi

# Run Python script
python main.py -m seed=0 mode=train experiment=helmholtz_conditional_SDA  window=3
