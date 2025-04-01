#!/bin/bash
#SBATCH --job-name=Hemholtz_train
#SBATCH --output=Hemholtz_train_%j.out
#SBATCH --error=Hemholtz_train_%j.err
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz886@duke.edu

# Load CUDA module
module load CUDA/12.4

# Verify GPU is available
nvidia-smi

# Run Python script
python main.py -m seed=0 mode=eval experiment=helmholtz_attention_based_conditional_PDERef window=3
