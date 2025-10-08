#!/bin/bash
#SBATCH --job-name=Hemholtz_train
#SBATCH --output=Hemholtz_train_%j.out
#SBATCH --error=Hemholtz_train_%j.err
#SBATCH --partition=h200ea
#SBATCH --gres=gpu:h200:1 
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz886@duke.edu
#SBATCH --account=h200ea

# Load modules and Conda
module load CUDA/12.4
source /hpc/group/salahshoorlab/yz886/miniconda3/etc/profile.d/conda.sh
conda activate pdediff

# ✅ Set WANDB to offline BEFORE Python starts
export WANDB_MODE=offline

# Verify GPU
nvidia-smi

python 1D.py seed=0 mode=train experiment=1D window=3
