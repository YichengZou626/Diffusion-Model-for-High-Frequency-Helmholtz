#!/bin/bash
#SBATCH --job-name=data_simulation
#SBATCH --output=data_simulation.out
#SBATCH --error=data_simulation.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=END
#SBATCH --mail-user=yz886@duke.edu
#SBATCH --account=salahshoorlab

# Run the Python script
python data.py
