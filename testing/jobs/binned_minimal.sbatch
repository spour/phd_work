#!/bin/bash
#SBATCH --job-name=bin_min
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=bin_min_%j.out
#SBATCH --error=bin_min_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python binned_minimal_model.py
