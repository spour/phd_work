#!/bin/bash
#SBATCH --job-name=bin_model
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=bin_model_%j.out
#SBATCH --error=bin_model_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python binned_splice_model.py
