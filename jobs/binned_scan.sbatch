#!/bin/bash
#SBATCH --job-name=bin_scan
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=bin_scan_%j.out
#SBATCH --error=bin_scan_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python binned_profile_scan.py
