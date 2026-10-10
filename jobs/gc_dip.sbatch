#!/bin/bash
#SBATCH --job-name=gc_dip
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=gc_dip_%j.out
#SBATCH --error=gc_dip_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python gc_dip_depth.py
