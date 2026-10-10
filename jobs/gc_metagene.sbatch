#!/bin/bash
#SBATCH --job-name=gc_meta
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=gc_meta_%j.out
#SBATCH --error=gc_meta_%j.err
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python gc_metagene_data.py
