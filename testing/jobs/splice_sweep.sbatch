#!/bin/bash
#SBATCH --job-name=splswp
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:15:00
#SBATCH --output=splswp_%j.out
#SBATCH --error=splswp_%j.err
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python splice_model_sweep.py
