#!/bin/bash
#SBATCH --job-name=splfinal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --output=splfinal_%j.out
#SBATCH --error=splfinal_%j.err
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python splice_model_final.py
