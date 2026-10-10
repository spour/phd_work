#!/bin/bash
#SBATCH --job-name=splchk
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:25:00
#SBATCH --output=splchk_%j.out
#SBATCH --error=splchk_%j.err
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python splice_model_checks.py
