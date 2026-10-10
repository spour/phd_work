#!/bin/bash
#SBATCH --job-name=splscore
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=splscore_%j.out
#SBATCH --error=splscore_%j.err
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python splice_score_redundancy.py 24
