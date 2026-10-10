#!/bin/bash
#SBATCH --job-name=spl26
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=spl26_%j.out
#SBATCH --error=spl26_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python splice26_scores.py
echo "[done] $(date -Is)"
