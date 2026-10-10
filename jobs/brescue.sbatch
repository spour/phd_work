#!/bin/bash
#SBATCH --job-name=brescue
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=brescue_%j.out
#SBATCH --error=brescue_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python baseline_rescue.py
echo "[done] $(date -Is)"
