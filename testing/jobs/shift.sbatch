#!/bin/bash
#SBATCH --job-name=shift
#SBATCH --account=def-thughes_gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=shift_%j.out
#SBATCH --error=shift_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python shortlist_shift.py
echo "[done] $(date -Is)"
