#!/bin/bash
# the cpu half of the loco comparison
#SBATCH --job-name=loco_fig
#SBATCH --account=def-thughes_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=loco_fig_%j.out
#SBATCH --error=loco_fig_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
test -s cnn_loco_scores.npz
python loco_figure_data.py
echo "[done] $(date -Is)"
