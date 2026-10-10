#!/bin/bash
#SBATCH --job-name=gia_don
#SBATCH --account=def-thughes_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=gia_don_%j.out
#SBATCH --error=gia_don_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python gia_donor_panel.py
echo "[done] $(date -Is)"
