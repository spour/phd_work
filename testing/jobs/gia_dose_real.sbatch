#!/bin/bash
#SBATCH --job-name=gia_real
#SBATCH --account=def-thughes_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=gia_real_%j.out
#SBATCH --error=gia_real_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python gia_dose_real.py
echo "[done] $(date -Is)"
