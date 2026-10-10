#!/bin/bash
#SBATCH --job-name=wg_len
#SBATCH --account=def-thughes_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=wg_len_%j.out
#SBATCH --error=wg_len_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python wg_by_length.py
echo "[done] $(date -Is)"
