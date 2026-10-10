#!/bin/bash
#SBATCH --job-name=ss_scramble
#SBATCH --account=def-thughes
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/ss_scramble_%j.out
#SBATCH --error=logs/ss_scramble_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub/thesis_ch3
source ~/splice_env2/bin/activate
python py/splice_scramble.py scan 16
python py/splice_scramble.py score
echo "[done] $(date -Is)"
