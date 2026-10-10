#!/bin/bash
#SBATCH --job-name=cascade
#SBATCH --account=def-thughes_gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=cascade_%j.out
#SBATCH --error=cascade_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python cascade_test.py
echo "[done] $(date -Is)"
