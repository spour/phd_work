#!/bin/bash
#SBATCH --job-name=gia_quick
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=gia_quick_%j.out
#SBATCH --error=gia_quick_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
python gia_dose_matched.py --gia_bg 1623 --suffix _quick
echo "[done] $(date -Is)"
