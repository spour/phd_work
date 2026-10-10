#!/bin/bash
#SBATCH --job-name=gia_ctrl
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=gia_ctrl_%j.out
#SBATCH --error=gia_ctrl_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
python gia_dose_controls.py
echo "[done] $(date -Is)"
