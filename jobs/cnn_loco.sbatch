#!/bin/bash
# 23-fold leave-one-chromosome-out retrain of the sequence cnn
#SBATCH --job-name=cnn_loco
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=cnn_loco_%j.out
#SBATCH --error=cnn_loco_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python cnn_loco_train.py --ksh 3

python loco_figure_data.py

echo "[done] $(date -Is)"
