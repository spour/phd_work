#!/bin/bash
# out-of-fold cnn scores over all 70,723 contested-gene candidates
#SBATCH --job-name=cnn_wg
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=cnn_wg_%j.out
#SBATCH --error=cnn_wg_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python cnn_wg_loco.py
echo "[done] $(date -Is)"
