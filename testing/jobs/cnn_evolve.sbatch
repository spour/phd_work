#!/bin/bash
# in-silico evolution against the reported cnn
#SBATCH --job-name=cnn_evolve
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=cnn_evolve_%j.out
#SBATCH --error=cnn_evolve_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python cnn_evolve.py
echo "[done] $(date -Is)"
