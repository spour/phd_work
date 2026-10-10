#!/bin/bash
# what lowers the model score, by two independent routes
#SBATCH --job-name=cnn_repress
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=cnn_repress_%j.out
#SBATCH --error=cnn_repress_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python cnn_kmer_scan.py
python cnn_evolve.py --direction min --start canonical --suffix _min
echo "[done] $(date -Is)"
