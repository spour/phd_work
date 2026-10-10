#!/bin/bash
# one bpnet-style retrain (l1 swept over three values
#SBATCH --job-name=bpnet
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=bpnet_%j.out
#SBATCH --error=bpnet_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python bpnet_variant.py --l1 1e-4,1e-3,1e-2
echo "[done] $(date -Is)"
