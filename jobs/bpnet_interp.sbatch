#!/bin/bash
# the interpretability suite on the bpnet variant checkpoint
#SBATCH --job-name=bpnet_int
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=bpnet_int_%j.out
#SBATCH --error=bpnet_int_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python bpnet_interp.py --stages 1,2,3
echo "[done] $(date -Is)"
