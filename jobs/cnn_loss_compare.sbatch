#!/bin/bash
# does the cnn's within-gene ranking depend on the pooled-bce training formulation?
#SBATCH --job-name=cnn_loss
#SBATCH --account=def-thughes_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=cnn_loss_%j.out
#SBATCH --error=cnn_loss_%j.err
set -euo pipefail
cd /home/spour98/scratch/remake_model/updated_HumanGeneEnds/reanal/clean/pub
source ~/splice_env2/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for LOSS in bce pairwise listwise; do
  for SEED in 42 43 44; do
    echo "=== $LOSS seed $SEED ==="
    python cnn_loss_compare.py --loss "$LOSS" --seed "$SEED"
  done
done

python cnn_loss_report.py
echo "[done] $(date -Is)"
