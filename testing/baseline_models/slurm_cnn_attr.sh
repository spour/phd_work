#!/bin/bash
#SBATCH --job-name=cnn_attr
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cnn_attr_%j.out
#SBATCH --error=logs/cnn_attr_%j.err

set -euo pipefail
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean
mkdir -p logs
source ~/bpnet_env/bin/activate 2>/dev/null || true
module load cuda 2>/dev/null || true

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"

python3 scripts/cnn_attribution.py \
  --train_npz peakseq_train.npz --train_meta peakseq_train.meta.parquet \
  --test_npz  peakseq_test.npz  --test_meta  peakseq_test.meta.parquet \
  --splice_scored rerank_test_scored.parquet \
  --upstream_nt 1000 --downstream_nt 500 --out cnn_attr