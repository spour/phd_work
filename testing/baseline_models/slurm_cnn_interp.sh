#!/bin/bash
#SBATCH --job-name=cnn_interp
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cnn_interp_%j.out
#SBATCH --error=logs/cnn_interp_%j.err

set -euo pipefail
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean
mkdir -p logs
module load arrow/17.0.0 2>/dev/null || true
module load cuda 2>/dev/null || true
source ~/bpnet_env/bin/activate 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"
export CUDA_VISIBLE_DEVICES=0
python3 -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))" || { echo "GPU not usable, exiting"; exit 1; }
echo "=== depth=2 gene-split ==="
python3 scripts/cnn_interp.py \
  --train_npz peakseq_train.npz --train_meta peakseq_train.meta.parquet \
  --test_npz  peakseq_test.npz  --test_meta  peakseq_test.meta.parquet \
  --splice_p1 0.335 --gene_split --depth 2 --epochs 40 --upstream_nt 1000 --out cnn_interp_d2

echo "=== depth=2 standard (motif weights) ==="
python3 scripts/cnn_interp.py \
  --train_npz peakseq_train.npz --train_meta peakseq_train.meta.parquet \
  --test_npz  peakseq_test.npz  --test_meta  peakseq_test.meta.parquet \
  --splice_p1 0.335 --depth 2 --seeds 3 --epochs 40 --upstream_nt 1000 --out cnn_interp_d2