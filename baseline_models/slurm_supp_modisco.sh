#!/bin/bash
#SBATCH --job-name=supp_modisco
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/supp_modisco_%j.out
#SBATCH --error=logs/supp_modisco_%j.err

set -euo pipefail
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean
mkdir -p logs
module load arrow/17.0.0 2>/dev/null || true
module load cuda 2>/dev/null || true
source ~/bpnet_env/bin/activate 2>/dev/null || true
python3 -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))" \
  || { echo "GPU not usable, exiting"; exit 1; }

echo "=== MATCHED (internal-used vs internal-cryptic) ==="

echo "=== UNMATCHED (confound baseline) ==="

python3 scripts/supp_cnn_modisco.py \
  --train_npz suppseq_unmatched_train.npz --train_meta suppseq_unmatched_train.meta.parquet \
  --test_npz suppseq_unmatched_test.npz --test_meta suppseq_unmatched_test.meta.parquet \
  --upstream_nt 1000 --epochs 30 --attr_class both --out supp_modisco_unmatched
