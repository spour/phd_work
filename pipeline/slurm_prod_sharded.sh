#!/bin/bash
#SBATCH --job-name=score_prod
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prod_%A_%a.out
#SBATCH --error=logs/prod_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
module purge
module load arrow/17.0.0 StdEnv/2023 2>/dev/null || true
module load bedtools
source ~/venv_2025/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PWM_LIST="${PWM_LIST:-/scratch/spour98/identify_model/pwm_paths}"
NSHARDS=10
SHARD=$SLURM_ARRAY_TASK_ID
GENES=genebeds/genes_prod.bed
OUT=scans/prod
mkdir -p "$OUT"

echo "[$(date)] prod shard $SHARD/$NSHARDS  total_genes=$(wc -l < $GENES)  host=$(hostname)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
python3 scripts/score_genes.py \
  --genes_bed "$GENES" \
  --cpa_bed data/Dominant_CPA_baseline_total.bed --cpa_bed_is_window \
  --genome ../train/data/hg38.fa \
  --pwms "$PWM_LIST" --bg 0.3 0.2 0.2 0.3 \
  --rf_dir  models/prod/rf \
  --lr_dir  models/prod/lr \
  --cnn_dir models/prod/cnn \
  --outdir "$OUT" \
  --shard_index "$SHARD" --n_shards "$NSHARDS" \
  --cnn_batch 256 \
  --resume
echo "[$(date)] prod shard $SHARD done"