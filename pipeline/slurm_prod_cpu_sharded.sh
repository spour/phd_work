#!/bin/bash
#SBATCH --job-name=prod_cpu
#SBATCH --array=0-19%10
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/prodcpu_%A_%a.out
#SBATCH --error=logs/prodcpu_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
module purge
module load StdEnv/2023 python/3.11 arrow/17.0.0
module load bedtools
source ~/venv_2025/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""          # force CPU: torch.cuda.is_available() -> False

PWM_LIST="${PWM_LIST:-/scratch/spour98/identify_model/pwm_paths}"
NSHARDS=20
SHARD=$SLURM_ARRAY_TASK_ID
GENES=genebeds/genes_prod.bed
OUT=scans/prod
mkdir -p "$OUT"

echo "[$(date)] CPU prod shard $SHARD/$NSHARDS  total=$(wc -l < $GENES)  host=$(hostname)"
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
  --cnn_batch 512 \
  --resume
echo "[$(date)] CPU prod shard $SHARD done"