#!/bin/bash
#SBATCH --job-name=score_cpu
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/score_%A_%a.out
#SBATCH --error=logs/score_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load arrow/17.0.0 StdEnv/2023 2>/dev/null || true
module load bedtools
source ~/venv_2025/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""          # force CPU: torch.cuda.is_available() -> False

PWM_LIST="${PWM_LIST:-/scratch/spour98/identify_model/pwm_paths}"
NAMES=(fold0 fold1 fold2 fold3 fold4 prod)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
GENES=genebeds/genes_${NAME}.bed
OUT=scans/$NAME
mkdir -p "$OUT"

echo "[$(date)] CPU score $NAME  genes=$(wc -l < $GENES)  host=$(hostname)"
python3 scripts/score_genes.py \
  --genes_bed "$GENES" \
  --cpa_bed data/Dominant_CPA_baseline_total.bed --cpa_bed_is_window \
  --genome ../train/data/hg38.fa \
  --pwms "$PWM_LIST" --bg 0.3 0.2 0.2 0.3 \
  --rf_dir  models/$NAME/rf \
  --lr_dir  models/$NAME/lr \
  --cnn_dir models/$NAME/cnn \
  --outdir "$OUT" \
  --cnn_batch 512 \
  --resume
echo "[$(date)] CPU score $NAME done -> $OUT/per_gene/"
