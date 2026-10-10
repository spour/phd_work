#!/bin/bash
#SBATCH --job-name=lr_array
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/lr_%A_%a.out
#SBATCH --error=logs/lr_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ~/venv_2025/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(fold0 fold1 fold2 fold3 fold4 prod)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
F=splits/$NAME/features
OUT=models/$NAME/lr
mkdir -p "$OUT"

echo "[$(date)] LR $NAME"
python3 scripts/train_lr.py \
  --pos_train "$F/pos_train.tsv" --neg_train "$F/neg_train.tsv" \
  --pos_val   "$F/pos_val.tsv"   --neg_val   "$F/neg_val.tsv" \
  --pos_test  "$F/pos_test.tsv"  --neg_test  "$F/neg_test.tsv" \
  --outdir "$OUT" \
  --penalty l1 --C 0.1 --tol 0.01 --solver saga --max_iter 2000 \
  --class_weight balanced --seed 42 --n_jobs $SLURM_CPUS_PER_TASK \
  --mode_label lr_l1_C0.1
echo "[$(date)] LR $NAME done -> $OUT"
