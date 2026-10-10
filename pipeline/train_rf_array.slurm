#!/bin/bash
#SBATCH --job-name=rf_array
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/rf_%A_%a.out
#SBATCH --error=logs/rf_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ~/venv_2025/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(fold0 fold1 fold2 fold3 fold4 prod)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
F=splits/$NAME/features
OUT=models/$NAME/rf
mkdir -p "$OUT"

echo "[$(date)] RF $NAME"
python3 scripts/train_rf.py \
  --pos_train "$F/pos_train.tsv" --neg_train "$F/neg_train.tsv" \
  --pos_val   "$F/pos_val.tsv"   --neg_val   "$F/neg_val.tsv" \
  --pos_test  "$F/pos_test.tsv"  --neg_test  "$F/neg_test.tsv" \
  --outdir "$OUT" \
  --trees 30000 --min_split 5 --min_leaf 4 \
  --criterion log_loss --class_weight balanced \
  --seed 42 --n_jobs $SLURM_CPUS_PER_TASK --mode_label logloss_leaf4
echo "[$(date)] RF $NAME done -> $OUT"
