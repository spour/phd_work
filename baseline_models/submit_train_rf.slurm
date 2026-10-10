#!/bin/bash
#SBATCH --job-name=rf_train
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/rf_%j.out
#SBATCH --error=logs/rf_%j.err

set -euo pipefail

PY=python
SCRIPT="scripts/train_rf.py"
FEAT_DIR="${FEAT_DIR:-train/features}"
RUN_TAG="${RUN_TAG:-rf_run}"
OUTDIR="train/rf_runs/${RUN_TAG}"

mkdir -p logs "$OUTDIR" train/rf_runs
source ~/venv_2025/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date)] RF training"
echo "  FEAT_DIR = $FEAT_DIR"
echo "  OUTDIR   = $OUTDIR"
NCOL=$(head -1 "$FEAT_DIR/pos_train.tsv" | awk -F'\t' '{print NF-1}')
echo "  feature columns in $FEAT_DIR/pos_train.tsv = $NCOL"

srun $PY "$SCRIPT" \
    --pos_train "$FEAT_DIR/pos_train.tsv" \
    --neg_train "$FEAT_DIR/neg_train.tsv" \
    --pos_val   "$FEAT_DIR/pos_val.tsv" \
    --neg_val   "$FEAT_DIR/neg_val.tsv" \
    --pos_test  "$FEAT_DIR/pos_test.tsv" \
    --neg_test  "$FEAT_DIR/neg_test.tsv" \
    --outdir    "$OUTDIR" \
    --trees     30000 \
    --min_split 5 \
    --min_leaf  4 \
    --criterion log_loss \
    --class_weight balanced \
    --seed      42 \
    --n_jobs    $SLURM_CPUS_PER_TASK \
    --mode_label logloss_leaf4

echo "[$(date)] Done: $OUTDIR"

