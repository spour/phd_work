#!/bin/bash
#SBATCH --job-name=lr_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/lr_%j.out
#SBATCH --error=logs/lr_%j.err

set -euo pipefail

PY=python
SCRIPT="../scripts/train_lr.py"
PWM_LIST="/scratch/spour98/identify_model/pwm_paths"
DATA_DIR="../train/data/splits"
FEAT_DIR="../train/features"
RUN_NAME="lr_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
OUTDIR="../train/lr_runs/${RUN_NAME}"

mkdir -p ../logs "$OUTDIR" ../train/lr_runs

source ~/venv_2025/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date)] Starting LR training -> $OUTDIR"

srun $PY "$SCRIPT" \
    --pos_train "$DATA_DIR/pos_train.fa" \
    --neg_train "$DATA_DIR/neg_train.fa" \
    --pos_val   "$DATA_DIR/pos_val.fa" \
    --neg_val   "$DATA_DIR/neg_val.fa" \
    --pos_test  "$DATA_DIR/pos_test.fa" \
    --neg_test  "$DATA_DIR/neg_test.fa" \
    --pwms      "$PWM_LIST" \
    --bg        0.3 0.2 0.2 0.3 \
    --outdir    "$OUTDIR" \
    --penalty   l1 \
    --C         0.1 \
    --tol       0.01 \
    --solver    saga \
    --max_iter  2000 \
    --class_weight balanced \
    --seed      42 \
    --n_jobs    $SLURM_CPUS_PER_TASK \
    --mode_label lr_l1_C0.1

echo "[$(date)] Done: $OUTDIR"