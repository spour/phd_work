#!/bin/bash
#SBATCH --job-name=cnn_v2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/cnn_v2_%j.out
#SBATCH --error=logs/cnn_v2_%j.err

set -euo pipefail

PY=python
SCRIPT="../scripts/train_cnn.py"
DATA_DIR="../train/data/splits"
RUN_NAME="cnnv2_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
OUTDIR="../train/cnn_runs/${RUN_NAME}"

mkdir -p ../logs "$OUTDIR" ../train/cnn_runs

source ~/venv_2025/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date)] Starting CNN v2 training -> $OUTDIR"
nvidia-smi || true

srun $PY "$SCRIPT" \
    --pos_train     "$DATA_DIR/pos_train.fa" \
    --neg_train     "$DATA_DIR/neg_train.fa" \
    --pos_val       "$DATA_DIR/pos_val.fa" \
    --neg_val       "$DATA_DIR/neg_val.fa" \
    --pos_test      "$DATA_DIR/pos_test.fa" \
    --neg_test      "$DATA_DIR/neg_test.fa" \
    --outdir        "$OUTDIR" \
    --seq_len       500 \
    --epochs        40 \
    --batch_size    256 \
    --lr            1e-3 \
    --weight_decay  1e-4 \
    --patience      3 \
    --label_smooth  0.1 \
    --pos_weight    1.0 \
    --grad_clip     1.0 \
    --warmup_frac   0.05 \
    --dropout       0.3 \
    --n_filters_1   128 \
    --n_filters_2   256 \
    --n_hidden      128 \
    --num_workers   ${SLURM_CPUS_PER_TASK} \
    --seed          42 \
    --mode_label    cnn_v2

echo "[$(date)] Done: $OUTDIR"