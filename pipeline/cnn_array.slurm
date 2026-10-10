#!/bin/bash
#SBATCH --job-name=cnn_array
#SBATCH --array=0-5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/cnn_%A_%a.out
#SBATCH --error=logs/cnn_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ~/venv_2025/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

NAMES=(fold0 fold1 fold2 fold3 fold4 prod)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
DIR=splits/$NAME
OUT=models/$NAME/cnn
mkdir -p "$OUT"

echo "[$(date)] CNN $NAME"; nvidia-smi || true
srun python3 scripts/train_cnn.py \
  --pos_train "$DIR/pos_train.fa" --neg_train "$DIR/neg_train.fa" \
  --pos_val   "$DIR/pos_val.fa"   --neg_val   "$DIR/neg_val.fa" \
  --pos_test  "$DIR/pos_test.fa"  --neg_test  "$DIR/neg_test.fa" \
  --outdir "$OUT" \
  --seq_len 500 --epochs 40 --batch_size 256 --lr 1e-3 --weight_decay 1e-4 \
  --patience 3 --label_smooth 0.1 --pos_weight 1.0 --grad_clip 1.0 \
  --warmup_frac 0.05 --dropout 0.3 \
  --n_filters_1 128 --n_filters_2 256 --n_hidden 128 \
  --num_workers $SLURM_CPUS_PER_TASK --seed 42 --mode_label cnn_v2
echo "[$(date)] CNN $NAME done -> $OUT"
