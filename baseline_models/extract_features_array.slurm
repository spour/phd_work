#!/bin/bash
#SBATCH --job-name=feat720
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/feat_%A_%a.out
#SBATCH --error=logs/feat_%A_%a.err

set -euo pipefail

PWM_LIST="/scratch/spour98/identify_model/pwm_paths"
PY=python
SCRIPT="../scripts/generate_bl_features.py"
DATA_DIR="../train/data/splits"
FEAT_DIR="../train/features"

mkdir -p ../logs "$FEAT_DIR"

FASTAS=(
  "$DATA_DIR/pos_train.fa"
  "$DATA_DIR/pos_val.fa"
  "$DATA_DIR/pos_test.fa"
  "$DATA_DIR/neg_train.fa"
  "$DATA_DIR/neg_val.fa"
  "$DATA_DIR/neg_test.fa"
)

OUTS=(
  "$FEAT_DIR/pos_train.tsv"
  "$FEAT_DIR/pos_val.tsv"
  "$FEAT_DIR/pos_test.tsv"
  "$FEAT_DIR/neg_train.tsv"
  "$FEAT_DIR/neg_val.tsv"
  "$FEAT_DIR/neg_test.tsv"
)

source ~/venv_2025/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

FASTA="${FASTAS[$SLURM_ARRAY_TASK_ID]}"
OUT="${OUTS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "$FASTA" ]]; then
    echo "ERROR: input FASTA not found: $FASTA"
    exit 1
fi

rm -f "$OUT"

n_seqs=$(grep -c '^>' "$FASTA" || true)
echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: $FASTA ($n_seqs seqs) -> $OUT"

srun $PY "$SCRIPT" \
  -f "$FASTA" \
  -pwms "$PWM_LIST" \
  -out "$OUT"

echo "[$(date)] Done: $OUT"