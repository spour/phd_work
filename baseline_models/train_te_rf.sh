#!/bin/bash
#SBATCH --job-name=te_rf
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/te_rf_%j.out
#SBATCH --error=logs/te_rf_%j.err

set -euo pipefail

TE_DIR="train/data/te_dataset"            # or data/te_dataset_hardneg
FEAT_DIR="train/data/te_features"
RUN="te_rf_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
OUTDIR="./train/te_runs/${RUN}"
mkdir -p logs "$FEAT_DIR" "$OUTDIR" ../train/te_runs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date)] featurize -> $FEAT_DIR"
module load StdEnv/2023 arrow/17.0.0 2>/dev/null || true
source ~/bpnet_env/bin/activate
python scripts/featurize_te.py \
    --te_dir "$TE_DIR" --scripts_dir scripts \
    --reference_fasta ./train/data/hg38.fa \
    --u1_lookup ./train/splice_train/data/u1_lookup.npy --me5_lookup ./train/splice_train/data/me5_lookup.npy \
    --upstream 3000 --downstream 1000 --n_bins 8 \
    --out_dir "$FEAT_DIR"
deactivate || true

echo "[$(date)] RF -> $OUTDIR"
source ~/venv_2025/bin/activate
python scripts/train_rf.py \
    --pos_train "$FEAT_DIR/pos_train.tsv" --neg_train "$FEAT_DIR/neg_train.tsv" \
    --pos_val   "$FEAT_DIR/pos_val.tsv"   --neg_val   "$FEAT_DIR/neg_val.tsv" \
    --pos_test  "$FEAT_DIR/pos_test.tsv"  --neg_test  "$FEAT_DIR/neg_test.tsv" \
    --trees 2000 --min_leaf 4 --criterion log_loss --class_weight balanced \
    --n_jobs "$SLURM_CPUS_PER_TASK" --seed 42 --mode_label te_rf_splicefeats \
    --outdir "$OUTDIR"
echo "[$(date)] done: $OUTDIR"
