#!/bin/bash
#SBATCH --job-name=spl_splits
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/spl_splits_%j.out
#SBATCH --error=logs/spl_splits_%j.err
set -euo pipefail
source ~/bpnet_env/bin/activate          # splice features need pysam/MaxEntScan/ViennaRNA
SCRIPTS=scripts
SPLITS=train/data/splits
OUT=train/features_splice
mkdir -p "$OUT" logs
ID_MODE=coord
for split in pos_train neg_train pos_val neg_val pos_test neg_test; do
  echo "[$(date)] $split"
  srun python3 $SCRIPTS/splice_features_for_bed.py \
    --scripts_dir $SCRIPTS \
    --bed $SPLITS/${split}.bed \
    --reference_fasta train/data/hg38.fa \
    --u1_lookup train/splice_train/data/u1_lookup.npy --me5_lookup train/splice_train/data/me5_lookup.npy \
    --upstream_nt 1000 --downstream_nt 500 --n_bins 8 \
    --id_mode $ID_MODE \
    --out $OUT/${split}.spl.tsv
done
echo "[$(date)] done splice features for all splits"