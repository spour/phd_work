#!/bin/bash
# (run from pub/) strand-aware fasta extraction for all 5 cross-fit folds + the production
set -euo pipefail
module load bedtools
cd "${SLURM_SUBMIT_DIR:-.}"

GENOME=../train/data/hg38.fa
[ -f "$GENOME" ] || { echo "[err] $GENOME not found (symlink data/ in?)"; exit 1; }

for NAME in fold0 fold1 fold2 fold3 fold4 prod; do
  DIR=splits/$NAME
  [ -d "$DIR" ] || { echo "[skip] $DIR missing"; continue; }
  for split in train val test; do
    for cls in pos neg; do
      bed="$DIR/${cls}_${split}.bed"; fa="$DIR/${cls}_${split}.fa"
      bedtools getfasta -s -name -fi "$GENOME" -bed "$bed" -fo "$fa"
    done
  done
  echo "[fa] $NAME done"
done
echo "[done] FASTAs extracted for all splits"