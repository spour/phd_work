#!/bin/bash
# (run from pub/) filter pc_genes.bed into per-fold gene beds by each fold's test
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

GENES=../train/data/pc_genes.bed
mkdir -p genebeds

for k in 0 1 2 3 4; do
  chroms="folds/test_chroms_f${k}.txt"
  [ -s "$chroms" ] || { echo "[err] $chroms missing (run run_make_splits.sh)"; exit 1; }
  awk 'NR==FNR{keep[$1]=1; next} ($1 in keep)' "$chroms" "$GENES" > "genebeds/genes_fold${k}.bed"
  echo "fold$k: $(wc -l < genebeds/genes_fold${k}.bed) genes  ($(tr '\n' ' ' < $chroms))"
done

cp "$GENES" genebeds/genes_prod.bed
echo "prod : $(wc -l < genebeds/genes_prod.bed) genes (all)"

nf=$(cat genebeds/genes_fold[0-4].bed | sort -u | wc -l)
echo "[check] union of fold gene beds = $nf genes (should cover all scored chromosomes once)"