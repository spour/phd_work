#!/bin/bash
# (run from pub/) build the leak-free per-gene store by symlinking each fold's per-gene
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DEST=scans/crossfit/per_gene
mkdir -p "$DEST"

n_link=0; n_dup=0
for k in 0 1 2 3 4; do
  src="scans/fold${k}/per_gene"
  [ -d "$src" ] || { echo "[err] $src missing (scoring not done?)"; exit 1; }
  for f in "$src"/*.parquet; do
    g=$(basename "$f")
    if [ -e "$DEST/$g" ]; then
      echo "[dup] $g already present (fold overlap!)"; n_dup=$((n_dup+1)); continue
    fi
    ln -s "$(readlink -f "$f")" "$DEST/$g"; n_link=$((n_link+1))
  done
done

n_have=$(ls "$DEST" | wc -l)
n_expect=$(cat genebeds/genes_fold[0-4].bed | cut -f4 | sort -u | wc -l)
echo "[merge] linked $n_link parquets, $n_dup duplicates"
echo "[merge] store has $n_have genes; expected $n_expect from fold gene beds"
if [ "$n_dup" -ne 0 ]; then
  echo "[FAIL] duplicates mean a gene was scored by >1 fold (chromosome in 2 folds). Investigate."; exit 1
fi
echo "[ok] clean cross-fit store at $DEST"
echo "[next] build_shortlist.py --per_gene_dir $DEST ..."