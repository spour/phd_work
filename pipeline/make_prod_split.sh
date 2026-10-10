#!/bin/bash
# (run from pub/) production split = the original train/val/test partition
set -euo pipefail
source ~/bpnet_env/bin/activate
cd "${SLURM_SUBMIT_DIR:-.}"

POS=data/Dominant_CPA_baseline_total.bed          # the ORIGINAL bed (chrY excluded by chrom lists)
DECOY=data/negatives_w_pa_clean.bed
NONDECOY=data/negatives_no_pa_clean.bed

mkdir -p folds splits/prod
printf "chr1\nchr2\nchr4\nchr5\nchr6\nchr7\nchr8\nchr9\nchr10\nchr11\nchr12\nchr14\nchr15\nchr16\nchr17\n" > folds/prod_train_chroms.txt
printf "chr3\nchr13\n" > folds/prod_val_chroms.txt
printf "chr18\nchr19\nchr20\nchr21\nchr22\nchrX\n" > folds/prod_test_chroms.txt

python3 scripts/make_splits.py \
  --pos_bed "$POS" \
  --neg_decoy_bed "$DECOY" --neg_nondecoy_bed "$NONDECOY" \
  --train_chroms folds/prod_train_chroms.txt \
  --val_chroms   folds/prod_val_chroms.txt \
  --test_chroms  folds/prod_test_chroms.txt \
  --outdir splits/prod
echo "[prod] split written to splits/prod/  (matches the original partition)"