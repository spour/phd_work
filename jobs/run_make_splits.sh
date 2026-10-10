#!/bin/bash
set -euo pipefail
# per-fold splits: test=fold k (scoring target), val=next fold, train=rest

cat folds/fold_0_chroms.txt > folds/test_chroms_f0.txt
cat folds/fold_1_chroms.txt > folds/val_chroms_f0.txt
cat folds/fold_2_chroms.txt folds/fold_3_chroms.txt folds/fold_4_chroms.txt > folds/train_chroms_f0.txt
python3 scripts/make_splits.py --pos_bed data/Dominant_CPA_baseline_total.noY.bed --neg_decoy_bed data/negatives_w_pa_clean.bed --neg_nondecoy_bed data/negatives_no_pa_clean.bed --train_chroms folds/train_chroms_f0.txt --val_chroms folds/val_chroms_f0.txt --test_chroms folds/test_chroms_f0.txt --outdir splits/fold0

cat folds/fold_1_chroms.txt > folds/test_chroms_f1.txt
cat folds/fold_2_chroms.txt > folds/val_chroms_f1.txt
cat folds/fold_0_chroms.txt folds/fold_3_chroms.txt folds/fold_4_chroms.txt > folds/train_chroms_f1.txt
python3 scripts/make_splits.py --pos_bed data/Dominant_CPA_baseline_total.noY.bed --neg_decoy_bed data/negatives_w_pa_clean.bed --neg_nondecoy_bed data/negatives_no_pa_clean.bed --train_chroms folds/train_chroms_f1.txt --val_chroms folds/val_chroms_f1.txt --test_chroms folds/test_chroms_f1.txt --outdir splits/fold1

cat folds/fold_2_chroms.txt > folds/test_chroms_f2.txt
cat folds/fold_3_chroms.txt > folds/val_chroms_f2.txt
cat folds/fold_0_chroms.txt folds/fold_1_chroms.txt folds/fold_4_chroms.txt > folds/train_chroms_f2.txt
python3 scripts/make_splits.py --pos_bed data/Dominant_CPA_baseline_total.noY.bed --neg_decoy_bed data/negatives_w_pa_clean.bed --neg_nondecoy_bed data/negatives_no_pa_clean.bed --train_chroms folds/train_chroms_f2.txt --val_chroms folds/val_chroms_f2.txt --test_chroms folds/test_chroms_f2.txt --outdir splits/fold2

cat folds/fold_3_chroms.txt > folds/test_chroms_f3.txt
cat folds/fold_4_chroms.txt > folds/val_chroms_f3.txt
cat folds/fold_0_chroms.txt folds/fold_1_chroms.txt folds/fold_2_chroms.txt > folds/train_chroms_f3.txt
python3 scripts/make_splits.py --pos_bed data/Dominant_CPA_baseline_total.noY.bed --neg_decoy_bed data/negatives_w_pa_clean.bed --neg_nondecoy_bed data/negatives_no_pa_clean.bed --train_chroms folds/train_chroms_f3.txt --val_chroms folds/val_chroms_f3.txt --test_chroms folds/test_chroms_f3.txt --outdir splits/fold3

cat folds/fold_4_chroms.txt > folds/test_chroms_f4.txt
cat folds/fold_0_chroms.txt > folds/val_chroms_f4.txt
cat folds/fold_1_chroms.txt folds/fold_2_chroms.txt folds/fold_3_chroms.txt > folds/train_chroms_f4.txt
python3 scripts/make_splits.py --pos_bed data/Dominant_CPA_baseline_total.noY.bed --neg_decoy_bed data/negatives_w_pa_clean.bed --neg_nondecoy_bed data/negatives_no_pa_clean.bed --train_chroms folds/train_chroms_f4.txt --val_chroms folds/val_chroms_f4.txt --test_chroms folds/test_chroms_f4.txt --outdir splits/fold4
