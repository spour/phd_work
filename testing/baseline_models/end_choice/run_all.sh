#!/bin/bash
#SBATCH --job-name=end_choice
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/end_choice_%j.out
set -euo pipefail
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean
mkdir -p logs train/data/end_choice

D=train/data
S=scripts
OUT=$D/end_choice

source ~/venv_2025/bin/activate
python $S/end_choice/01_build_choice_dataset.py \
  --atlas $D/atlas.chr.canonical.bed --floor 0.05 \
  --cpccpa_intragenic $D/cpccpa_3model_intragenic.bed \
  --pc_genes          $D/pc_genes.bed \
  --rf_peak_table     $D/peak_table_8axes_rank.parquet \
  --out $OUT/choice_sites.parquet

source ~/bpnet_env/bin/activate
python $S/end_choice/02_featurize_local.py \
  --choice_sites $OUT/choice_sites.parquet \
  --scripts_dir  $S \
  --reference_fasta $D/hg38.fa \
  --u1_lookup train/splice_train/data/u1_lookup.npy \
  --me5_lookup train/splice_train/data/me5_lookup.npy \
  --up 3000 --dn 1000 \
  --out $OUT/choice_features.parquet

source ~/venv_2025/bin/activate
python $S/end_choice/03_train_eval.py \
  --features $OUT/choice_features.parquet \
  --train_chroms $D/train_chroms.txt \
  --test_chroms  $D/test_chroms.txt \
  --out_json $OUT/results.json

echo "[ALL DONE] results in $OUT/results.json"