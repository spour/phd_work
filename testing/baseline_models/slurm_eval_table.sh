#!/bin/bash
#SBATCH --job-name=eval_tbl
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_tbl_%j.out
#SBATCH --error=logs/eval_tbl_%j.err
set -euo pipefail
source ~/bpnet_env/bin/activate          # needs Bio (featurization) + MaxEntScan (splice)
SCRIPTS=scripts
mkdir -p data logs
srun python3 $SCRIPTS/build_eval_table.py \
  --scripts_dir $SCRIPTS \
  --per_gene_dir /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/train/gene_scans/per_gene \
  --pc_genes_bed train/data/pc_genes.bed \
  --reference_fasta train/data/hg38.fa \
  --pwms /scratch/spour98/identify_model/pwm_paths \
  --u1_lookup train/splice_train/data/u1_lookup.npy --me5_lookup train/splice_train/data/me5_lookup.npy \
  --win500 500 --upstream_nt 1000 --downstream_nt 500 --n_bins 8 --center_offset 249 \
  --max_peaks 200 --min_peak_distance 250 --tolerance_nt 250 \
  --test_chroms chr18,chr19,chr20,chr21,chr22,chrX \
  --out data/eval_table_test.parquet
echo "[$(date)] eval table done"