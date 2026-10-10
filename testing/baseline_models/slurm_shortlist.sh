#!/bin/bash
#SBATCH --job-name=shortlist
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/shortlist_%A_%a.out

source ~/venv_2025/bin/activate  2>/dev/null || true   # any env with pandas+pyarrow
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean

mkdir -p logs
python3 scripts/build_shortlist.py \
  --per_gene_dir /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/train/gene_scans/per_gene \
  --canonical_bed train/data/Dominant_CPA_baseline_total.bed \
  --win 500 \
  --shard ${SLURM_ARRAY_TASK_ID} --n_shards 16 \
  --out shortlist