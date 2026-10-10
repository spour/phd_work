#!/bin/bash
#SBATCH --job-name=rerank_score
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --output=logs/rerank_%j.out
#SBATCH --error=logs/rerank_%j.err

set -euo pipefail
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean
mkdir -p logs
source ~/bpnet_env/bin/activate    # sklearn 1.8 (HistGBM monotonic_cst); any env with sklearn>=1.4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 scripts/score_rerank.py \
  --train_feat shortlist_feat_train.parquet \
  --test_feat  shortlist_feat.parquet \
  --summary    shortlist_summary.tsv \
  --model gbt --monotone \
  --importance --gene_split_check --capacity_sweep --genomewide  --lean \
  --out rerank