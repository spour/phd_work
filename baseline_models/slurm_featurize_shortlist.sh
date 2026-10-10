#!/bin/bash
#SBATCH --job-name=feat_short
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/feat_short_%A_%a.out

source ~/bpnet_env/bin/activate
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean
mkdir -p logs

python3 scripts/featurize_shortlist.py --scripts_dir scripts \
  --shortlist shortlist_windows.parquet \
  --reference_fasta train/data/hg38.fa \
  --u1_lookup train/splice_train/data/u1_lookup.npy \
  --me5_lookup train/splice_train/data/me5_lookup.npy \
  --split test --contested_only \
  --shard ${SLURM_ARRAY_TASK_ID} --n_shards 8 \
  --out shortlist_feat