#!/bin/bash
#SBATCH --job-name=feat_short
#SBATCH --account=def-thughes
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/feat_short_%A_%a.out
#SBATCH --error=logs/feat_short_%A_%a.err
set -euo pipefail
module purge 2>/dev/null || true
module load StdEnv/2023 gcc python/3.11 arrow/17.0.0
source ~/splice_env2/bin/activate
cd /scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean
mkdir -p logs
echo "[run] host=$(hostname) task=${SLURM_ARRAY_TASK_ID} start=$(date)"

python - <<'PY'
import sys; sys.path.insert(0, "scripts")
import maxentpy, numpy, pandas, pysam, pyarrow
from maxentpy import maxent
from extract_splicing_features import GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan
from compute_gene_baselines import derive_axes
import splice_extra_features, splice_common
print("[preflight] maxentpy:", maxent.__file__)
print("[preflight] local chain OK")
PY

python3 scripts/featurize_shortlist.py --scripts_dir scripts \
  --shortlist shortlist_crossfit_windows.parquet \
  --reference_fasta train/data/hg38.fa \
  --u1_lookup train/splice_train/data/u1_lookup.npy \
  --me5_lookup train/splice_train/data/me5_lookup.npy \
  --split test --contested_only \
  --shard ${SLURM_ARRAY_TASK_ID} --n_shards 8 \
  --out shortlist_feat

echo "[run] task=${SLURM_ARRAY_TASK_ID} done=$(date)"