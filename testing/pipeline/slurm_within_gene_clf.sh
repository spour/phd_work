#!/bin/bash
#SBATCH --job-name=wg_clf
#SBATCH --account=def-thughes
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=wg_clf_%j.out
#SBATCH --error=wg_clf_%j.err
set -euo pipefail
module purge 2>/dev/null || true
module load StdEnv/2023 gcc python/3.11 arrow/17.0.0
source ~/splice_env2/bin/activate
cd "$SLURM_SUBMIT_DIR"
echo "[run] host=$(hostname) start=$(date)"
echo "[run] python=$(which python)"

python - <<'PY'
import maxentpy, numpy, pandas, sklearn, pysam, pyarrow
from maxentpy import maxent
print("[preflight] maxentpy:", maxent.__file__)
print("[preflight] numpy:", numpy.__version__, "| ok")
PY

python scripts/within_gene_classifier.py \
  --shortlist shortlist_crossfit_windows.parquet \
  --reference_fasta ../train/data/hg38.fa \
  --u1_lookup ../train/splice_train/data/u1_lookup.npy \
  --window 2000 --folds 5 --out within_gene_clf

echo "[run] done=$(date)"
ls -la within_gene_clf_*.tsv 2>/dev/null || true