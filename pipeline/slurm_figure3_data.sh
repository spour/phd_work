#!/bin/bash
#SBATCH --job-name=fig3_data
#SBATCH --account=def-thughes
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=fig3_data_%j.out
#SBATCH --error=fig3_data_%j.err
set -euo pipefail
module purge 2>/dev/null || true
module load StdEnv/2023 gcc python/3.11 arrow/17.0.0
source ~/splice_env2/bin/activate
cd "$SLURM_SUBMIT_DIR"

export PYTHONHASHSEED=0

echo "[run] host=$(hostname) start=$(date)"
echo "[run] python=$(which python)  PYTHONHASHSEED=$PYTHONHASHSEED"

python - <<'PY'
import maxentpy, numpy, pandas, pysam, scipy
from maxentpy import maxent
print("[preflight] maxentpy:", maxent.__file__)
PY

SHORTLIST_TSV=shortlist_crossfit_windows.tsv
if [ ! -s "$SHORTLIST_TSV" ]; then
  echo "[prep] converting parquet shortlist -> tsv (arrow loaded)"
  python - <<'PY'
import pandas as pd
d = pd.read_parquet("shortlist_crossfit_windows.parquet")
d.to_csv("shortlist_crossfit_windows.tsv", sep="\t", index=False)
print(f"[prep] wrote shortlist_crossfit_windows.tsv rows={len(d)}")
PY
else
  echo "[prep] reusing existing $SHORTLIST_TSV"
fi

module unload arrow/17.0.0
echo "[run] arrow unloaded; scoring now"

for f in scripts/figure3_data.py "$SHORTLIST_TSV" ../train/data/hg38.fa \
         ../train/splice_train/data/u1_lookup.npy \
         ../train/splice_train/data/me5_lookup.npy; do
  cat "$f" > /dev/null 2>&1 || { echo "[err] cannot read $f -- aborting" >&2; exit 1; }
done
echo "[run] all inputs readable"

python scripts/figure3_data.py --scripts_dir scripts \
  --shortlist "$SHORTLIST_TSV" \
  --reference_fasta ../train/data/hg38.fa \
  --u1_lookup ../train/splice_train/data/u1_lookup.npy \
  --me5_lookup ../train/splice_train/data/me5_lookup.npy \
  --imbalanced --max_per_class 70000 \
  --n_jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --out fig3_new

echo "[run] done=$(date)"
echo "[run] positional output:"
wc -l fig3_new_positional_features.tsv 2>/dev/null || echo "[err] positional table missing"
