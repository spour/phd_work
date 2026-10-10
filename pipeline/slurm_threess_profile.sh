#!/bin/bash
#SBATCH --job-name=threess_prof
#SBATCH --account=def-thughes
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=threess_prof_%j.out
#SBATCH --error=threess_prof_%j.err
set -euo pipefail
module purge 2>/dev/null || true
module load StdEnv/2023 gcc python/3.11 arrow/17.0.0
source ~/splice_env2/bin/activate
cd "$SLURM_SUBMIT_DIR"
echo "[run] host=$(hostname) start=$(date)"
echo "[run] python=$(which python)"

python - <<'PY'
import maxentpy, numpy, pandas, sklearn, pysam, pyarrow, scipy, matplotlib
from maxentpy import maxent
print("[preflight] maxentpy:", maxent.__file__)
PY

for f in scripts/threess_distance_profile.py \
         shortlist_crossfit_windows.parquet ../train/data/hg38.fa \
         ../train/data/Dominant_CPA_baseline_total.bed; do
  cat "$f" > /dev/null 2>&1 || { echo "[err] cannot read $f -- aborting" >&2; exit 1; }
done
echo "[run] all inputs readable"

python scripts/threess_distance_profile.py \
  --shortlist shortlist_crossfit_windows.parquet \
  --cpa_bed ../train/data/Dominant_CPA_baseline_total.bed \
  --genes_bed ../train/data/pc_genes.bed --random_n 1000 \
  --internal_exon_gtf ../train/data/annotation/gencode.v49.annotation.gtf --internal_exon_n 1000 \
  --reference_fasta ../train/data/hg38.fa \
  --me5_lookup ../train/splice_train/data/me5_lookup.npy \
  --max_nt 6200 --step 250 --max_peaks 1000 --paired --out threess_profile

echo "[run] done=$(date)"; echo "[run] outputs:"
ls -la threess_profile_long.tsv threess_profile_paired_bydist.tsv threess_profile.png 2>/dev/null || true