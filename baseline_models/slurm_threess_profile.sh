#!/bin/bash
#SBATCH --job-name=threess_prof
#SBATCH --account=def-thughes
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=threess_prof_%j.out
#SBATCH --error=threess_prof_%j.err

set -euo pipefail
module load arrow/17.0.0 StdEnv/2023 2>/dev/null || true
source ~/bpnet_env/bin/activate

CLEAN="$SLURM_SUBMIT_DIR"
cd "$CLEAN"
echo "[run] host=$(hostname)  dir=$CLEAN  start=$(date)"

for f in scripts/threess_distance_profile.py scripts/compute_gene_baselines.py \
         shortlist_windows.parquet train/data/hg38.fa \
         train/data/Dominant_CPA_baseline_total.bed train/data/atlas.chr.250.bed; do
  if ! cat "$f" > /dev/null 2>&1; then
    echo "[err] cannot read $f (filesystem issue?) -- aborting before the scan" >&2
    exit 1
  fi
done
echo "[run] all inputs readable"

python3 scripts/threess_distance_profile.py --scripts_dir $(pwd)/scripts \
  --shortlist shortlist_windows.parquet \
  --cpa_bed train/data/Dominant_CPA_baseline_total.bed \
  --atlas_bed train/data/atlas.chr.250.bed --atlas_ann_split TE,IN \
  --genes_bed train/data/pc_genes.bed --random_n 15000 \
  --internal_exon_gtf train/data/annotation/gencode.v49.annotation.gtf --internal_exon_n 15000 \
  --reference_fasta train/data/hg38.fa \
  --max_nt 6200 --step 250 --paired --out threess_profile

echo "[run] done=$(date)"
echo "[run] outputs:"
ls -la threess_profile_long.tsv threess_profile.png 2>/dev/null || true
