#!/bin/bash
#SBATCH --job-name=fivess_profile
#SBATCH --account=def-thughes
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=fivess_profile_%j.out
#SBATCH --error=fivess_profile_%j.err
set -euo pipefail
module load arrow/17.0.0 StdEnv/2023 2>/dev/null || true
source ~/bpnet_env/bin/activate   # pysam, MaxEntScan, numpy, pandas, matplotlib
CLEAN="$SLURM_SUBMIT_DIR"; cd "$CLEAN"; S="$CLEAN/scripts"
echo "[run] host=$(hostname) dir=$CLEAN start=$(date)"

SHORTLIST=shortlist_crossfit_windows.parquet
CPA=../train/data/Dominant_CPA_baseline_total.bed
GENES=../train/data/pc_genes.bed
GTF=../train/data/annotation/gencode.v49.annotation.gtf
FASTA=../train/data/hg38.fa
U1=../train/splice_train/data/u1_lookup.npy
ME5=../train/splice_train/data/me5_lookup.npy

for f in "$SHORTLIST" "$FASTA" "$S/fivess_distance_profile.py"; do
  cat "$f" > /dev/null 2>&1 || { echo "[err] cannot read $f -- aborting" >&2; exit 1; }
done
echo "[run] inputs readable"

COMMON=(--scripts_dir "$S" \
  --shortlist "$SHORTLIST" \
  --cpa_bed   "$CPA" \
  --genes_bed "$GENES" --random_n 4000 \
  --internal_exon_gtf "$GTF" --internal_exon_n 4000 \
  --reference_fasta "$FASTA" \
  --max_nt 2000 --step 100 --max_peaks 4000 --paired)

echo "[1/2] U1 hybridization upstream profile + paired"
python3 "$S/fivess_distance_profile.py" "${COMMON[@]}" \
  --u1_lookup "$U1" --metric u1 --out fivess_profile_u1

echo "[2/2] MaxEntScan 5'ss upstream profile + paired"
python3 "$S/fivess_distance_profile.py" "${COMMON[@]}" \
  --me5_lookup "$ME5" --metric me5 --out fivess_profile_me5

echo ""; echo "[run] done=$(date)"; echo "[run] outputs:"
ls -la fivess_profile_u1* fivess_profile_me5* 2>/dev/null
echo ""
echo "READ THE PAIRED RESULT: *_paired.tsv + 'WITHIN-GENE paired' stderr block is the"
echo "honest suppression test. Suppression predicts competitor > canonical in upstream"
echo "U1 density/total within the same gene."