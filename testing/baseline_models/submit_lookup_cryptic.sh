#!/bin/bash
#SBATCH --job-name=cryptic_lookup
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cryptic_%j.out
#SBATCH --error=logs/cryptic_%j.err

set -euo pipefail

PY=python
SCRIPT="../scripts/lookup_cryptic_sites.py"
SITES_BED="../train/Cryptic_CPA_sites.hg38.bed"
SCAN_DIR="../train/gene_scans"
OUT_TSV="../train/cryptic_scored.tsv"

mkdir -p ../logs

module purge 2>/dev/null || true
module load StdEnv/2023 bedtools arrow/17.0.0
source ~/venv_2025/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -c "import numpy, pandas, pyarrow; \
print(f'numpy {numpy.__version__} from {numpy.__file__}'); \
print(f'pandas {pandas.__version__}'); \
print(f'pyarrow {pyarrow.__version__}')"

echo "[$(date)] Looking up cryptic sites"
echo "  sites: $SITES_BED ($(wc -l < $SITES_BED) lines)"
echo "  scans: $SCAN_DIR"
echo "  out:   $OUT_TSV"

srun $PY "$SCRIPT" \
    --sites_bed "$SITES_BED" \
    --scan_dir  "$SCAN_DIR" \
    --out_tsv   "$OUT_TSV"

echo "[$(date)] Done"