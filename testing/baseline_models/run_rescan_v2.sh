#!/bin/bash
#SBATCH --job-name=rescan_v2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rescan_v2_%j.out
#SBATCH --error=logs/rescan_v2_%j.err

set -euo pipefail

PY=python
SCRIPT="../scripts/rescan_cnn_v2.py"
SCAN_DIR="../train/gene_scans"
PC_GENES_BED="../train/data/pc_genes.bed"
REFERENCE_FASTA="../../hg38.fa"
V2_CKPT="../train/cnn_runs/cnnv2_20260522_163423_41078027/cnn_model.pt"
LOG_JSON="../logs/rescan_v2_summary.json"

mkdir -p ../logs

module purge 2>/dev/null || true
module load StdEnv/2023 arrow/17.0.0
source ~/bpnet_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date)] Rescan v2"
echo "  checkpoint:  $V2_CKPT"
echo "  scan_dir:    $SCAN_DIR"
nvidia-smi || true

echo ""
echo "=========================================="
echo "PHASE 1: dry run on 20 genes"
echo "=========================================="
srun $PY "$SCRIPT" \
    --scan_dir         "$SCAN_DIR" \
    --pc_genes_bed     "$PC_GENES_BED" \
    --reference_fasta  "$REFERENCE_FASTA" \
    --v2_checkpoint    "$V2_CKPT" \
    --dry_run \
    --limit 20

echo ""
echo "=========================================="
echo "PHASE 2: full rescan, writing parquets"
echo "=========================================="
srun $PY "$SCRIPT" \
    --scan_dir         "$SCAN_DIR" \
    --pc_genes_bed     "$PC_GENES_BED" \
    --reference_fasta  "$REFERENCE_FASTA" \
    --v2_checkpoint    "$V2_CKPT" \
    --save_logits \
    --log_json         "$LOG_JSON"

echo ""
echo "[$(date)] Done."
echo ""
echo "Cleanup of stale caches:"
ls -la ../train/figures/_cache_peaks_*.parquet 2>/dev/null && \
    echo "  (run: rm -f ../train/figures/_cache_peaks_*.parquet)" || \
    echo "  no stale peak caches found"
ls -la ../train/cryptic_sites_paper/cryptic_cnn.rep.bed 2>/dev/null && \
    echo "  (mv ../train/cryptic_sites_paper/cryptic_cnn.rep.bed{,.v1bak})" || \
    echo "  no stale cnn rep bed found"