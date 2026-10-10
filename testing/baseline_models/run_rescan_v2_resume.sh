#!/bin/bash
#SBATCH --job-name=rescan_v2_resume
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rescan_v2_resume_%j.out
#SBATCH --error=logs/rescan_v2_resume_%j.err

set -euo pipefail

PY=python
SCRIPT="../scripts/rescan_cnn_v2.py"
CHECK_SCRIPT="../scripts/check_scan_version.py"
SCAN_DIR="../train/gene_scans"
PC_GENES_BED="../train/data/pc_genes.bed"
REFERENCE_FASTA="../../hg38.fa"
V2_CKPT="../train/cnn_runs/cnnv2_20260522_163423_41078027/cnn_model.pt"
LOG_JSON="../logs/rescan_v2_resume_summary.json"
TODO_FILE="../logs/rescan_todo_${SLURM_JOB_ID}.txt"

mkdir -p ../logs

module purge 2>/dev/null || true
module load StdEnv/2023 arrow/17.0.0
source ~/bpnet_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date)] Rescan v2 (resume)"
echo "  checkpoint:  $V2_CKPT"
echo "  scan_dir:    $SCAN_DIR"
nvidia-smi || true

echo ""
echo "=========================================="
echo "PHASE 1: identify un-rescanned genes"
echo "=========================================="
srun $PY "$CHECK_SCRIPT" \
    --scan_dir   "$SCAN_DIR" \
    --write_todo "$TODO_FILE"

if [[ ! -s "$TODO_FILE" ]]; then
    echo "Nothing to do — all parquets are at v2."
    exit 0
fi

N_TODO=$(wc -l < "$TODO_FILE")
echo ""
echo "  $N_TODO genes need rescoring; TODO list at $TODO_FILE"

echo ""
echo "=========================================="
echo "PHASE 2: rescan the $N_TODO un-rescanned genes"
echo "=========================================="
srun $PY "$SCRIPT" \
    --scan_dir         "$SCAN_DIR" \
    --pc_genes_bed     "$PC_GENES_BED" \
    --reference_fasta  "$REFERENCE_FASTA" \
    --v2_checkpoint    "$V2_CKPT" \
    --save_logits \
    --log_json         "$LOG_JSON" \
    --only_gene_file   "$TODO_FILE"

echo ""
echo "[$(date)] Done."
echo ""

echo "=========================================="
echo "PHASE 3: post-rescan verification"
echo "=========================================="
srun $PY "$CHECK_SCRIPT" --scan_dir "$SCAN_DIR"