#!/bin/bash
#SBATCH --job-name=score_genes
#SBATCH --array=0-19
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_%A_%a.out
#SBATCH --error=logs/score_%A_%a.err

set -euo pipefail

PY=python
SCRIPT="../scripts/score_genes.py"
GENES_BED="../train/data/pc_genes.bed"
GENOME="/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/hg38.fa"                  # <-- SET THIS
PWM_LIST="/scratch/spour98/identify_model/pwm_paths"

RF_DIR="../train/rf_runs/rf_20260517_080850_40272020"
LR_DIR="../train/lr_runs/lr_20260517_081451_40272571"
CNN_DIR="../train/cnn_runs/cnn_20260517_083922_40273075"

CPA_BED="../train/data/Dominant_CPA_baseline_total.bed"
CPA_BED_IS_WINDOW=1

OUTDIR="../train/gene_scans"
N_SHARDS=20
module load StdEnv/2023 bedtools arrow/17.0.0
mkdir -p ../logs "$OUTDIR"
source ~/venv_2025/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date)] Shard $SLURM_ARRAY_TASK_ID / $N_SHARDS"
nvidia-smi || true

CPA_WIN_FLAG=""
if [[ "${CPA_BED_IS_WINDOW}" == "1" ]]; then
    CPA_WIN_FLAG="--cpa_bed_is_window"
fi

srun $PY "$SCRIPT" \
    --genes_bed   "$GENES_BED" \
    --cpa_bed     "$CPA_BED" \
    $CPA_WIN_FLAG \
    --genome      "$GENOME" \
    --pwms        "$PWM_LIST" \
    --bg          0.3 0.2 0.2 0.3 \
    --rf_dir      "$RF_DIR" \
    --lr_dir      "$LR_DIR" \
    --cnn_dir     "$CNN_DIR" \
    --outdir      "$OUTDIR" \
    --shard_index "$SLURM_ARRAY_TASK_ID" \
    --n_shards    "$N_SHARDS" \
    --cnn_batch   512 \
    --resume

echo "[$(date)] Shard $SLURM_ARRAY_TASK_ID done"