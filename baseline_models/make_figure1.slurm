#!/bin/bash
#SBATCH --job-name=figures
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --output=logs/figures_%j.out
#SBATCH --error=logs/figures_%j.err

set -euo pipefail

PY=python
SCRIPT="../scripts/make_figure1.py"
SCAN_DIR="../train/gene_scans"
PC_GENES_BED="../train/data/pc_genes.bed"

CANONICAL_BED="../train/data/Dominant_CPA_baseline_total.bed"
TEST_CHROMS="../train/data/test_chroms.txt"

THREE_PRIME_ZONE_BED="../train/data/annotation/three_prime_zone.bed"
POLYASITE_BED="../train/data/atlas.chr.250.bed"              # PolyASite 2.0
POLYADB_MAIN_BED="../train/data/polyadb_v4/HumanPas/polyadb_v4_main.bed"
POLYADB_MAX_BED="../train/data/polyadb_v4/HumanPas/polyadb_v4_max.bed"
RANDOM_BED="../train/data/random_control.bed"

RF_DIR="../train/rf_runs/rf_20260517_080850_40272020"
LR_DIR="../train/lr_runs/lr_20260517_081451_40272571"
CNN_DIR="../train/cnn_runs/cnnv2_20260522_163423_41078027"

OUT_DIR="../train/figures"
TRACE_GENES="FGF20 ERP27 PABPN1 CSTF3 RCC1L PPIF"
REF_FASTA="../../hg38.fa"

SKIP=""

MODELS=""

CLUSTER_SLACK=250        # single-link slack (nt) for multi-model peak clustering
CLUSTER_HALFWIDTH=250    # +/- this many nt around cluster center in tier BEDs
CPCCPA_SALIENCY_TIER="tier3_exactly_three"   # which tier feeds Fig 12 cryptic-saliency

mkdir -p ../logs "$OUT_DIR"

module purge 2>/dev/null || true
module load StdEnv/2023 bedtools/2.31.0 arrow/17.0.0
source ~/bpnet_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -c "import numpy, pandas, plotnine, pyarrow; \
print(f'numpy {numpy.__version__}'); \
print(f'pandas {pandas.__version__}'); \
print(f'plotnine {plotnine.__version__}'); \
print(f'pyarrow {pyarrow.__version__}')"

echo
echo "[$(date)] Verifying CPCCPA inputs..."
for f in "$THREE_PRIME_ZONE_BED" "$POLYASITE_BED" "$POLYADB_MAIN_BED" \
         "$POLYADB_MAX_BED" "$RANDOM_BED"; do
    if [[ ! -s "$f" ]]; then
        echo "  ERROR: required CPCCPA input missing or empty: $f"
        exit 1
    fi
    n_lines=$(wc -l < "$f")
    echo "  OK: $f ($n_lines lines)"
done

echo
echo "[$(date)] Running $SCRIPT"
echo "  scan_dir:       $SCAN_DIR"
echo "  out_dir:        $OUT_DIR"
echo "  trace_gene:     $TRACE_GENES"
echo "  test_chroms:    $TEST_CHROMS"
echo "  models:         ${MODELS:-(auto-detect)}"
[[ -n "$SKIP" ]] && echo "  skip_fig:       $SKIP"
echo "  cluster_slack:  $CLUSTER_SLACK"
echo "  saliency_tier:  $CPCCPA_SALIENCY_TIER"

EXTRA_ARGS=()
[[ -n "$SKIP"   ]] && EXTRA_ARGS+=(--skip_fig "$SKIP")
[[ -n "$MODELS" ]] && EXTRA_ARGS+=(--models $MODELS)

srun $PY "$SCRIPT" \
    --scan_dir              "$SCAN_DIR" \
    --pc_genes_bed          "$PC_GENES_BED" \
    --canonical_bed         "$CANONICAL_BED" \
    --three_prime_zone_bed  "$THREE_PRIME_ZONE_BED" \
    --polyasite_bed         "$POLYASITE_BED" \
    --polyadb_main_bed      "$POLYADB_MAIN_BED" \
    --polyadb_max_bed       "$POLYADB_MAX_BED" \
    --random_bed            "$RANDOM_BED" \
    --test_chroms           "$TEST_CHROMS" \
    --rf_dir                "$RF_DIR" \
    --lr_dir                "$LR_DIR" \
    --cnn_dir               "$CNN_DIR" \
    --trace_gene            $TRACE_GENES \
    --out_dir               "$OUT_DIR" \
    --reference_fasta       "$REF_FASTA" \
    --cluster_slack         "$CLUSTER_SLACK" \
    --cluster_halfwidth     "$CLUSTER_HALFWIDTH" \
    --cpccpa_saliency_tier  "$CPCCPA_SALIENCY_TIER" \
    --enable_saliency \
    "${EXTRA_ARGS[@]}"

echo "[$(date)] Done. Figures in $OUT_DIR"
ls -la "$OUT_DIR"