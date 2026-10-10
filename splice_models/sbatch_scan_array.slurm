#!/bin/bash
#SBATCH --job-name=splice_scan
#SBATCH --array=0-127
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/splice_scan_%A_%a.out
#SBATCH --error=logs/splice_scan_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SPLICE_TRAIN_DIR="${SPLICE_TRAIN_DIR:-$SLURM_SUBMIT_DIR}"
else
    SPLICE_TRAIN_DIR="${SPLICE_TRAIN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi

cd "$SPLICE_TRAIN_DIR"
echo "[scan] working in: $PWD"
mkdir -p logs
mkdir -p gene_scans/splicing_scores/per_gene
mkdir -p gene_scans/splicing_scores/gene_lists

N_ARRAY_TASKS="${SLURM_ARRAY_TASK_COUNT:-128}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

N_WORKERS_PER_TASK=4

GENE_LISTS_DIR="gene_scans/splicing_scores/gene_lists"
CPA_PARQUET_DIR="/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/train/gene_scans/per_gene"

if [[ ! -f "$GENE_LISTS_DIR/chunk_000.txt" ]]; then
    if [[ "$TASK_ID" == "0" ]]; then
        echo "[scan] building gene list chunks (first run)"
        ls "$CPA_PARQUET_DIR"/*.parquet | xargs -n1 basename | sed 's/\.parquet$//' | sort > "$GENE_LISTS_DIR/all_genes.txt"
        N_GENES=$(wc -l < "$GENE_LISTS_DIR/all_genes.txt")
        echo "[scan]   $N_GENES total genes"
        split -n "l/$N_ARRAY_TASKS" -d -a 3 "$GENE_LISTS_DIR/all_genes.txt" "$GENE_LISTS_DIR/chunk_"
        for f in "$GENE_LISTS_DIR/chunk_"???; do
            mv "$f" "$f.txt"
        done
        echo "[scan]   wrote $(ls $GENE_LISTS_DIR/chunk_*.txt | wc -l) chunks"
    else
        echo "[scan] task $TASK_ID waiting for chunks (task 0 builds them)"
        while [[ ! -f "$GENE_LISTS_DIR/chunk_000.txt" ]]; do
            sleep 5
        done
        sleep 5
    fi
fi

CHUNK_FILE=$(printf "$GENE_LISTS_DIR/chunk_%03d.txt" "$TASK_ID")
if [[ ! -f "$CHUNK_FILE" ]]; then
    echo "[scan] ERROR: chunk file $CHUNK_FILE not found"
    exit 1
fi
N_GENES_CHUNK=$(wc -l < "$CHUNK_FILE")
echo "[scan] task $TASK_ID: $N_GENES_CHUNK genes, parallelizing $N_WORKERS_PER_TASK-way"

TMPDIR_LOCAL="${SLURM_TMPDIR:-/tmp}/splice_scan_$$"
mkdir -p "$TMPDIR_LOCAL"
SUBCHUNK_PREFIX="$TMPDIR_LOCAL/subchunk_"
split -n "l/$N_WORKERS_PER_TASK" -d -a 1 "$CHUNK_FILE" "$SUBCHUNK_PREFIX"
SUBCHUNKS=("$SUBCHUNK_PREFIX"?)
echo "[scan]   sub-chunk sizes:"
for sc in "${SUBCHUNKS[@]}"; do
    echo "[scan]     $(basename $sc): $(wc -l < $sc) genes"
done

module load StdEnv/2023 arrow/17.0.0 2>/dev/null || true
source ~/bpnet_env/bin/activate

echo "[scan] launching $N_WORKERS_PER_TASK parallel workers"
PIDS=()
for i in $(seq 0 $((N_WORKERS_PER_TASK - 1))); do
    SC="${SUBCHUNKS[$i]}"
    if [[ ! -f "$SC" ]]; then
        echo "[scan] worker $i: no sub-chunk file, skipping"
        continue
    fi
    LOGFILE="logs/splice_scan_${SLURM_ARRAY_JOB_ID:-test}_${TASK_ID}_w${i}.log"
    echo "[scan] worker $i: $SC -> $LOGFILE"
    (
        python3 scripts/scan_splicing_genomewide.py \
            --gene_list_file "$SC" \
            --cpa_parquet_dir "$CPA_PARQUET_DIR" \
            --pc_genes_bed ../data/pc_genes.bed \
            --reference_fasta ../data/hg38.fa \
            --u1_lookup data/u1_lookup.npy \
            --me5_lookup data/me5_lookup.npy \
            --gene_baselines data/gene_baselines.tsv \
            --model_joblib splicing_benchmark/u3000_d1000/chrsplit/gbt_combined_chrsplit.joblib \
            --out_dir gene_scans/splicing_scores/per_gene \
            --skip_if_exists \
            --log_every 5 > "$LOGFILE" 2>&1
    ) &
    PIDS+=($!)
done

echo "[scan] waiting for ${#PIDS[@]} workers..."
EXIT_CODES=()
for PID in "${PIDS[@]}"; do
    wait $PID
    EXIT_CODES+=($?)
done

echo "[scan] all workers complete. exit codes: ${EXIT_CODES[*]}"
N_FAILED=0
for ec in "${EXIT_CODES[@]}"; do
    [[ "$ec" != "0" ]] && N_FAILED=$((N_FAILED + 1))
done
if [[ "$N_FAILED" -gt 0 ]]; then
    echo "[scan] WARNING: $N_FAILED workers failed. Check logs/splice_scan_*_w*.log"
fi

rm -rf "$TMPDIR_LOCAL"
echo "[scan] task $TASK_ID complete"