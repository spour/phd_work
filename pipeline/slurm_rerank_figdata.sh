#!/bin/bash
#SBATCH --job-name=rerank_figdata
#SBATCH --account=def-thughes
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=rerank_figdata_%j.out
#SBATCH --error=rerank_figdata_%j.err
set -euo pipefail
module load arrow/17.0.0 StdEnv/2023 2>/dev/null || true
source ~/bpnet_env/bin/activate
CLEAN="$SLURM_SUBMIT_DIR"
cd "$CLEAN"
S="$CLEAN/scripts"
OUTD="$CLEAN/figdata_final"
mkdir -p "$OUTD"
echo "[run] host=$(hostname) dir=$CLEAN start=$(date)"

TR=shortlist_crossfit_feat_train.parquet
TE=shortlist_crossfit_feat_test.parquet
SUMM=shortlist_crossfit_summary.tsv
GTF=../train/data/annotation/gencode.v49.annotation.gtf
ATLAS=../train/data/atlas.chr.250.bed

for f in "$TR" "$TE" "$SUMM" "$S/score_rerank1.py"; do
  cat "$f" > /dev/null 2>&1 || { echo "[err] cannot read $f -- aborting" >&2; exit 1; }
done
echo "[run] inputs readable (cross-fit peak set)"

echo "[1/6] score_rerank (metrics + importance + genomewide LOCO)"
python3 "$S/score_rerank1.py" \
  --train_feat "$TR" --test_feat "$TE" --summary "$SUMM"  --model gbt --monotone --genomewide --importance \
  --out "$OUTD/rerank"
GW_SCORED="$OUTD/rerank_genomewide_scored.parquet"
TE_SCORED="$OUTD/rerank_test_scored.parquet"

echo "[2/6] permutation null"
python3 "$S/rerank_permnull.py" \
  --train_feat "$TR" --test_feat "$TE" --n_perm 1000 --n_retrain 200 \
  --out "$OUTD/rerank_permnull"

echo "[3/6] ablation"
python3 "$S/rerank_ablation.py" \
  --train_feat "$TR" --test_feat "$TE" --summary "$SUMM" \
  --out "$OUTD/rerank_ablation"

echo "[4/6] feature violins"
python3 "$S/feature_violins.py" \
  --train_feat "$TR" --test_feat "$TE" \
  --out "$OUTD/feature_violins"

echo "[5/6] override geometry"
python3 "$S/override_geometry.py" \
  --scored "$GW_SCORED" --max_kb 50 \
  --out "$OUTD/override_geometry"

echo "[6/6] stratified + residual-miss atlas"
python3 "$S/rerank_stratified.py" \
  --scored "$GW_SCORED" --gtf "$GTF" --summary "$SUMM" \
  --out "$OUTD/rerank_strat" || echo "[warn] stratified failed (needs GTF); panel D/E-length may be missing"

if [ -f "$ATLAS" ]; then
  python3 "$S/residual_miss_atlas.py" \
    --scored "$GW_SCORED" --atlas_bed "$ATLAS" --tol 50 \
    --out "$OUTD/residual_miss_gw" || echo "[warn] residual_miss failed; panel C may use defaults"
fi

echo ""
echo "[run] done=$(date)"
echo "[run] figure-data files in $OUTD:"
ls -la "$OUTD"/*.tsv 2>/dev/null
echo ""
echo "Next: cd figdata_final && Rscript ../scripts/make_rerank_figure.R"
echo "(TSV filenames unchanged, so no R edits needed; peak set is now cross-fit/leak-free)"