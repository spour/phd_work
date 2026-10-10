#!/usr/bin/env bash
#SBATCH --job-name=splice_bench
#SBATCH --array=0-2                    # 3 configs: u3000_d1000, u5000_d1000, u10000_d2000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/splice_bench_%A_%a.out
#SBATCH --error=logs/splice_bench_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SPLICE_TRAIN_DIR="${SPLICE_TRAIN_DIR:-$SLURM_SUBMIT_DIR}"
else
    SPLICE_TRAIN_DIR="${SPLICE_TRAIN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
SCRIPT_DIR="$SPLICE_TRAIN_DIR/scripts"

cd "$SPLICE_TRAIN_DIR"
echo "[task] working in: $PWD"
mkdir -p logs

source ~/bpnet_env/bin/activate

CONFIGS=(
    "3000 1000"
    "5000 1000"
    "10000 2000"
)
cfg="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
up=$(echo "$cfg" | awk '{print $1}')
dn=$(echo "$cfg" | awk '{print $2}')
tag="u${up}_d${dn}"

echo "[task] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "[task] config: upstream=$up downstream=$dn  tag=$tag"
echo "[task] start: $(date)"
echo ""

POS_BED="${POS_BED:-../data/Dominant_CPA_baseline_total.bed}"
NEG_BED="${NEG_BED:-../figures/cpccpa_3way_rflrcnn_intragenic.bed}"
REFERENCE_FASTA="${REFERENCE_FASTA:-../data/hg38.fa}"
U1_LOOKUP="${U1_LOOKUP:-data/u1_lookup.npy}"
ME5_LOOKUP="${ME5_LOOKUP:-data/me5_lookup.npy}"

OUTDIR="${OUTDIR:-splicing_benchmark}"
cfgdir="$OUTDIR/$tag"
mkdir -p "$cfgdir"

EXTRACT_SCRIPT="$SCRIPT_DIR/extract_splicing_features.py"
LR_SCRIPT="$SCRIPT_DIR/minimal_splice_lr.py"
GBT_SCRIPT="$SCRIPT_DIR/run_gbt_7axis.py"

for f in "$POS_BED" "$NEG_BED" "$REFERENCE_FASTA" "$U1_LOOKUP" "$ME5_LOOKUP" \
         "$EXTRACT_SCRIPT" "$LR_SCRIPT" "$GBT_SCRIPT"; do
    if [[ ! -e "$f" ]]; then
        echo "[task] ERROR: missing $f"
        exit 1
    fi
done

features_tsv="$cfgdir/features.tsv"

if [[ -s "$features_tsv" ]]; then
    echo "[task] features already at $features_tsv; skipping extraction"
else
    echo "[task] extracting features -> $features_tsv"
    python3 "$EXTRACT_SCRIPT" \
        --pos_bed "$POS_BED" \
        --neg_bed "$NEG_BED" \
        --reference_fasta "$REFERENCE_FASTA" \
        --u1_lookup "$U1_LOOKUP" \
        --me5_lookup "$ME5_LOOKUP" \
        --upstream_nt "$up" \
        --downstream_nt "$dn" \
        --n_bins 8 \
        --out_tsv "$features_tsv"
fi

echo ""
echo "[task] extraction done: $(date)"
echo ""

echo "[task] LR with summary stats"
python3 "$LR_SCRIPT" \
    --features_tsv "$features_tsv" \
    --out_prefix "$cfgdir/lr" \
    --cv_mode groupkfold --n_splits 5 \
    --seed 1 --n_boot 1000 \
    --class_weight_balanced \
    --model_tier all || echo "[task] LR failed"

echo "[task] GBT 7-axis raw (no SHAP)"
python3 "$GBT_SCRIPT" \
    --mode raw \
    --features_tsv "$features_tsv" \
    --out_prefix "$cfgdir/gbt_raw" \
    --skip_add_minimal --no_shap || echo "[task] GBT raw failed"

echo "[task] GBT 7-axis wg (no SHAP)"
python3 "$GBT_SCRIPT" \
    --mode wg \
    --features_tsv "$features_tsv" \
    --out_prefix "$cfgdir/gbt_wg" \
    --skip_add_minimal --no_shap || echo "[task] GBT wg failed"

echo "[task] GBT full per-bin (raw + wg)"
python3 - "$features_tsv" "$cfgdir" "$tag" << 'PY_EOF'
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

features_tsv = sys.argv[1]
cfgdir = sys.argv[2]
tag = sys.argv[3]

df = pd.read_csv(features_tsv, sep="\t")
y = df["y"].values.astype(int)
groups = df["gene"].astype(str).values

full_cols = [c for c in df.columns
              if c.startswith("bin") and (
                  c.endswith("_mean") or c.endswith("_max"))]
print(f"[bench-full] using {len(full_cols)} per-bin features")

def fit_oof(X, y, groups, label):
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gbt", HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, min_samples_leaf=50,
            learning_rate=0.05, class_weight="balanced",
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42)),
    ])
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    print(f"[bench-full] {label}  AUC={auc:.4f}  AP={ap:.4f}  n={len(y)}")
    return auc, ap

# Raw per-bin
X_raw = df[full_cols].values
auc_raw, ap_raw = fit_oof(X_raw, y, groups, "raw")

# Within-gene-centered per-bin
df_wg = df.copy()
gp = df_wg.groupby("gene", sort=False)
for c in full_cols:
    df_wg[c] = df_wg[c] - gp[c].transform("mean")
X_wg = df_wg[full_cols].values
auc_wg, ap_wg = fit_oof(X_wg, y, groups, "wg")

pd.DataFrame([
    {"config": tag, "feature_mode": "full_per_bin_raw",
     "n_features": len(full_cols), "n_sites": len(y),
     "auc_oof": auc_raw, "ap_oof": ap_raw},
    {"config": tag, "feature_mode": "full_per_bin_wg",
     "n_features": len(full_cols), "n_sites": len(y),
     "auc_oof": auc_wg,  "ap_oof": ap_wg},
]).to_csv(f"{cfgdir}/gbt_full.summary.tsv", sep="\t", index=False)
PY_EOF

echo ""
echo "[task] done: $(date)"