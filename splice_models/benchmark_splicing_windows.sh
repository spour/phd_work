#!/usr/bin/env bash
# sweep over (window config) and feature mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POS_BED="${POS_BED:-../data/Dominant_CPA_baseline_total.bed}"
NEG_BED="${NEG_BED:-../figures/cpccpa_3way_rflrcnn_intragenic.bed}"
REFERENCE_FASTA="${REFERENCE_FASTA:-../../../hg38.fa}"
U1_LOOKUP="${U1_LOOKUP:-data/u1_lookup.npy}"
ME5_LOOKUP="${ME5_LOOKUP:-data/me5_lookup.npy}"
N_WORKERS="${N_WORKERS:-1}"

EXTRACT_SCRIPT="${EXTRACT_SCRIPT:-$SCRIPT_DIR/extract_splicing_features.py}"
BUILD_U1_SCRIPT="${BUILD_U1_SCRIPT:-$SCRIPT_DIR/build_u1_lookup.py}"
BUILD_ME5_SCRIPT="${BUILD_ME5_SCRIPT:-$SCRIPT_DIR/build_me5_lookup.py}"
LR_SCRIPT="${LR_SCRIPT:-$SCRIPT_DIR/minimal_splice_lr.py}"
GBT_SCRIPT="${GBT_SCRIPT:-$SCRIPT_DIR/run_gbt_7axis.py}"

OUTDIR="${OUTDIR:-splicing_benchmark}"
mkdir -p "$OUTDIR"

CONFIGS=(
    "3000 1000"     # the canonical original
    "5000 1000"     # wider upstream
    "10000 2000"    # very wide
    "1500 500"      # narrow
)

echo "[bench] paths:"
echo "  SCRIPT_DIR       = $SCRIPT_DIR"
echo "  CWD              = $(pwd)"
echo "  POS_BED          = $POS_BED"
echo "  NEG_BED          = $NEG_BED"
echo "  REFERENCE_FASTA  = $REFERENCE_FASTA"
echo "  U1_LOOKUP        = $U1_LOOKUP"
echo "  EXTRACT_SCRIPT   = $EXTRACT_SCRIPT"
echo "  OUTDIR           = $OUTDIR"
echo ""

missing=0
check_exists() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then
        printf "  [OK]      %-22s  -> %s\n" "$label" "$path"
    else
        printf "  [MISSING] %-22s  -> %s\n" "$label" "$path"
        missing=$((missing + 1))
    fi
}
echo "[bench] required-files check:"
check_exists "POS_BED"          "$POS_BED"
check_exists "NEG_BED"          "$NEG_BED"
check_exists "REFERENCE_FASTA"  "$REFERENCE_FASTA"
check_exists "EXTRACT_SCRIPT"   "$EXTRACT_SCRIPT"
[[ -e "$LR_SCRIPT" ]] && check_exists "LR_SCRIPT"  "$LR_SCRIPT" || \
    echo "  [WARN]    LR_SCRIPT              -> $LR_SCRIPT  (will skip LR runs)"
[[ -e "$GBT_SCRIPT" ]] && check_exists "GBT_SCRIPT" "$GBT_SCRIPT" || \
    echo "  [WARN]    GBT_SCRIPT             -> $GBT_SCRIPT  (will skip GBT runs)"

if [[ "$missing" -gt 0 ]]; then
    echo ""
    echo "[bench] ERROR: $missing required input(s) missing. Override via env vars:"
    echo "  POS_BED=/path/to/canonical.bed \\"
    echo "  NEG_BED=/path/to/3way.bed \\"
    echo "  REFERENCE_FASTA=/path/to/hg38.fa \\"
    echo "  bash $0"
    exit 1
fi
echo ""

if [[ ! -f "$U1_LOOKUP" ]]; then
    echo "[bench] building U1 lookup at $U1_LOOKUP"
    python3 "$BUILD_U1_SCRIPT" --out "$U1_LOOKUP"
fi

if [[ ! -f "$ME5_LOOKUP" ]]; then
    echo "[bench] building MaxEntScan 5'SS lookup at $ME5_LOOKUP"
    python3 "$BUILD_ME5_SCRIPT" --out "$ME5_LOOKUP"
fi

for cfg in "${CONFIGS[@]}"; do
    up="$(echo "$cfg" | awk '{print $1}')"
    dn="$(echo "$cfg" | awk '{print $2}')"
    tag="u${up}_d${dn}"
    cfgdir="$OUTDIR/$tag"
    mkdir -p "$cfgdir"

    echo ""
    echo "=========================================="
    echo "Config: upstream=$up  downstream=$dn"
    echo "=========================================="

    features_tsv="$cfgdir/features.tsv"

    if [[ -s "$features_tsv" ]]; then
        echo "[bench] features already at $features_tsv; skipping extraction"
    else
        echo "[bench] extracting features -> $features_tsv"
        python3 "$EXTRACT_SCRIPT" \
            --pos_bed "$POS_BED" \
            --neg_bed "$NEG_BED" \
            --reference_fasta "$REFERENCE_FASTA" \
            --u1_lookup "$U1_LOOKUP" \
            --me5_lookup "$ME5_LOOKUP" \
            --upstream_nt "$up" \
            --downstream_nt "$dn" \
            --n_bins 8 \
            --n_workers "$N_WORKERS" \
            --out_tsv "$features_tsv"
    fi

    if [[ -e "$LR_SCRIPT" ]]; then
        echo "[bench] LR with summary stats (--model_tier all)"
        python3 "$LR_SCRIPT" \
            --features_tsv "$features_tsv" \
            --out_prefix "$cfgdir/lr" \
            --cv_mode groupkfold --n_splits 5 \
            --seed 1 --n_boot 1000 \
            --class_weight_balanced \
            --model_tier all || echo "[bench] LR failed for $tag"
    fi

    if [[ -e "$GBT_SCRIPT" ]]; then
        echo "[bench] GBT 7-axis (raw)"
        python3 "$GBT_SCRIPT" \
            --mode raw \
            --features_tsv "$features_tsv" \
            --out_prefix "$cfgdir/gbt_raw" \
            --skip_add_minimal || echo "[bench] GBT raw failed for $tag"

        echo "[bench] GBT 7-axis (wg)"
        python3 "$GBT_SCRIPT" \
            --mode wg \
            --features_tsv "$features_tsv" \
            --out_prefix "$cfgdir/gbt_wg" \
            --skip_add_minimal || echo "[bench] GBT wg failed for $tag"

        echo "[bench] GBT 7-axis (wg) on both-class subset"
        both_tsv="$cfgdir/features_bothclass.tsv"
        python3 - "$features_tsv" "$both_tsv" << 'PY_EOF'
import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, sep="\t")
gene_classes = df.groupby("gene")["y"].nunique()
both_genes = gene_classes[gene_classes == 2].index
sub = df[df["gene"].isin(both_genes)]
sub.to_csv(out, sep="\t", index=False)
print(f"[bench-bothclass]   full: {len(df):,} sites / {df['gene'].nunique():,} genes")
print(f"[bench-bothclass] subset: {len(sub):,} sites / {sub['gene'].nunique():,} genes "
      f"({len(sub)/len(df)*100:.1f}% retained)")
PY_EOF
        python3 "$GBT_SCRIPT" \
            --mode wg \
            --features_tsv "$both_tsv" \
            --out_prefix "$cfgdir/gbt_wg_bothclass" \
            --skip_add_minimal \
            --no_shap || echo "[bench] GBT wg bothclass failed for $tag"
    fi

    echo "[bench] GBT on full per-bin feature table"
    cfgdir_abs="$(cd "$cfgdir" && pwd)"
    features_tsv_abs="$(cd "$(dirname "$features_tsv")" && pwd)/$(basename "$features_tsv")"
    tag_local="$tag"
    python3 - "$features_tsv_abs" "$cfgdir_abs" "$tag_local" << 'PY_EOF'
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
done

echo ""
echo "=========================================="
echo "Aggregating results"
echo "=========================================="

OUTDIR_ABS="$(cd "$OUTDIR" && pwd)"
python3 - "$OUTDIR_ABS" << 'PY_EOF'
import sys, glob, os, pandas as pd
outdir = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(outdir, "*/"))):
    cfg = os.path.basename(os.path.normpath(d))
    lr_summary = os.path.join(d, "lr.summary.tsv")
    if os.path.exists(lr_summary):
        df = pd.read_csv(lr_summary, sep="\t")
        best = df.sort_values("auc", ascending=False).iloc[0]
        rows.append({
            "config": cfg, "approach": "LR_summary_best",
            "model": best["model"],
            "auc": best["auc"], "mrr": best.get("mrr", float("nan")),
            "p_at_1": best.get("top1_acc", float("nan")),
            "n_features": int(best["n_features"]),
        })
    for m in ("raw", "wg"):
        gbt_summary = os.path.join(d, f"gbt_{m}.summary.tsv")
        if os.path.exists(gbt_summary):
            df = pd.read_csv(gbt_summary, sep="\t")
            rows.append({
                "config": cfg, "approach": f"GBT_mech7_{m}",
                "model": df["model"].iloc[0],
                "auc": df["site_auroc_oof"].iloc[0],
                "mrr": df["mrr"].iloc[0],
                "p_at_1": df["top1_acc"].iloc[0],
                "n_features": int(df["n_features"].iloc[0]),
            })
    # both-class subset (wg only)
    bc_summary = os.path.join(d, "gbt_wg_bothclass.summary.tsv")
    if os.path.exists(bc_summary):
        df = pd.read_csv(bc_summary, sep="\t")
        rows.append({
            "config": cfg, "approach": "GBT_mech7_wg_bothclass",
            "model": df["model"].iloc[0],
            "auc": df["site_auroc_oof"].iloc[0],
            "mrr": df["mrr"].iloc[0],
            "p_at_1": df["top1_acc"].iloc[0],
            "n_features": int(df["n_features"].iloc[0]),
        })
    gbt_full = os.path.join(d, "gbt_full.summary.tsv")
    if os.path.exists(gbt_full):
        df = pd.read_csv(gbt_full, sep="\t")
        for _, r in df.iterrows():
            rows.append({
                "config": cfg, "approach": f"GBT_full_per_bin_{r['feature_mode'].split('_')[-1]}",
                "model": r["feature_mode"],
                "auc": r["auc_oof"],
                "mrr": float("nan"),
                "p_at_1": float("nan"),
                "n_features": int(r["n_features"]),
            })

out = pd.DataFrame(rows)
summary_path = os.path.join(outdir, "summary_by_config.tsv")
out.to_csv(summary_path, sep="\t", index=False)
print(out.to_string(index=False))
print(f"\n[bench] wrote {summary_path}")
PY_EOF

echo ""
echo "[bench] Done."