#!/usr/bin/env bash
#SBATCH --job-name=splice_agg
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/splice_agg_%j.out
#SBATCH --error=logs/splice_agg_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SPLICE_TRAIN_DIR="${SPLICE_TRAIN_DIR:-$SLURM_SUBMIT_DIR}"
else
    SPLICE_TRAIN_DIR="${SPLICE_TRAIN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
SCRIPT_DIR="$SPLICE_TRAIN_DIR/scripts"
cd "$SPLICE_TRAIN_DIR"
echo "[agg] working in: $PWD"

source ~/bpnet_env/bin/activate

OUTDIR="${OUTDIR:-splicing_benchmark}"
GBT_SCRIPT="$SCRIPT_DIR/run_gbt_7axis.py"

echo "[agg] aggregating from $OUTDIR/"
echo "[agg] start: $(date)"

for cfgdir in "$OUTDIR"/*/; do
    cfg=$(basename "$cfgdir")
    features_tsv="$cfgdir/features.tsv"
    if [[ ! -s "$features_tsv" ]]; then
        echo "[agg] skip $cfg (no features.tsv)"
        continue
    fi
    bothclass_tsv="$cfgdir/features_bothclass.tsv"

    echo ""
    echo "[agg] $cfg: building both-class subset"
    python3 - "$features_tsv" "$bothclass_tsv" << 'PY_EOF'
import sys, pandas as pd
inp, outp = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp, sep="\t")
gc = df.groupby("gene")["y"].nunique()
both = gc[gc == 2].index
sub = df[df["gene"].isin(both)]
sub.to_csv(outp, sep="\t", index=False)
print(f"  full: {len(df):,} sites in {df['gene'].nunique():,} genes")
print(f"  both-class subset: {len(sub):,} sites in {sub['gene'].nunique():,} genes  "
      f"({100*len(sub)/len(df):.0f}%)")
PY_EOF

    echo "[agg] $cfg: GBT wg on both-class subset"
    python3 "$GBT_SCRIPT" \
        --mode wg \
        --features_tsv "$bothclass_tsv" \
        --out_prefix "$cfgdir/gbt_wg_bothclass" \
        --skip_add_minimal --no_shap || echo "[agg] bothclass GBT failed for $cfg"
done

echo ""
echo "[agg] building summary_by_config.tsv"
python3 - "$OUTDIR" << 'PY_EOF'
import os, sys, glob, pandas as pd
outdir = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(outdir, "*/"))):
    cfg = os.path.basename(os.path.normpath(d))

    # LR best-of-library
    lr = os.path.join(d, "lr.summary.tsv")
    if os.path.exists(lr):
        df = pd.read_csv(lr, sep="\t")
        best = df.sort_values("auc", ascending=False).iloc[0]
        rows.append({
            "config": cfg, "approach": "LR_summary_best",
            "model": best["model"], "auc": best["auc"],
            "mrr": best.get("mrr", float("nan")),
            "p_at_1": best.get("top1_acc", float("nan")),
            "n_features": int(best["n_features"]),
            "n_sites": float("nan"),
        })

    # GBT 7-axis raw + wg + bothclass
    for m in ("raw", "wg", "wg_bothclass"):
        p = os.path.join(d, f"gbt_{m}.summary.tsv")
        if os.path.exists(p):
            df = pd.read_csv(p, sep="\t")
            rows.append({
                "config": cfg, "approach": f"GBT_mech7_{m}",
                "model": df["model"].iloc[0],
                "auc": df["site_auroc_oof"].iloc[0],
                "mrr": df["mrr"].iloc[0],
                "p_at_1": df["top1_acc"].iloc[0],
                "n_features": int(df["n_features"].iloc[0]),
                "n_sites": int(df["n_sites"].iloc[0]),
            })

    # Full per-bin (raw + wg)
    p = os.path.join(d, "gbt_full.summary.tsv")
    if os.path.exists(p):
        df = pd.read_csv(p, sep="\t")
        for _, r in df.iterrows():
            mode = r["feature_mode"].split("_")[-1]  # raw or wg
            rows.append({
                "config": cfg, "approach": f"GBT_full_per_bin_{mode}",
                "model": r["feature_mode"],
                "auc": r["auc_oof"], "mrr": float("nan"), "p_at_1": float("nan"),
                "n_features": int(r["n_features"]),
                "n_sites": int(r["n_sites"]),
            })

out = pd.DataFrame(rows)
out_path = os.path.join(outdir, "summary_by_config.tsv")
out.to_csv(out_path, sep="\t", index=False)

print()
print(out.to_string(index=False))
print()
print(f"[agg] wrote {out_path}")
PY_EOF

echo ""
echo "[agg] done: $(date)"