#!/usr/bin/env bash
# leave-one-out ablation on the 7-axis splicing gbt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FEATURES_TSV="${FEATURES_TSV:?must set FEATURES_TSV}"
OUTDIR="${OUTDIR:?must set OUTDIR}"
MODE="${MODE:-wg}"   # 'wg' or 'raw' — should match the panel you defended in fig 1
GBT_SCRIPT="${GBT_SCRIPT:-$SCRIPT_DIR/run_gbt_7axis.py}"

mkdir -p "$OUTDIR"

AXES=(
    "upstream_mean_u1"
    "upstream_max_me3"
    "term_3ss"
    "local_me5"
    "pas_prox_u1"
    "dn5ss_has_strong"
    "interv_me5_max"
)
DISPLAYS=(
    "Mean upstream U1 affinity"
    "Best upstream 3SS (MaxEntScan)"
    "Terminal 3SS strength"
    "Local 5SS strength (MaxEntScan)"
    "PAS-proximal U1 affinity"
    "Strong downstream 5SS present"
    "Intervening 5SS in terminal exon"
)

suffix_axes() {
    local mode="$1"; shift
    local arr=("$@")
    if [[ "$mode" == "wg" ]]; then
        for i in "${!arr[@]}"; do
            arr[$i]="${arr[$i]}__wg"
        done
    fi
    printf "%s\n" "${arr[@]}"
}

mapfile -t AXES_SUFFIXED < <(suffix_axes "$MODE" "${AXES[@]}")

echo ""
echo "=========================================="
echo "Baseline: full 7-axis panel (mode=$MODE)"
echo "=========================================="
python3 "$GBT_SCRIPT" \
    --mode "$MODE" \
    --features_tsv "$FEATURES_TSV" \
    --out_prefix "$OUTDIR/baseline" \
    --no_shap

WORK_TSV="$OUTDIR/_ablation_features.tsv"
python3 - "$FEATURES_TSV" "$WORK_TSV" << 'PY_EOF'
import sys, pandas as pd
inp, outp = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp, sep="\t")
df["_dummy"] = 0.0
df.to_csv(outp, sep="\t", index=False)
print(f"[ablate] wrote {outp} (added constant '_dummy' column)")
PY_EOF

for i in "${!AXES[@]}"; do
    axis="${AXES[$i]}"
    display="${DISPLAYS[$i]}"
    axis_suffixed="${AXES_SUFFIXED[$i]}"

    keep_axes=()
    keep_displays=()
    for j in "${!AXES_SUFFIXED[@]}"; do
        if [[ "$j" != "$i" ]]; then
            keep_axes+=("${AXES_SUFFIXED[$j]}")
            keep_displays+=("${DISPLAYS[$j]}")
        fi
    done
    keep_axes+=("_dummy")
    keep_displays+=("(dummy, dropped)")

    features_override=$(IFS=,; echo "${keep_axes[*]}")
    display_override=$(IFS=,; echo "${keep_displays[*]}")

    echo ""
    echo "=========================================="
    echo "Drop axis: $axis_suffixed  ($display)"
    echo "=========================================="
    python3 "$GBT_SCRIPT" \
        --mode "$MODE" \
        --features_tsv "$WORK_TSV" \
        --out_prefix "$OUTDIR/drop_${axis}" \
        --no_shap \
        --features_override "$features_override" \
        --display_override "$display_override"
done

echo ""
echo "=========================================="
echo "Aggregating ablation table"
echo "=========================================="
python3 - "$OUTDIR" << 'PY_EOF'
import os, sys, glob, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

outdir = sys.argv[1]
baseline_path = os.path.join(outdir, "baseline.summary.tsv")
if not os.path.exists(baseline_path):
    sys.exit(f"baseline.summary.tsv missing in {outdir}")

baseline = pd.read_csv(baseline_path, sep="\t").iloc[0]
b_auroc = float(baseline["site_auroc_oof"])
b_auprc = float(baseline["site_auprc_oof"])
b_mrr   = float(baseline["mrr"])
b_p1    = float(baseline["top1_acc"])

rows = [{
    "config": "baseline_full_7axis",
    "axis_dropped": "(none)",
    "display": "(none)",
    "auroc": b_auroc, "delta_auroc": 0.0,
    "auprc": b_auprc, "delta_auprc": 0.0,
    "mrr":   b_mrr,   "delta_mrr":   0.0,
    "p_at_1": b_p1,   "delta_p_at_1": 0.0,
}]

drop_files = sorted(glob.glob(os.path.join(outdir, "drop_*.summary.tsv")))
for p in drop_files:
    axis = os.path.basename(p).replace("drop_", "").replace(".summary.tsv", "")
    df = pd.read_csv(p, sep="\t").iloc[0]
    rows.append({
        "config": f"drop_{axis}",
        "axis_dropped": axis,
        "display": axis,  # display name from the script's FEAT_DISPLAY lookup
        "auroc": float(df["site_auroc_oof"]),
        "delta_auroc": float(df["site_auroc_oof"]) - b_auroc,
        "auprc": float(df["site_auprc_oof"]),
        "delta_auprc": float(df["site_auprc_oof"]) - b_auprc,
        "mrr":   float(df["mrr"]),
        "delta_mrr":   float(df["mrr"]) - b_mrr,
        "p_at_1": float(df["top1_acc"]),
        "delta_p_at_1": float(df["top1_acc"]) - b_p1,
    })

out = pd.DataFrame(rows)
out_path = os.path.join(outdir, "ablation_table.tsv")
out.to_csv(out_path, sep="\t", index=False)
print(f"[ablate] wrote {out_path}")
print(out.to_string(index=False))

# Bar plot of AUROC deltas
sub = out[out["axis_dropped"] != "(none)"].copy()
sub = sub.sort_values("delta_auroc")
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#C23B3B" if v < 0 else "#2E8B57" for v in sub["delta_auroc"]]
y = np.arange(len(sub))
ax.barh(y, sub["delta_auroc"], color=colors, alpha=0.85, edgecolor="white")
for yi, v in zip(y, sub["delta_auroc"]):
    txt = f"{v:+.4f}"
    dx = 0.0002 if v >= 0 else -0.0002
    ha = "left" if v >= 0 else "right"
    ax.text(v + dx, yi, txt, ha=ha, va="center", fontsize=9, fontweight="bold")
ax.axvline(0, color="#444", linewidth=1.0)
ax.set_yticks(y)
ax.set_yticklabels(sub["axis_dropped"].values, fontsize=10)
ax.set_xlabel(f"Delta AUROC vs full 7-axis baseline ({b_auroc:.4f})")
ax.set_title("Leave-one-out ablation: which axes are pulling weight?",
             fontsize=12, fontweight="bold", loc="left")
ax.grid(axis="x", color="#EEEEEE", linewidth=0.7)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
png_path = os.path.join(outdir, "ablation_bar.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[ablate] wrote {png_path}")
PY_EOF

echo ""
echo "[ablate] Done. Inspect $OUTDIR/ablation_table.tsv and ablation_bar.png"
echo ""
echo "Interpretation guide:"
echo "  Large negative delta_auroc when an axis is dropped  -> that axis is essential"
echo "  Near-zero delta_auroc when an axis is dropped       -> that axis is redundant"
echo "  Positive delta_auroc when an axis is dropped        -> that axis was hurting (rare)"
echo ""
echo "Define the minimal set as the smallest subset where dropping any included axis"
echo "loses >= 0.005 AUROC. Typically: 6-7 of the original 7 axes."