#!/usr/bin/env python3
# renders the design-decision figure showing performance across feature sets
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


KNOWN = {
    ("7_axes", "raw"): {
        "auroc": 0.7230, "auprc": 0.4840,
        "source": "GBT_mech7_raw"},
    ("7_axes", "leaky_wg"): {
        "auroc": 0.8774, "auprc": 0.7643,
        "source": "GBT_mech7_wg (per-gene-mean of labeled sites — LEAKY)"},
    ("7_axes", "gene_baseline"): {
        "auroc": 0.7472, "auprc": 0.5834,
        "source": "GBT_mech7_genebaseline"},
    ("48_per_bin", "raw"): {
        "auroc": 0.7613, "auprc": 0.5400,
        "source": "GBT_full_per_bin_raw"},
    ("48_per_bin", "leaky_wg"): {
        "auroc": 0.8685, "auprc": 0.7569,
        "source": "GBT_full_per_bin_wg (per-gene-mean — LEAKY upper bound)"},
    ("55_combined", "gene_baseline"): {
        "auroc": 0.8205, "auprc": 0.6672,
        "source": "GBT combined (48 raw + 7 gene-baselined)"},
}


def fit_gbt_oof(df, feature_cols, y, groups):
    X = df[feature_cols].values
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gbt", HistGradientBoostingClassifier(
            max_iter=500, max_depth=4, min_samples_leaf=50, learning_rate=0.05,
            class_weight="balanced", early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42)),
    ])
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), float(average_precision_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_tsv", required=True,
                     help="features_genebaselines.tsv (has raw bin features, "
                          "raw 7 axes, and gene-baselined 7 axes)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[fig] reading features")
    df = pd.read_csv(args.features_tsv, sep="\t")
    y = df["y"].values.astype(int)
    groups = df["gene"].astype(str).values

    bin_cols = [c for c in df.columns
                 if c.startswith("bin") and (c.endswith("_mean") or c.endswith("_max"))]
    wg_cols = [c for c in df.columns if c.endswith("__wg")]
    raw_axes = ["upstream_mean_u1", "upstream_max_me3", "term_3ss",
                 "local_me5", "pas_prox_u1", "dn5ss_has_strong", "interv_me5_max"]
    raw_axes = [c for c in raw_axes if c in df.columns]

    print(f"[fig]   {len(bin_cols)} per-bin, {len(wg_cols)} gene-baselined axes, "
          f"{len(raw_axes)} raw axes")

    results = dict(KNOWN)

    print(f"\n[fig] computing 48 per-bin LEAKY __wg (per-gene-mean centering)")
    df_leaky = df.copy()
    gp = df_leaky.groupby("gene", sort=False)
    for c in bin_cols:
        df_leaky[f"{c}__leaky_wg"] = df_leaky[c] - gp[c].transform("mean")
    leaky_bin_cols = [f"{c}__leaky_wg" for c in bin_cols]
    auc, ap = fit_gbt_oof(df_leaky, leaky_bin_cols, y, groups)
    results[("48_per_bin", "leaky_wg")] = {
        "auroc": auc, "auprc": ap,
        "source": "GBT 48 per-bin __wg (per-gene-mean, leaky) — proxy upper bound"}
    print(f"  AUROC={auc:.4f} AUPRC={ap:.4f}")

    print(f"\n[fig] computing 55 combined RAW (no centering anywhere)")
    combined_raw_cols = bin_cols + raw_axes
    auc, ap = fit_gbt_oof(df, combined_raw_cols, y, groups)
    results[("55_combined", "raw")] = {
        "auroc": auc, "auprc": ap,
        "source": "GBT combined (48 raw + 7 raw axes)"}
    print(f"  AUROC={auc:.4f} AUPRC={ap:.4f}")

    print(f"\n[fig] computing 55 combined LEAKY (48 raw + 7 leaky-centered)")
    df_leaky_axes = df.copy()
    for c in raw_axes:
        df_leaky_axes[f"{c}__leaky_wg"] = df_leaky_axes[c] - df_leaky_axes.groupby("gene", sort=False)[c].transform("mean")
    leaky_axes_cols = [f"{c}__leaky_wg" for c in raw_axes]
    auc, ap = fit_gbt_oof(df_leaky_axes, bin_cols + leaky_axes_cols, y, groups)
    results[("55_combined", "leaky_wg")] = {
        "auroc": auc, "auprc": ap,
        "source": "GBT combined (48 raw + 7 leaky-centered) — composition leak"}
    print(f"  AUROC={auc:.4f} AUPRC={ap:.4f}")

    results[("48_per_bin", "gene_baseline")] = {
        "auroc": np.nan, "auprc": np.nan,
        "source": "NOT computed (would require per-bin gene baselines)"}
    print(f"\n[fig] 48 per-bin gene-baselined: SKIPPED (would need per-bin baselines)")

    rows = []
    feature_sets = ["7_axes", "48_per_bin", "55_combined"]
    centerings = ["raw", "leaky_wg", "gene_baseline"]
    for fs in feature_sets:
        for cn in centerings:
            r = results.get((fs, cn), {"auroc": np.nan, "auprc": np.nan, "source": "missing"})
            rows.append({"feature_set": fs, "centering": cn,
                          "auroc": r["auroc"], "auprc": r["auprc"],
                          "source": r["source"]})
    table = pd.DataFrame(rows)
    table_path = os.path.join(args.out_dir, "centering_x_feature_set_matrix.tsv")
    table.to_csv(table_path, sep="\t", index=False)
    print(f"\n[fig] wrote {table_path}")
    print(table.to_string(index=False))

    plt.rcParams.update({"font.size": 10, "axes.spines.right": False,
                          "axes.spines.top": False})

    feature_set_labels = {
        "7_axes": "7 engineered\naxes",
        "48_per_bin": "48 per-bin\nfeatures",
        "55_combined": "55 combined\n(48 + 7)",
    }
    centering_labels = {
        "raw": "Raw\n(no centering)",
        "leaky_wg": "Per-gene mean\n(leaky)",
        "gene_baseline": "Gene baseline\n(label-free)",
    }
    centering_colors = {
        "raw": "#A6BFD9",
        "leaky_wg": "#C68C2F",
        "gene_baseline": "#3E8C68",
    }

    fig, (ax_auroc, ax_auprc) = plt.subplots(1, 2, figsize=(15, 5.5))
    n_fs = len(feature_sets); n_cn = len(centerings)
    width = 0.26
    x_base = np.arange(n_fs)

    for ax, metric, ylab in [(ax_auroc, "auroc", "AUROC"),
                              (ax_auprc, "auprc", "AUPRC")]:
        for i, cn in enumerate(centerings):
            vals = []
            hatches = []
            for fs in feature_sets:
                r = results.get((fs, cn), {"auroc": np.nan, "auprc": np.nan})
                vals.append(r[metric])
                hatches.append("///" if cn == "leaky_wg" else "")
            xs = x_base + (i - 1) * width
            bars = ax.bar(xs, vals, width, color=centering_colors[cn],
                            label=centering_labels[cn].replace("\n", " "),
                            alpha=0.92, edgecolor="white", linewidth=1.1,
                            hatch=hatches[0])
            for j, v in enumerate(vals):
                if np.isfinite(v):
                    ax.text(xs[j], v + 0.012, f"{v:.3f}",
                             ha="center", fontsize=8)
                else:
                    ax.text(xs[j], 0.02, "n/c",
                             ha="center", fontsize=8, color="#888",
                             style="italic")
        ax.set_xticks(x_base)
        ax.set_xticklabels([feature_set_labels[fs] for fs in feature_sets],
                             fontsize=10)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1)
        ax.set_title(ylab, fontweight="bold")
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.7)

    ax_auroc.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Performance by feature set × centering choice "
                  "(u3000_d1000, 5-fold GroupKFold by gene)\n"
                  "Hatched bars = leaky centering (composition leak); "
                  "n/c = not computed",
                  fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(args.out_dir, "figure_centering_x_features.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] wrote {fig_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, label in [(axes[0], "auroc", "AUROC"),
                                (axes[1], "auprc", "AUPRC")]:
        mat = np.full((n_fs, n_cn), np.nan)
        for i, fs in enumerate(feature_sets):
            for j, cn in enumerate(centerings):
                r = results.get((fs, cn))
                if r:
                    mat[i, j] = r[metric]

        im = ax.imshow(mat, cmap="viridis", vmin=0.4, vmax=0.95,
                         aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_cn))
        ax.set_xticklabels([centering_labels[c].replace("\n", " ")
                              for c in centerings], fontsize=9)
        ax.set_yticks(range(n_fs))
        ax.set_yticklabels([feature_set_labels[f].replace("\n", " ")
                              for f in feature_sets], fontsize=9)
        ax.set_title(label, fontweight="bold")
        for i in range(n_fs):
            for j in range(n_cn):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if v < 0.7 else "black"
                    leaky_mark = "*" if centerings[j] == "leaky_wg" else ""
                    ax.text(j, i, f"{v:.3f}{leaky_mark}",
                             ha="center", va="center", color=color, fontsize=11,
                             fontweight="bold")
                else:
                    ax.text(j, i, "n/c", ha="center", va="center",
                             color="white", fontsize=10, style="italic")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Feature set × centering matrix  "
                  "(* = leaky, not genome-wide-usable)",
                  fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    heatmap_path = os.path.join(args.out_dir, "figure_centering_heatmap.png")
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {heatmap_path}")

    print(f"\n[fig] done.")


if __name__ == "__main__":
    main()