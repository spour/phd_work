#!/usr/bin/env python3
# canonical (y=1) vs cryptic (y=0) for every centered feature, sorted by effect size
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN = {f"chr{c}" for c in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]}
VAL = {"chr3", "chr13"}
TEST = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
RAW_DISTANCE = ["all_u1_decay_up_tau150", "all_u1_decay_up_tau400",
                "all_me3_decay_up_tau150", "all_me3_decay_up_tau400",
                "term_me3_top3mean_500_25", "dn5ss_best_score", "dn5ss_n_strong",
                "interv_me5_max_500", "interv_me5_n_strong_500"]


def norm(c):
    c = str(c).strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2) + 1e-12)
    return (a.mean() - b.mean()) / sp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--include_raw", action="store_true")
    ap.add_argument("--split", default="test", choices=["all", "train", "val", "test"])
    ap.add_argument("--ncols", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.features_tsv, sep="\t")
    df["c"] = df["chrom"].apply(norm)
    if args.split != "all":
        df = df[df["c"].isin({"train": TRAIN, "val": VAL, "test": TEST}[args.split])]

    if args.features:
        feats = [f for f in args.features if f in df.columns]
    else:
        feats = [c for c in df.columns if c.endswith("__wg")]
        if args.include_raw:
            feats += [c for c in RAW_DISTANCE if c in df.columns and c not in feats]
    if not feats:
        sys.exit("no features to plot")

    rows = []
    for f in feats:
        sub = df[["y", f]].dropna()
        if sub["y"].nunique() < 2:
            continue
        canon = sub.loc[sub["y"] == 1, f].to_numpy()
        cryptic = sub.loc[sub["y"] == 0, f].to_numpy()
        try:
            auc = roc_auc_score(sub["y"].to_numpy(), sub[f].to_numpy())
        except ValueError:
            auc = np.nan
        try:
            mw = mannwhitneyu(canon, cryptic, alternative="two-sided").pvalue
        except ValueError:
            mw = np.nan
        rows.append({"feature": f, "n_canon": len(canon), "n_cryptic": len(cryptic),
                     "median_canon": round(float(np.median(canon)), 4),
                     "median_cryptic": round(float(np.median(cryptic)), 4),
                     "cohens_d": round(float(cohens_d(canon, cryptic)), 4),
                     "auroc_single": round(float(auc), 4),
                     "effect": round(abs(float(auc) - 0.5), 4),
                     "is_raw": not f.endswith("__wg"), "mannwhitney_p": mw})
    st = pd.DataFrame(rows).sort_values("effect", ascending=False).reset_index(drop=True)
    st.to_csv(f"{args.out_prefix}_stats.tsv", sep="\t", index=False)
    print(st.drop(columns=["mannwhitney_p"]).to_string(index=False), file=sys.stderr)

    ordered = st["feature"].tolist()
    ncols = args.ncols
    nrows = int(np.ceil(len(ordered) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, ordered):
        sub = df[["y", f]].dropna()
        canon = sub.loc[sub["y"] == 1, f].to_numpy()
        cryptic = sub.loc[sub["y"] == 0, f].to_numpy()
        lo, hi = np.percentile(np.concatenate([canon, cryptic]), [1, 99])
        bins = np.linspace(lo, hi, 50) if hi > lo else 20
        ax.hist(cryptic, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color="#C23B3B")
        ax.hist(canon, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color="#1F77B4")
        r = st[st["feature"] == f].iloc[0]
        tag = " [raw]" if r["is_raw"] else ""
        ax.set_title(f"{f}{tag}\nAUROC={r['auroc_single']:.3f} d={r['cohens_d']:+.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[len(ordered):]:
        ax.axis("off")
    axes[0].plot([], [], color="#1F77B4", label="canonical"); axes[0].plot([], [], color="#C23B3B", label="cryptic")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(f"Canonical vs cryptic by feature, {args.split} (sorted by effect size)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{args.out_prefix}_distributions.png", dpi=170)

    corr = df[ordered].corr(method="spearman")
    corr.to_csv(f"{args.out_prefix}_corr.tsv", sep="\t")
    fig2, ax2 = plt.subplots(figsize=(0.5 * len(ordered) + 3, 0.5 * len(ordered) + 3))
    im = ax2.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax2.set_xticks(range(len(ordered))); ax2.set_xticklabels(ordered, rotation=90, fontsize=6)
    ax2.set_yticks(range(len(ordered))); ax2.set_yticklabels(ordered, fontsize=6)
    for i in range(len(ordered)):
        for j in range(len(ordered)):
            v = corr.iloc[i, j]
            if i != j and abs(v) >= 0.8:
                ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig2.colorbar(im, ax=ax2, shrink=0.7, label="Spearman r")
    ax2.set_title("Feature correlation (|r|>=0.8 labeled = redundant)", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(f"{args.out_prefix}_corr.png", dpi=170)

    print(f"\n[dist] wrote {args.out_prefix}_distributions.png, _corr.png, _stats.tsv, _corr.tsv", file=sys.stderr)
    redundant = [(ordered[i], ordered[j], round(corr.iloc[i, j], 2))
                 for i in range(len(ordered)) for j in range(i+1, len(ordered))
                 if abs(corr.iloc[i, j]) >= 0.8]
    if redundant:
        print("[dist] redundant pairs (|r|>=0.8):", file=sys.stderr)
        for a, b, r in redundant:
            print(f"    {a}  ~  {b}   r={r}", file=sys.stderr)


if __name__ == "__main__":
    main()