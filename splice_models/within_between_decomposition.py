#!/usr/bin/env python3
# show why raw splicing features have the wrong mechanistic sign and centered
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-9
TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}

AXES = {
    "pas_prox_u1":      -1,
    "local_me5":        -1,
    "upstream_mean_u1": -1,
    "interv_me5_max":   -1,
    "dn5ss_has_strong": -1,
    "term_3ss":         +1,
    "upstream_max_me3": +1,
}


def norm_chrom(c):
    if not isinstance(c, str):
        return c
    c = c.strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def slope(y, x):
    ok = np.isfinite(x)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return np.nan
    xs = (x[ok] - x[ok].mean()) / (x[ok].std() + EPS)
    clf = LogisticRegression(C=1e6, max_iter=2000).fit(xs.reshape(-1, 1), y[ok])
    return float(clf.coef_[0, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--split", default="test", choices=["test", "all"])
    args = ap.parse_args()

    df = pd.read_csv(args.features_tsv, sep="\t")
    if args.split == "test":
        df = df[df["chrom"].astype(str).apply(norm_chrom).isin(TEST_CHROMS)].copy()
    y = df["y"].to_numpy().astype(int)
    gene = df["gene"].to_numpy()

    rows = []
    for axis, exp_sign in AXES.items():
        if axis not in df.columns:
            print(f"[wb] axis not found, skipping: {axis}", file=sys.stderr)
            continue
        x = df[axis].to_numpy(float)
        gm = df.groupby("gene")[axis].transform("mean").to_numpy(float)
        within = x - gm
        wg_col = f"{axis}__wg"
        wg = df[wg_col].to_numpy(float) if wg_col in df.columns else np.full(len(df), np.nan)

        rows.append({
            "axis": axis, "expected_within_sign": exp_sign,
            "pooled": round(slope(y, x), 3),
            "between": round(slope(y, gm), 3),
            "within": round(slope(y, within), 3),
            "wg_random_baseline": round(slope(y, wg), 3) if np.isfinite(wg).any() else np.nan,
        })

    tab = pd.DataFrame(rows)
    tab["sign_flip_pooled_to_within"] = (
        np.sign(tab["pooled"]) != np.sign(tab["within"])).where(tab["within"].notna())
    tab["within_matches_mechanism"] = np.sign(tab["within"]) == tab["expected_within_sign"]
    tab.to_csv(f"{args.out_prefix}_decomposition.tsv", sep="\t", index=False)
    print(tab.to_string(index=False), file=sys.stderr)
    n_flip = int(tab["sign_flip_pooled_to_within"].sum())
    n_match = int(tab["within_matches_mechanism"].sum())
    print(f"\n[wb] {n_flip}/{len(tab)} axes flip sign pooled->within; "
          f"{n_match}/{len(tab)} within-effects match the mechanism", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(9, 5))
    yv = np.arange(len(tab))
    ax.scatter(tab["pooled"], yv - 0.15, color="#C23B3B", label="pooled (raw, confounded)", zorder=3)
    ax.scatter(tab["within"], yv + 0.15, color="#1F77B4", label="within-gene (centered)", zorder=3)
    for i, r in tab.iterrows():
        ax.plot([r["pooled"], r["within"]], [i - 0.15, i + 0.15], color="#BBB", lw=1, zorder=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(yv)
    ax.set_yticklabels(tab["axis"])
    ax.set_xlabel("standardized logistic slope on canonical label")
    ax.set_title("Raw vs within-gene effect per axis\nsign flip = between-gene confound removed by centering")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_decomposition.png", dpi=200)
    print(f"\n[wb] wrote {args.out_prefix}_decomposition.tsv and _decomposition.png", file=sys.stderr)


if __name__ == "__main__":
    main()