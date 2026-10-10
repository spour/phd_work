#!/usr/bin/env python3
# table-only leakage decomposition, feature-agnostic
import argparse
import numpy as np, pandas as pd
from scipy.stats import rankdata

TEST = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL  = {"chr3","chr13"}

def auroc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--gene_baselines", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test","all"])
    a = ap.parse_args()

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr"+df["chrom"])
    if a.split == "train":  df = df[~df.chrom.isin(TEST | VAL)]
    elif a.split == "val":  df = df[df.chrom.isin(VAL)]
    elif a.split == "test": df = df[df.chrom.isin(TEST)]

    base = (pd.read_csv(a.gene_baselines, sep="\t")
              .drop_duplicates("gene", keep="first").set_index("gene"))
    for f in a.features:
        ax = f[:-4] if f.endswith("__wg") else f
        bcol = f"baseline_{ax}"
        df[f"__base_{f}"] = df.gene.map(base[bcol]) if bcol in base.columns else np.nan
        df[f"__site_{f}"] = df[f] + df[f"__base_{f}"]
        if bcol not in base.columns:
            print(f"  [warn] {bcol} not in gene_baselines; site/baseline NA for {f}")

    canon = df[df.in_zone == 1]; intern = df[df.in_zone == 0]
    print(f"[data] split={a.split}: {df.gene.nunique()} genes, "
          f"{len(canon)} canonical, {len(intern)} internal peaks")
    print(f"\n[canonical-vs-internal AUROC, decomposed]  (AUROC>0.5 = canonical scores higher)")
    print(f"  {'feature':<22}{'wg(model)':>11}{'site':>9}{'baseline':>10}")
    for f in a.features:
        awg   = auroc(canon[f],            intern[f])
        asite = auroc(canon[f"__site_{f}"], intern[f"__site_{f}"])
        abase = auroc(canon[f"__base_{f}"], intern[f"__base_{f}"])
        print(f"  {f:<22}{awg:>11.3f}{asite:>9.3f}{abase:>10.3f}")
    print("  READ: baseline ~0.50 AND site carries the separation = local/real (claim safe).")
    print("        baseline far from 0.50 = the centered feature is keying on gene identity (leakage).")

    def cmarg(g):
        c = g.loc[g.in_zone == 1, "rf_prob"]; i = g.loc[g.in_zone == 0, "rf_prob"]
        if len(c) == 0 or len(i) == 0:
            return np.nan
        return float(c.max() - i.max())
    cm = df.groupby("gene").apply(cmarg, include_groups=False)
    contested = set(cm[cm <= 0].index)
    sub = df[df.gene.isin(contested)]
    print(f"\n[within-gene SITE win-rate on contested genes]  n_genes={len(contested):,}")
    print(f"  does the canonical's raw site value beat its OWN gene's max internal site value?")
    for f in a.features:
        wins = n = 0
        for _, g in sub.groupby("gene", sort=False):
            c = g.loc[g.in_zone == 1, f"__site_{f}"]; i = g.loc[g.in_zone == 0, f"__site_{f}"]
            c = c[np.isfinite(c)]; i = i[np.isfinite(i)]
            if len(c) == 0 or len(i) == 0:
                continue
            wins += int(c.max() > i.max()); n += 1
        print(f"  {f:<22} {wins}/{n} = {(wins/n if n else float('nan')):.3f}")

if __name__ == "__main__":
    main()