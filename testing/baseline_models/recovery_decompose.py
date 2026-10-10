#!/usr/bin/env python3
# the honest effect-size accounting for "splicing gates cpa"
import argparse
import numpy as np, pandas as pd

TEST = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL  = {"chr3","chr13"}

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def top_is_canon(g, col):
    iz = g["in_zone"].to_numpy()
    s = g[col].to_numpy()
    return iz[int(np.argmax(s))] == 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--score", default="te_prob", help="splice score column")
    ap.add_argument("--combo_weight", type=float, default=1.0)
    ap.add_argument("--split", default="test", choices=["test","val","all"])
    a = ap.parse_args()
    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr" + df["chrom"])
    if a.split == "test": df = df[df.chrom.isin(TEST)]
    elif a.split == "val": df = df[df.chrom.isin(VAL)]
    df["rf_prob"] = df["rf_prob"].fillna(0.0)
    df[a.score] = df[a.score].fillna(0.0)
    df["combo"] = logit(df.rf_prob) + a.combo_weight * logit(df[a.score])
    g_contested = {}
    for gene, g in df.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            g_contested[gene] = False; continue
        crf = float(g.rf_prob[iz == 1].max())
        g_contested[gene] = bool((g.rf_prob[iz == 0] >= crf).any())
    df["_gc"] = df["gene"].map(g_contested)
    df["gated"] = np.where(df["_gc"], df[a.score], df["rf_prob"])

    rows = []
    for gene, g in df.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            continue
        crf = float(g.rf_prob[iz == 1].max())
        contested = bool((g.rf_prob[iz == 0] >= crf).any())
        gated_col = a.score if contested else "rf_prob"
        rows.append({"gene": gene, "contested": contested,
                     "cpa": top_is_canon(g, "rf_prob"),
                     "splice": top_is_canon(g, a.score),
                     "combo": top_is_canon(g, "combo"),
                     "gated": top_is_canon(g, gated_col),
                     "n_cand": len(g),
                     "rand": float(iz.sum()) / len(g)})
    R = pd.DataFrame(rows)
    n = len(R); nc = int(R.contested.sum())
    print(f"[split={a.split}] genes {n:,} | contested {nc:,} ({nc/n:.3f}) | "
          f"non-contested {n-nc:,}\n")

    def block(name, sub):
        if len(sub) == 0:
            return
        print(f"  {name:<16} n={len(sub):<6}  "
              f"cpa={sub.cpa.mean():.3f}  splice={sub.splice.mean():.3f}  "
              f"combo={sub.combo.mean():.3f}  gated={sub.gated.mean():.3f}  "
              f"random={sub['rand'].mean():.3f}")
    print("P@1 by population (top-ranked candidate is the true canonical end):")
    block("ALL", R)
    block("CONTESTED", R[R.contested])
    block("NON-CONTESTED", R[~R.contested])

    for label, col in [("UNGATED combo", "combo"), ("GATED (deploy)", "gated")]:
        resc = R[(~R.cpa) & (R[col])]
        brok = R[(R.cpa) & (~R[col])]
        print(f"\nFlip accounting, {label} vs cpa (all genes):")
        print(f"  rescued (cpa wrong -> right): {len(resc):,}")
        print(f"  broken  (cpa right -> wrong): {len(brok):,}")
        print(f"  net = {len(resc)-len(brok):+,}  ({(len(resc)-len(brok))/n:+.3f} of all genes)")
        print(f"  of broken, {int(brok.contested.sum()):,} were contested, "
              f"{int((~brok.contested).sum()):,} non-contested")

    print(f"\nHeadline (GATED deployment):")
    print(f"  ALL genes P@1:   cpa={R.cpa.mean():.3f} -> gated={R.gated.mean():.3f}  "
          f"({R.gated.mean()-R.cpa.mean():+.3f})")
    rc = R[R.contested]
    print(f"  CONTESTED P@1:   cpa={rc.cpa.mean():.3f} -> gated={rc.gated.mean():.3f}  "
          f"(vs random {rc.gated.mean()-rc['rand'].mean():+.3f})")
    print(f"  contested true ends RESCUED by splice: {int((rc.gated & ~rc.cpa).sum()):,} of {len(rc):,}")

if __name__ == "__main__":
    main()