#!/usr/bin/env python3
# p@1 (does the top-ranked candidate equal the true canonical end) for three rankers
import argparse
import numpy as np, pandas as pd

TEST = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL  = {"chr3","chr13"}

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def p_at_1(df, score_col, genes):
    hit = n = 0
    sub = df[df.gene.isin(genes)]
    for _, g in sub.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            continue
        s = g[score_col].to_numpy()
        top = int(np.argmax(s))
        hit += int(iz[top] == 1); n += 1
    return hit, n, (hit / n if n else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--split", default="test", choices=["test","val","trainval_excluded","all"])
    a = ap.parse_args()
    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr" + df["chrom"])
    if a.split == "test": df = df[df.chrom.isin(TEST)]
    elif a.split == "val": df = df[df.chrom.isin(VAL)]
    elif a.split == "trainval_excluded": df = df[df.chrom.isin(TEST)]
    for c in ("rf_prob", "te_prob"):
        df[c] = df[c].fillna(0.0)
    df["logit_sum"] = logit(df.rf_prob) + logit(df.te_prob)
    df["product"]   = df.rf_prob * df.te_prob

    contested = set()
    allg = set()
    for gene, g in df.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            continue
        allg.add(gene)
        crf = float(g.rf_prob[iz == 1].max())
        if (g.rf_prob[iz == 0] >= crf).any():
            contested.add(gene)

    print(f"[split={a.split}] genes scored {len(allg):,}  contested {len(contested):,} "
          f"({len(contested)/max(len(allg),1):.3f})\n")
    rankers = [("cpa_alone","rf_prob"), ("splice_alone","te_prob"),
               ("logit_sum","logit_sum"), ("product","product")]

    print(f"{'ranker':<14}{'P@1 all':>12}{'P@1 contested':>16}")
    for name, col in rankers:
        _, _, pa = p_at_1(df, col, allg)
        _, _, pc = p_at_1(df, col, contested)
        print(f"{name:<14}{pa:>12.3f}{pc:>16.3f}")
    print("\nread: P@1 all is the system number (dominated by easy genes).")
    print("      P@1 contested is where splice can add; cpa_alone there is ~0 by definition.")

if __name__ == "__main__":
    main()