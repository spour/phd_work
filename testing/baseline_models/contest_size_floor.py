#!/usr/bin/env python3
# reports shortlist ("contest") size statistics and the exact random-pick p@1 floor
import argparse, sys
import numpy as np, pandas as pd

def sizes_from_feat(paths):
    dfs=[pd.read_parquet(p) for p in paths]
    df=pd.concat(dfs, ignore_index=True)
    n=df.groupby("gene").size().to_numpy()
    return n

def sizes_from_summary(path):
    S=pd.read_csv(path, sep="\t")
    if "n_shortlist" in S.columns:
        n=S["n_shortlist"].to_numpy()
    elif "n_internal" in S.columns:
        n=(S["n_internal"].to_numpy()+1)
    else:
        sys.exit("[err] summary needs n_shortlist or n_internal")
    return n.astype(int)

def describe(n, label):
    n=np.asarray(n, float); n=n[n>=1]
    floor_exact = float(np.mean(1.0/n))
    floor_naive = 1.0/float(np.mean(n))
    print(f"\n=== {label} (n_genes={len(n):,}) ===", file=sys.stderr)
    print(f"  shortlist size: mean={n.mean():.2f}  median={np.median(n):.0f}  "
          f"min={n.min():.0f}  max={n.max():.0f}  "
          f"q25={np.percentile(n,25):.0f}  q75={np.percentile(n,75):.0f}", file=sys.stderr)
    print(f"  random P@1 floor (mean of 1/n)  = {floor_exact:.4f}   <-- use this", file=sys.stderr)
    print(f"  random P@1 if you used 1/mean(n) = {floor_naive:.4f}   (biased; Jensen gap "
          f"{floor_exact-floor_naive:+.4f})", file=sys.stderr)
    return dict(scope=label, n_genes=len(n), mean_size=n.mean(), median_size=float(np.median(n)),
                min_size=float(n.min()), max_size=float(n.max()),
                random_floor_exact=floor_exact, random_floor_naive=floor_naive)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feat", nargs="*", default=None, help="featurized peak parquet(s)")
    ap.add_argument("--summary", default=None, help="shortlist_summary.tsv")
    ap.add_argument("--out", default="contest_size_floor")
    a=ap.parse_args()
    if not a.feat and not a.summary:
        sys.exit("[err] pass --feat <parquet...> and/or --summary <tsv>")

    rows=[]
    if a.summary:
        n_all=sizes_from_summary(a.summary)
        rows.append(describe(n_all, "ALL genes (incl uncontested)"))
        rows.append(describe(n_all[n_all>=2], "CONTESTED only (n>=2)"))
    if a.feat:
        n_feat=sizes_from_feat(a.feat)
        rows.append(describe(n_feat, "FEATURIZED peaks, all genes present"))
        rows.append(describe(n_feat[n_feat>=2], "FEATURIZED, contested (n>=2)"))

    pd.DataFrame(rows).to_csv(f"{a.out}.tsv", sep="\t", index=False)
    print(f"\n[out] {a.out}.tsv", file=sys.stderr)

if __name__=="__main__": main()
