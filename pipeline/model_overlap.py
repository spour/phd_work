#!/usr/bin/env python3
# do rf, lr, cnn independently flag the same internal sites as beating the canonical
import argparse, os, sys, glob
import numpy as np, pandas as pd
from itertools import combinations

MODELS = ["rf_prob", "lr_prob", "cnn_prob"]
MNAME  = {"rf_prob": "RF", "lr_prob": "LR", "cnn_prob": "CNN"}

def load_canonical(bed):
    can = {}
    for line in open(bed):
        f = line.rstrip("\n").split("\t")
        can.setdefault(f[3], []).append((int(f[1]), int(f[2])))
    return can

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--canonical_bed", required=True)
    ap.add_argument("--tol", type=int, default=250, help="same-interval tolerance (nt)")
    ap.add_argument("--out", default="model_overlap")
    a = ap.parse_args()

    can = load_canonical(a.canonical_bed)
    files = glob.glob(os.path.join(a.scan_dir, "*.parquet"))
    print(f"[data] {len(files):,} scanned genes", file=sys.stderr)

    contested_by = {m: set() for m in MODELS}
    top_internal = []
    n_any = 0

    for p in files:
        g = os.path.basename(p)[:-8]
        if g not in can: continue
        d = pd.read_parquet(p, columns=["window_start_genomic"] + MODELS + ["is_constitutive_cpa"])
        if not d["is_constitutive_cpa"].any(): continue
        ws = d["window_start_genomic"].to_numpy()
        center = ws + 250
        isc = d["is_constitutive_cpa"].to_numpy()
        canon_center = int(center[isc][0])
        internal_mask = np.abs(center - canon_center) > a.tol
        if not internal_mask.any(): continue
        rec = {}
        any_model = False
        for m in MODELS:
            s = d[m].to_numpy()
            canon_score = s[isc][0]
            si = s.copy(); si[~internal_mask] = -np.inf
            top_i = int(np.argmax(si))
            if si[top_i] >= canon_score:
                contested_by[m].add(g)
                rec[m] = int(center[top_i])
                any_model = True
        if any_model:
            n_any += 1
            top_internal.append((g, rec))

    # (1) venn regions
    R, L, C = contested_by["rf_prob"], contested_by["lr_prob"], contested_by["cnn_prob"]
    regions = {
        "RF only":  len(R - L - C),
        "LR only":  len(L - R - C),
        "CNN only": len(C - R - L),
        "RF&LR":    len((R & L) - C),
        "RF&CNN":   len((R & C) - L),
        "LR&CNN":   len((L & C) - R),
        "all three":len(R & L & C),
    }
    pd.DataFrame([dict(region=k, n=v) for k, v in regions.items()]).to_csv(
        f"{a.out}_venn.tsv", sep="\t", index=False)
    total_contested = len(R | L | C)
    print(f"\n=== gene-level: models flagging an internal site > canonical ===", file=sys.stderr)
    for m in MODELS: print(f"  {MNAME[m]:<4} {len(contested_by[m]):,} genes", file=sys.stderr)
    print(f"  union {total_contested:,}; all-three {regions['all three']:,} "
          f"({regions['all three']/max(1,total_contested):.1%} of union)", file=sys.stderr)

    # (2) interval agreement among genes where all 3 are contested
    agree_rows = []; dist_rows = []
    all3 = 0
    PAIRS = [("rf_prob","lr_prob","RF-LR"), ("rf_prob","cnn_prob","RF-CNN"),
             ("lr_prob","cnn_prob","LR-CNN")]
    for g, rec in top_internal:
        if len(rec) < 3:
            continue
        all3 += 1
        cs = [rec["rf_prob"], rec["lr_prob"], rec["cnn_prob"]]
        pw = [abs(cs[i]-cs[j]) <= a.tol for i,j in combinations(range(3),2)]
        if all(pw): cat = "all 3 same site"
        elif any(pw): cat = "2 of 3 same"
        else: cat = "all different"
        agree_rows.append(dict(gene=g, category=cat))
        for m1, m2, name in PAIRS:
            dist_rows.append(dict(gene=g, pair=name, dist=abs(rec[m1]-rec[m2])))
    A = pd.DataFrame(agree_rows)
    if len(A):
        agg = A.category.value_counts().reindex(
            ["all 3 same site","2 of 3 same","all different"]).fillna(0).astype(int)
        agg_df = pd.DataFrame(dict(category=agg.index, n=agg.values,
                                   frac=(agg.values/agg.values.sum())))
        agg_df.to_csv(f"{a.out}_agreement.tsv", sep="\t", index=False)
        print(f"\n=== interval agreement (genes contested in ALL 3 models, n={all3:,}) ===", file=sys.stderr)
        for _, r in agg_df.iterrows():
            print(f"  {r['category']:<16} {int(r['n']):,} ({r['frac']:.1%})", file=sys.stderr)
    if dist_rows:
        DD = pd.DataFrame(dist_rows)
        DD.to_csv(f"{a.out}_pairdist.tsv", sep="\t", index=False)
        print(f"\n=== pairwise distance between models' top internal picks (n={all3:,} genes) ===", file=sys.stderr)
        for name, gg in DD.groupby("pair"):
            within = (gg.dist <= a.tol).mean()
            print(f"  {name:<7} median {gg.dist.median():>7,.0f} nt  "
                  f"within +/-{a.tol}nt: {within:.1%}", file=sys.stderr)
        print(f"[out] {a.out}_venn.tsv, {a.out}_agreement.tsv, {a.out}_pairdist.tsv", file=sys.stderr)
    else:
        print(f"\n[out] {a.out}_venn.tsv, {a.out}_agreement.tsv", file=sys.stderr)

if __name__ == "__main__":
    main()