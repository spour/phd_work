#!/usr/bin/env python3
# genome-wide biological characterization
import argparse, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LEN_BINS = [0, 5000, 20000, 50000, 100000, 500000, 5e8]
LEN_LABELS = ["<5kb", "5-20kb", "20-50kb", "50-100kb", "100-500kb", ">500kb"]
DEFAULT_FEATS = ["term_3ss__wg", "term_ppt__wg", "upstream_max_me3__wg"]

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, max(0.0, c-h), min(1.0, c+h)

def loco_oof(df, feats, label="in_zone"):
    sp = np.full(len(df), np.nan)
    X = df[feats].to_numpy(); y = df[label].astype(int).to_numpy()
    chrom = df["chrom"].to_numpy()
    for c in sorted(df.chrom.unique()):
        tr = chrom != c; te = ~tr
        if te.sum() == 0 or tr.sum() == 0 or y[tr].sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        sp[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return sp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATS)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="internal is a competitor if rf >= canonical_rf - margin")
    ap.add_argument("--permute", action="store_true",
                    help="shuffle splice_prob within each gene (feature-side null; "
                         "should collapse splice recovery onto random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", default="genomewide_recovery")
    a = ap.parse_args()

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr"+df["chrom"])
    miss = [f for f in a.features if f not in df.columns]
    if miss:
        raise SystemExit(f"[err] features not in table: {miss}")
    df[a.features] = df[a.features].fillna(0.0)
    df = df.reset_index(drop=True)

    print(f"[data] {len(df):,} peaks, {df.gene.nunique():,} genes, {df.chrom.nunique()} chroms")
    print(f"[features] {a.features}")
    print(f"[loco] fitting leave-one-chromosome-out logistic over {df.chrom.nunique()} chroms ...")
    df["splice_prob"] = loco_oof(df, a.features)
    if a.permute:
        print(f"[permute] within-gene shuffle of splice_prob ON (feature-side null)")

    rng = np.random.default_rng(a.seed)
    rows = []
    for gene, g in df.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            continue
        rf = g["rf_prob"].to_numpy(); sp = g["splice_prob"].to_numpy()
        gl = float(g["gene_length"].iloc[0])
        crf = float(rf[iz == 1].max())
        cand = np.where((iz == 1) | (rf >= crf - a.margin))[0]
        csize = len(cand)
        contested = int(csize > 1)
        spc = sp[cand].copy()
        if a.permute and csize > 1:
            spc = spc[rng.permutation(csize)]
        win = cand[int(np.argmax(spc))]
        rows.append((gene, gl, csize, contested,
                     int(iz[int(np.argmax(rf))] == 1),
                     int(iz[win] == 1),
                     float(iz[cand].sum())/csize))
    G = pd.DataFrame(rows, columns=["gene", "gene_length", "contest_size", "contested",
                                    "rf_hit", "splice_hit", "rand_hit"])
    G["lb"] = pd.cut(G.gene_length, LEN_BINS, labels=LEN_LABELS)
    G.to_csv(f"{a.out_prefix}_pergene.tsv", sep="\t", index=False)

    N = len(G); nC = int(G.contested.sum()); nNC = N - nC
    print(f"\n[genome-wide, all scorable genes] N = {N:,}")
    print(f"  contested (an internal RF-competitor exists): {nC:,}  ({nC/N:.3f})")
    print(f"  non-contested (canonical strictly top RF)   : {nNC:,}  ({nNC/N:.3f})")
    print(f"\n[why pooled 'random' looks high -- it is the trivial genes]")
    print(f"  non-contested genes have candidate set = {{canonical}}, so random = 1.000 there.")
    print(f"  pooled random = {nNC/N:.3f}(non-contested, =1.0) + {nC/N:.3f}(contested) blend.")
    print(f"  The ONLY splice-over-random signal lives in the contested set below.")

    C = G[G.contested == 1]
    r_rand = float(C.rand_hit.mean())
    k_spl = int(C.splice_hit.sum()); p_spl, lo_spl, hi_spl = wilson(k_spl, len(C))
    d = C.splice_hit.to_numpy() - C.rand_hit.to_numpy()
    B = 4000; bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    print(f"\n[CONTESTED genes -- THE biological result] n = {len(C):,}")
    print(f"  RF-argmax recovery : {C.rf_hit.mean():.3f}   (0 by construction)")
    print(f"  random tie-break   : {r_rand:.3f}")
    print(f"  splice recovery    : {p_spl:.3f}   [{lo_spl:.3f}, {hi_spl:.3f}]")
    print(f"  splice - random    : {d.mean():+.3f}   "
          f"paired CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")

    print(f"\n[overall, all scorable genes -- performance frame, mixes both populations]")
    print(f"  rf={G.rf_hit.mean():.3f}  random={G.rand_hit.mean():.3f}  "
          f"splice={G.splice_hit.mean():.3f}  spl-rnd={G.splice_hit.mean()-G.rand_hit.mean():+.3f}")

    print(f"\n[CONTESTED recovery by gene length, vs random]")
    print(f"  {'length':<10}{'n':>6}{'random':>9}{'splice':>9}{'spl-rnd':>9}{'95% CI':>18}")
    lr = []
    for b in LEN_LABELS:
        gb = C[C.lb == b]
        if len(gb) == 0:
            continue
        rr = float(gb.rand_hit.mean()); k = int(gb.splice_hit.sum())
        p, lo, hi = wilson(k, len(gb))
        print(f"  {b:<10}{len(gb):>6}{rr:>9.3f}{p:>9.3f}{p-rr:>+9.3f}   [{lo:.3f}, {hi:.3f}]")
        lr.append({"length_bin": b, "n": len(gb), "random": rr, "splice": p,
                   "spl_minus_rnd": p-rr, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(lr).to_csv(f"{a.out_prefix}_by_length.tsv", sep="\t", index=False)
    print(f"\n[out] {a.out_prefix}_pergene.tsv, {a.out_prefix}_by_length.tsv")

if __name__ == "__main__":
    main()