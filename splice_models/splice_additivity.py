#!/usr/bin/env python3
# evidence that splicing features gate on top of the cpa grammar rather than redundantly
import argparse, math
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PRIOR = {"term_3ss__wg": +1, "term_ppt__wg": +1, "upstream_max_me3__wg": +1,
         "upstream_mean_u1__wg": -1, "pas_prox_u1__wg": -1}

def auroc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return (np.nan, 0, 0)
    r = rankdata(np.concatenate([pos, neg]))
    return (float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))),
            len(pos), len(neg))

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, max(0.0, c-h), min(1.0, c+h)

def loco(df, feats):
    sp = np.full(len(df), np.nan)
    X = df[feats].to_numpy(); y = df["in_zone"].astype(int).to_numpy()
    ch = df["chrom"].to_numpy()
    for c in sorted(df.chrom.unique()):
        tr = ch != c; te = ~tr
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
    ap.add_argument("--features", nargs="+",
                    default=["term_3ss__wg", "term_ppt__wg", "upstream_max_me3__wg"])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--out_prefix", default="splice_additivity")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr"+df["chrom"])
    df[a.features] = df[a.features].fillna(0.0)
    df = df.reset_index(drop=True)
    print(f"[data] {len(df):,} peaks, {df.gene.nunique():,} genes")
    print(f"[features] {a.features}")
    print(f"[loco] fitting leave-one-chromosome-out logistic over {df.chrom.nunique()} chroms ...")
    df["splice_prob"] = loco(df, a.features)

    # coefficients + directions (fit once on all peaks, for the readout)
    sc = StandardScaler().fit(df[a.features].to_numpy())
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(
        sc.transform(df[a.features].to_numpy()), df["in_zone"].astype(int).to_numpy())
    print(f"\n[standardized logistic coefficients and direction vs mechanistic prior]")
    for f, w in zip(a.features, clf.coef_[0]):
        pr = PRIOR.get(f)
        ok = "n/a" if pr is None else ("ok" if np.sign(w) == pr else "INVERTED")
        print(f"  {f:<22} coef={w:+.3f}  prior={pr}  {ok}")

    # panel 1: within-rf-stratum splice auroc
    print(f"\n[Panel 1: splice canonical-vs-internal AUROC WITHIN rf bins]")
    print(f"  >0.5 in the HIGH-rf bins = splice adds discrimination RF lacks (on top of).")
    print(f"  (only trust this if the leakage check shows the signal is local.)")
    edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    print(f"  {'rf_bin':<12}{'n_canon':>9}{'n_intern':>10}{'splice_AUROC':>14}")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df.rf_prob >= lo) & (df.rf_prob < hi)
        sub = df[m]
        au, nc, ni = auroc(sub[sub.in_zone == 1]["splice_prob"],
                           sub[sub.in_zone == 0]["splice_prob"])
        if nc == 0 or ni == 0:
            continue
        print(f"  [{lo:.1f},{hi:.1f}){'':<3}{nc:>9d}{ni:>10d}{au:>14.3f}")
        rows.append({"rf_lo": lo, "rf_hi": hi, "n_canon": nc, "n_internal": ni,
                     "splice_auroc": au})
    pd.DataFrame(rows).to_csv(f"{a.out_prefix}_rf_stratum_auroc.tsv", sep="\t", index=False)

    # contested set, margin-stratified recovery
    rec = []
    for gene, g in df.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            continue
        rf = g["rf_prob"].to_numpy(); sp = g["splice_prob"].to_numpy()
        gl = float(g["gene_length"].iloc[0])
        crf = float(rf[iz == 1].max())
        cand = np.where((iz == 1) | (rf >= crf - a.margin))[0]
        if len(cand) <= 1:
            continue
        gap = float(rf[cand].max() - crf)
        win = cand[int(np.argmax(sp[cand]))]
        rec.append((gene, gl, len(cand), gap, int(iz[win] == 1), float(iz[cand].sum())/len(cand)))
    C = pd.DataFrame(rec, columns=["gene", "gene_length", "contest_size",
                                   "gap", "splice_hit", "rand_hit"])
    C.to_csv(f"{a.out_prefix}_contested_pergene.tsv", sep="\t", index=False)

    print(f"\n[Panel 2: contested recovery stratified by RF contest gap]  n={len(C):,}")
    print(f"  gap = top_internal_rf - canonical_rf (>=0). Mechanism: splice recovers")
    print(f"  narrow-gap contests best. random = 1/contest_size.")
    gbins = [-1e-9, 0.02, 0.05, 0.1, 0.2, 0.5, 1.01]
    glabels = ["0-.02", ".02-.05", ".05-.1", ".1-.2", ".2-.5", ".5-1"]
    C["gb"] = pd.cut(C.gap, gbins, labels=glabels)
    print(f"  {'gap_bin':<10}{'n':>6}{'random':>9}{'splice':>9}{'spl-rnd':>9}{'95% CI':>18}")
    mr = []
    for b in glabels:
        gb = C[C.gb == b]
        if len(gb) == 0:
            continue
        rr = float(gb.rand_hit.mean()); k = int(gb.splice_hit.sum()); p, lo, hi = wilson(k, len(gb))
        print(f"  {b:<10}{len(gb):>6}{rr:>9.3f}{p:>9.3f}{p-rr:>+9.3f}   [{lo:.3f}, {hi:.3f}]")
        mr.append({"gap_bin": b, "n": len(gb), "random": rr, "splice": p,
                   "spl_minus_rnd": p-rr, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(mr).to_csv(f"{a.out_prefix}_margin_stratified.tsv", sep="\t", index=False)
    print(f"\n[out] {a.out_prefix}_rf_stratum_auroc.tsv, _margin_stratified.tsv, _contested_pergene.tsv")

if __name__ == "__main__":
    main()