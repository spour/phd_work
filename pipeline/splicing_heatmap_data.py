#!/usr/bin/env python3
# companion to splicing_model_data.py
import argparse, os, sys
import numpy as np, pandas as pd

MODELS = ["gc_only", "splicing_centred"]
LABELS = {"gc_only": "Window GC", "splicing_centred": "Splicing (gene-centred)"}


def gene_stats(sub, col):
    can = sub[sub.is_canonical]
    oth = sub[~sub.is_canonical]
    if len(can) != 1 or len(oth) == 0:
        return None
    cs = float(can[col].iloc[0])
    v = pd.to_numeric(oth[col], errors="coerce").dropna().to_numpy()
    if not np.isfinite(cs) or v.size == 0:
        return None
    n_above = int((v > cs).sum()); n_tie = int((v == cs).sum())
    auc = ((v < cs).sum() + 0.5 * n_tie) / v.size
    rank = 1 + n_above + 0.5 * n_tie
    return dict(auc=float(auc), rr=1.0 / rank, p1=1.0 if rank <= 1.0 else 0.0,
                rank=rank, n_ccpa=int(v.size),
                rank_pct=float((rank - 1) / max(v.size, 1)))


def boot_ci(v, rng, n=400):
    if len(v) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="splmodel")
    ap.add_argument("--scores", default="scores.tsv")
    ap.add_argument("--n_gc_bins", type=int, default=10)
    ap.add_argument("--n_rank_bins", type=int, default=10)
    ap.add_argument("--holdout_only", action="store_true", default=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    W = lambda n, df: df.to_csv(os.path.join(a.indir, n), sep="\t", index=False)

    p = os.path.join(a.indir, a.scores)
    if not os.path.exists(p):
        sys.exit("missing " + p + " -- run splicing_model_data.py first")
    d = pd.read_csv(p, sep="\t")

    missing = [m for m in MODELS if "score_" + m not in d.columns]
    if missing:
        sys.exit("scores.tsv has no column for: " + ", ".join(missing))

    if a.holdout_only and "in_gene_holdout" in d.columns:
        n0 = len(d)
        d = d[d.in_gene_holdout.astype(bool)].reset_index(drop=True)
        print(f"[sub]  held-out genes only: {len(d):,}/{n0:,} peaks",
              file=sys.stderr)

    ok_genes, dropped = [], 0
    for g, sub in d.groupby("gene", sort=False):
        if sub.is_canonical.sum() == 1 and (~sub.is_canonical).sum() >= 1:
            ok_genes.append(g)
        else:
            dropped += 1
    d = d[d.gene.isin(set(ok_genes))].reset_index(drop=True)
    print(f"[sub]  {len(ok_genes):,} usable genes, {dropped:,} dropped "
          f"(subsampling left them without a canonical site or without any "
          f"cCPA site)", file=sys.stderr)

    # per-gene statistics, per model
    rows = []
    for g, sub in d.groupby("gene", sort=False):
        gc_can = float(sub.loc[sub.is_canonical, "window_gc"].iloc[0])
        rec = dict(gene=g, gc_canonical=gc_can,
                   gc_ccpa_mean=float(sub.loc[~sub.is_canonical, "window_gc"].mean()))
        keep = True
        for m in MODELS:
            st = gene_stats(sub, "score_" + m)
            if st is None:
                keep = False
                break
            for k, v in st.items():
                rec[m + "_" + k] = v
        if keep:
            rows.append(rec)
    G = pd.DataFrame(rows)
    print(f"[gene] {len(G):,} genes with a complete statistic for both models",
          file=sys.stderr)

    G["gc_bin"] = pd.qcut(G.gc_canonical.rank(method="first"),
                          a.n_gc_bins, labels=False) + 1
    W("hm_gene_strip.tsv",
      G[["gene", "gc_canonical", "gc_ccpa_mean", "gc_bin"] +
        [m + s for m in MODELS for s in ("_auc", "_rr", "_p1", "_rank",
                                         "_rank_pct", "_n_ccpa")]])

    # gene-level metrics by gc decile
    met = []
    for m in MODELS:
        for b in sorted(G.gc_bin.unique()):
            s = G[G.gc_bin == b]
            if len(s) < 10:
                continue
            for metric, col in [("Within-gene AUROC", m + "_auc"),
                                ("MRR", m + "_rr"),
                                ("P@1", m + "_p1")]:
                v = s[col].to_numpy(float)
                lo, hi = boot_ci(v, rng)
                met.append(dict(model=m, label=LABELS[m], metric=metric,
                                gc_bin=int(b), n_genes=len(s),
                                mean_gc=round(float(s.gc_canonical.mean()), 4),
                                value=round(float(v.mean()), 4),
                                lo=round(lo, 4), hi=round(hi, 4)))
    MET = pd.DataFrame(met)
    W("hm_metrics_by_decile.tsv", MET)

    # overall, pooled over deciles
    ov = []
    for m in MODELS:
        for metric, col in [("Within-gene AUROC", m + "_auc"),
                            ("MRR", m + "_rr"), ("P@1", m + "_p1")]:
            v = G[col].to_numpy(float)
            lo, hi = boot_ci(v, rng, 1000)
            ov.append(dict(model=m, label=LABELS[m], metric=metric,
                           n_genes=len(G), value=round(float(v.mean()), 4),
                           lo=round(lo, 4), hi=round(hi, 4)))
    OV = pd.DataFrame(ov)
    W("hm_overall.tsv", OV)
    print("\n[gene-level metrics, held-out genes]", file=sys.stderr)
    for r in OV.itertuples():
        print(f"  {r.label:<26} {r.metric:<18} {r.value:.3f} "
              f"[{r.lo:.3f}, {r.hi:.3f}]", file=sys.stderr)

    # rank x gc density: the actual data, binned
    dens = []
    for m in MODELS:
        for b in sorted(G.gc_bin.unique()):
            s = G[G.gc_bin == b]
            if len(s) < 10:
                continue
            rp = s[m + "_rank_pct"].to_numpy(float)
            rb = np.clip((rp * a.n_rank_bins).astype(int) + 1, 1, a.n_rank_bins)
            cnt = np.bincount(rb, minlength=a.n_rank_bins + 1)[1:]
            for k in range(a.n_rank_bins):
                dens.append(dict(model=m, label=LABELS[m], gc_bin=int(b),
                                 mean_gc=round(float(s.gc_canonical.mean()), 4),
                                 rank_bin=k + 1, n_genes=len(s),
                                 frac=round(float(cnt[k] / len(s)), 4),
                                 count=int(cnt[k])))
    W("hm_rank_density.tsv", pd.DataFrame(dens))

    print("\n[extremes] within-gene AUROC in the first and last GC decile:",
          file=sys.stderr)
    for m in MODELS:
        s = MET[(MET.model == m) & (MET.metric == "Within-gene AUROC")]
        if len(s):
            lo_ = s[s.gc_bin == s.gc_bin.min()].iloc[0]
            hi_ = s[s.gc_bin == s.gc_bin.max()].iloc[0]
            print(f"  {LABELS[m]:<26} GC {lo_.mean_gc:.3f}: {lo_.value:.3f}   "
                  f"GC {hi_.mean_gc:.3f}: {hi_.value:.3f}   "
                  f"(range {abs(hi_.value - lo_.value):.3f})", file=sys.stderr)

    print(f"\n[out]  {a.indir}/hm_metrics_by_decile.tsv, hm_rank_density.tsv, "
          f"hm_gene_strip.tsv, hm_overall.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()