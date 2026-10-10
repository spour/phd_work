#!/usr/bin/env python3
# the 2d generalization of the cumulative "strongest upstream 3'ss vs distance" curve
import argparse, os, sys
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts_dir", default=None)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--max_nt", type=int, default=2000, help="upstream window (nt)")
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--dist_bin", type=int, default=50)
    ap.add_argument("--score_bin", type=float, default=0.5)
    ap.add_argument("--score_lo", type=float, default=-15.0)
    ap.add_argument("--score_hi", type=float, default=14.0)
    ap.add_argument("--max_per_class", type=int, default=8000,
                    help="subsample peaks per class (canonical n is ~10x smaller; equal n "
                         "keeps the two maps equally noisy)")
    ap.add_argument("--balanced", action="store_true", default=True,
                    help="use the same n for both classes (recommended)")
    ap.add_argument("--min_cell", type=int, default=30,
                    help="mask ratio cells with fewer than this many total peaks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fig3_landscape")
    a = ap.parse_args()
    if a.scripts_dir: sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import GenomeFasta, init_maxentscan
    _, score3, _, m3 = init_maxentscan()
    fa = GenomeFasta(a.reference_fasta)

    df = pd.read_parquet(a.shortlist)
    df = df[df.n_internal > 0]
    df["cls"] = np.where(df.is_canonical, "Canonical", "Competitor")
    rng = np.random.default_rng(a.seed)
    parts = []
    n_target = a.max_per_class
    if a.balanced:
        n_target = min(n_target, int(df.cls.value_counts().min()))
    for cls, gg in df.groupby("cls"):
        if len(gg) > n_target: gg = gg.sample(n_target, random_state=a.seed)
        parts.append(gg)
    df = pd.concat(parts, ignore_index=True)
    print(f"[data] {len(df):,} peaks "
          f"({int((df.cls=='Canonical').sum()):,} canonical, "
          f"{int((df.cls=='Competitor').sum()):,} competitor); "
          f"upstream {a.max_nt}nt", file=sys.stderr)

    wins = np.arange(a.dist_bin, a.max_nt + a.dist_bin, a.dist_bin)
    sedges = np.arange(a.score_lo, a.score_hi + a.score_bin, a.score_bin)
    nW, nS = len(wins), len(sedges) - 1
    acc = {c: np.zeros((nW, nS)) for c in ("Canonical", "Competitor")}
    npk = {c: 0 for c in ("Canonical", "Competitor")}
    n_bad = 0
    runmax_rows = {c: [] for c in ("Canonical", "Competitor")}
    gene_rows = {c: [] for c in ("Canonical", "Competitor")}

    for _, r in tqdm(df.iterrows(), total=len(df), desc="acceptors"):
        c = int(r["center"]); strand = r["strand"]; chrom = str(r["chrom"]); cls = r["cls"]
        if strand == "+": w0, w1 = max(0, c - a.max_nt), c + a.pad
        else:             w0, w1 = max(0, c - a.pad), c + a.max_nt
        try: seq = fa.fetch(chrom, w0, w1, strand=strand)
        except Exception: seq = None
        if not seq or len(seq) < 30: n_bad += 1; continue
        seq = seq.upper(); L = len(seq); site = L - a.pad
        npk[cls] += 1
        arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
        ags = np.where((arr[:-1] == 65) & (arr[1:] == 71))[0] + 2
        ds = []; ss = []
        for i in ags:
            p = int(i)
            if p - 20 < 0 or p + 3 > L: continue
            if p > site - a.pad: continue
            try:
                v = score3(seq[p-20:p+3])
                v = float(np.nanmax(np.asarray(v, float))) if not np.isscalar(v) else float(v)
            except Exception:
                continue
            if not np.isfinite(v) or v <= -99: continue
            ds.append(site - p); ss.append(v)
        if not ds: continue
        ds = np.asarray(ds, float); ss = np.asarray(ss, float)
        o = np.argsort(ds); ds = ds[o]; ss = ss[o]
        runmax = np.maximum.accumulate(ss)
        idx = np.searchsorted(ds, wins, side="right") - 1
        row = np.full(nW, np.nan)
        for wi, k in enumerate(idx):
            if k < 0: continue
            row[wi] = runmax[k]
            j = int(np.clip(np.searchsorted(sedges, runmax[k], side="right") - 1, 0, nS - 1))
            acc[cls][wi, j] += 1.0
        runmax_rows[cls].append(row)
        gene_rows[cls].append(str(r["gene"]))

    rows = []
    sc = (sedges[:-1] + sedges[1:]) / 2
    for cls in ("Canonical", "Competitor"):
        if npk[cls] == 0: continue
        M = acc[cls] / npk[cls]
        for i in range(nW):
            for j in range(nS):
                if M[i, j] == 0: continue
                rows.append(dict(cls=cls, window=int(wins[i]), maxscore=float(sc[j]),
                                 frac=float(M[i, j])))
    L = pd.DataFrame(rows)
    L.to_csv(f"{a.out}_landscape.tsv", sep="\t", index=False)

    crows = []
    for cls in ("Canonical", "Competitor"):
        if npk[cls] == 0: continue
        M = acc[cls]
        for i in range(nW):
            n = M[i].sum()
            if n == 0: continue
            mu = float((M[i] * sc).sum() / n)
            var = float((M[i] * (sc - mu) ** 2).sum() / max(n, 1))
            crows.append(dict(cls=cls, window=int(wins[i]), mean_max=mu,
                              se=float(np.sqrt(var / max(n, 1))), n=int(n)))
    pd.DataFrame(crows).to_csv(f"{a.out}_curve.tsv", sep="\t", index=False)

    RM = {c: np.array(runmax_rows[c], float) for c in runmax_rows if runmax_rows[c]}
    # between-class #1: quantile curves of the running max (tail, not mean)
    qrows = []
    for cls, M in RM.items():
        for wi, w in enumerate(wins):
            col = M[:, wi]; col = col[np.isfinite(col)]
            if len(col) < 20: continue
            for q in (0.5, 0.75, 0.9, 0.95):
                qrows.append(dict(cls=cls, window=int(w), quantile=q,
                                  value=float(np.quantile(col, q)), n=len(col)))
    pd.DataFrame(qrows).to_csv(f"{a.out}_quantiles.tsv", sep="\t", index=False)

    # between-class #2: p(strong acceptor exists by window), threshold sweep
    trows = []
    for cls, M in RM.items():
        for thr in (6.5, 8.0, 10.0):
            for wi, w in enumerate(wins):
                col = M[:, wi]; col = col[np.isfinite(col)]
                if len(col) < 20: continue
                trows.append(dict(cls=cls, threshold=thr, window=int(w),
                                  frac=float((col >= thr).mean()), n=len(col)))
    pd.DataFrame(trows).to_csv(f"{a.out}_threshold.tsv", sep="\t", index=False)

    # between-class #3: auroc of "strongest upstream 3'ss within x" vs window
    def _auroc(pos, neg):
        s2 = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        o = np.argsort(s2, kind="mergesort"); r = np.empty(len(s2)); r[o] = np.arange(1, len(s2)+1)
        _, inv, cnt = np.unique(s2, return_inverse=True, return_counts=True)
        cs = np.cumsum(cnt); r = ((cs - cnt) + cs + 1)[inv] / 2.0
        return (r[y==1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
    if {"Canonical","Competitor"} <= set(RM):
        arows = []
        for wi, w in enumerate(wins):
            pc = RM["Canonical"][:, wi]; kc = RM["Competitor"][:, wi]
            pc = pc[np.isfinite(pc)]; kc = kc[np.isfinite(kc)]
            if len(pc) < 20 or len(kc) < 20: continue
            arows.append(dict(window=int(w), auroc=float(_auroc(pc, kc)),
                              n_canon=len(pc), n_comp=len(kc)))
        pd.DataFrame(arows).to_csv(f"{a.out}_auroc.tsv", sep="\t", index=False)

    # within-gene paired: canonical vs its own gene's competitors
    if {"Canonical","Competitor"} <= set(RM):
        wlast = nW - 1
        cg = {}; kg = {}
        for cls, store in (("Canonical", cg), ("Competitor", kg)):
            for gi, g in enumerate(gene_rows[cls]):
                v = RM[cls][gi, wlast]
                if np.isfinite(v): store.setdefault(g, []).append(v)
        prows = []
        for g in set(cg) & set(kg):
            cval = np.mean(cg[g]); comp = np.array(kg[g])
            prows.append(dict(gene=g, canon=float(cval),
                              comp_mean=float(comp.mean()), comp_max=float(comp.max()),
                              comp_min=float(comp.min()), k=len(comp)))
        P = pd.DataFrame(prows)
        P.to_csv(f"{a.out}_paired.tsv", sep="\t", index=False)
        if len(P):
            null_pair = 0.5
            null_max = float((1.0/(P.k+1)).mean())
            null_min = float((P.k/(P.k+1)).mean())
            obs_pair = float((P.canon > P.comp_mean).mean())
            obs_max  = float((P.canon > P.comp_max).mean())
            obs_min  = float((P.canon > P.comp_min).mean())
            pd.DataFrame([
                dict(agg="pair (mean competitor)", observed=obs_pair, null=null_pair, n=len(P)),
                dict(agg="vs strongest competitor", observed=obs_max, null=null_max, n=len(P)),
                dict(agg="vs weakest competitor",   observed=obs_min, null=null_min, n=len(P)),
            ]).to_csv(f"{a.out}_paired_summary.tsv", sep="\t", index=False)
            print(f"[paired @ {a.max_nt}nt] canonical best-acceptor beats:", file=sys.stderr)
            print(f"    mean competitor: {obs_pair:.3f} (null 0.500)", file=sys.stderr)
            print(f"    STRONGEST comp:  {obs_max:.3f} (null {null_max:.3f})  <- the honest test", file=sys.stderr)
            print(f"    weakest comp:    {obs_min:.3f} (null {null_min:.3f})", file=sys.stderr)

    if all(npk[c] > 0 for c in ("Canonical", "Competitor")):
        D = acc["Canonical"]/npk["Canonical"] - acc["Competitor"]/npk["Competitor"]
        drows = [dict(window=int(wins[i]), maxscore=float(sc[j]), diff=float(D[i, j]))
                 for i in range(nW) for j in range(nS)]
        pd.DataFrame(drows).to_csv(f"{a.out}_diff.tsv", sep="\t", index=False)
        k = np.unravel_index(np.argmax(D), D.shape)
        print(f"[diff] max canonical excess {D[k]:+.4f} of peaks at "
              f"window~{wins[k[0]]}nt, running-max score~{sc[k[1]]:.1f}", file=sys.stderr)
        MC = acc["Canonical"]/npk["Canonical"]; MK = acc["Competitor"]/npk["Competitor"]
        eps = 1.0/max(npk["Canonical"],npk["Competitor"])
        ratio = np.log2((MC+eps)/(MK+eps))
        tot = acc["Canonical"] + acc["Competitor"]
        ratio = np.where(tot >= a.min_cell, ratio, np.nan)
        rrows=[dict(window=int(wins[i]), maxscore=float(sc[j]), log2ratio=float(ratio[i,j]))
               for i in range(nW) for j in range(nS) if np.isfinite(ratio[i,j])]
        pd.DataFrame(rrows).to_csv(f"{a.out}_ratio.tsv", sep="\t", index=False)
        print(f"[ratio] log2(canon/comp) range "
              f"[{np.nanmin(ratio):+.2f}, {np.nanmax(ratio):+.2f}] over "
              f"{np.isfinite(ratio).sum()} cells", file=sys.stderr)
        strong = sc >= 6.5
        for w_probe in (250, 500, 1000, 2000):
            if w_probe > a.max_nt: continue
            wi = int(np.searchsorted(wins, w_probe))
            if wi >= nW: continue
            out = []
            for cls in ("Canonical","Competitor"):
                M = acc[cls][wi]
                out.append(M[strong].sum() / max(M.sum(), 1))
            print(f"[strong@{w_probe:>4}nt] P(best acceptor >= 6.5): "
                  f"canonical {out[0]:.3f}  competitor {out[1]:.3f}  diff {out[0]-out[1]:+.3f}",
                  file=sys.stderr)

    print(f"[peaks] used {npk}; {n_bad} unfetchable", file=sys.stderr)
    print(f"[out] {a.out}_landscape.tsv, {a.out}_diff.tsv", file=sys.stderr)

if __name__ == "__main__":
    main()