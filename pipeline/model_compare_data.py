#!/usr/bin/env python3
# builds the data for a two-model comparison figure: a
import argparse, os, sys, json
import numpy as np, pandas as pd

KEY = ["gene", "chrom", "center", "strand"]
CCLAB = "cCPA site"

GC_MODEL = ["spl_upstream_max_me3", "gc_contrast"]


# genome
class Genome:

    def __init__(self, path):
        fai = path + ".fai"
        if not os.path.exists(fai):
            sys.exit("need " + fai + "; run `samtools faidx " + path + "`")
        self.idx = {}
        for line in open(fai):
            f = line.split("\t")
            self.idx[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        self.fh = open(path, "rb")

    def fetch(self, chrom, start, end):
        if chrom not in self.idx or end <= start:
            return ""
        ln, off, lb, lw = self.idx[chrom]
        start, end = max(0, start), min(end, ln)
        if end <= start:
            return ""
        def seek(p):
            return off + p // lb * lw + p % lb
        self.fh.seek(seek(start))
        return self.fh.read(seek(end) - seek(start)).decode().replace("\n", "").upper()


def window_gc(gen, chrom, center, half):
    s = gen.fetch(chrom, int(center) - half, int(center) + half)
    if not s:
        return np.nan
    g = s.count("G") + s.count("C")
    n = g + s.count("A") + s.count("T")
    return g / n if n else np.nan


# model
def fit_logistic(X, y, l2=1.0, iters=200, tol=1e-8):
    n, p = X.shape
    Xd = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(p + 1)
    pen = np.full(p + 1, l2); pen[0] = 0.0
    for _ in range(iters):
        z = Xd @ w
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        s = np.clip(mu * (1 - mu), 1e-9, None)
        g = Xd.T @ (mu - y) + pen * w
        H = (Xd * s[:, None]).T @ Xd + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w - step
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def predict(w, X):
    return X @ w[1:] + w[0]


class Standardizer:

    def fit(self, X):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd[~np.isfinite(self.sd) | (self.sd < 1e-9)] = 1.0
        self.mu[~np.isfinite(self.mu)] = 0.0
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


# metrics
def auroc(y, s):
    m = np.isfinite(s)
    y = y[m].astype(int); s = s[m]
    npos, nneg = int(y.sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    ranks = ((start + csum + 1) / 2.0)[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def auprc(y, s):
    m = np.isfinite(s)
    y = y[m].astype(int); s = s[m]
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return np.nan, np.nan
    o = np.argsort(-s, kind="mergesort"); ys = y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / npos
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return ap, npos / len(y)


def within_gene(df, score_col):
    p1, rr, win, ns = [], [], [], []
    for g, sub in df.groupby("gene", sort=False):
        can = sub[sub.is_canonical]
        oth = sub[~sub.is_canonical]
        if len(can) != 1 or len(oth) == 0:
            continue
        cs = float(can[score_col].iloc[0])
        v = oth[score_col].to_numpy(float)
        v = v[np.isfinite(v)]
        if not np.isfinite(cs) or v.size == 0:
            continue
        n_above = int((v > cs).sum()); n_tie = int((v == cs).sum())
        rank = 1 + n_above + 0.5 * n_tie
        p1.append(1.0 if rank <= 1.0 else 0.0)
        rr.append(1.0 / rank)
        win.append(((v < cs).sum() + 0.5 * n_tie) / v.size)
        ns.append(v.size)
    return (np.array(p1), np.array(rr), np.array(win), np.array(ns))


def boot_ci(v, rng, n=1000):
    if len(v) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--features", required=True,
                    help="shortlist_feat_all.parquet; spl_* merged on "
                         "gene/chrom/center/strand")
    ap.add_argument("--genome", required=True,
                    help="reference FASTA; window GC computed here because "
                         "neither input carries it")
    ap.add_argument("--up", type=int, default=2000,
                    help="upstream extent of the feature window (nt)")
    ap.add_argument("--down", type=int, default=1000,
                    help="downstream extent of the feature window (nt)")
    ap.add_argument("--gc_half", type=int, default=250,
                    help="half-width for the single-window GC (500 nt total)")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=20,
                    help="within-gene feature permutations for the null")
    ap.add_argument("--out", default="modelcmp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, d: d.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)

    pk = pd.read_parquet(a.peaks)
    pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    ft = pd.read_parquet(a.features)
    spl = [c for c in ft.columns if c.startswith("spl_")]
    print(f"[load] peaks {len(pk):,}; {len(spl)} splicing features", file=sys.stderr)

    dup = [c for c in spl if c in pk.columns]
    if dup:
        print(f"[merge] {len(dup)} spl_* column(s) already in the peaks file "
              f"({', '.join(dup)}); taking the feature-file copy", file=sys.stderr)
        pk = pk.drop(columns=dup)
    d = pk.merge(ft[KEY + spl], on=KEY, how="left")

    missing = [c for c in GC_MODEL if c not in d.columns]
    if missing:
        sys.exit("GC model needs " + ", ".join(missing) + " -- not in the merge")

    joined = d[spl].notna().any(axis=1)
    print(f"[merge] {int(joined.sum()):,}/{len(d):,} peaks joined "
          f"({100*joined.mean():.1f}%)", file=sys.stderr)
    d = d[joined].reset_index(drop=True)

    DIST_FEATS = [c for c in spl if "dist" in c]
    SENTINEL = float(max(a.up, a.down))
    added = []
    for c in DIST_FEATS:
        frac = float(d[c].isna().mean())
        if frac == 0:
            continue
        ind = c + "_missing"
        d[ind] = d[c].isna().astype(float)
        d[c] = d[c].fillna(SENTINEL)
        added.append(ind)
        print(f"[nan]   {c}: {100*frac:.1f}% had no site in range -> filled "
              f"{SENTINEL:.0f} nt, indicator {ind} added", file=sys.stderr)
    spl = spl + added

    for c in DIST_FEATS:
        v = d[c].to_numpy(float)
        if np.nanmin(v) < 0:
            oob = (v < -a.up) | (v > a.down)
            v = np.clip(v, -a.up, a.down)
        else:
            oob = v > SENTINEL
            v = np.clip(v, 0, SENTINEL)
        if oob.any():
            print(f"[win]   {c}: {int(oob.sum()):,} values outside "
                  f"-{a.up}/+{a.down} clipped to the edge", file=sys.stderr)
        d[c] = v
    print(f"[win]   feature window -{a.up}/+{a.down} nt", file=sys.stderr)

    nleft = int((~np.isfinite(d[spl].to_numpy(float))).sum())
    if nleft:
        print(f"[nan]   {nleft:,} residual non-finite values median-imputed",
              file=sys.stderr)
        d[spl] = d[spl].apply(lambda s: s.fillna(s.median()))

    n_can = int(d.is_canonical.sum()); n_cc = int((~d.is_canonical).sum())
    print(f"[set]   {n_can:,} canonical, {n_cc:,} {CCLAB}s, "
          f"{d.gene.nunique():,} genes", file=sys.stderr)

    gen = Genome(a.genome)
    gc = np.array([window_gc(gen, c, ce, a.gc_half)
                   for c, ce in zip(d.chrom.astype(str), d.center)])
    d["window_gc"] = gc
    ok = np.isfinite(gc)
    if not ok.all():
        print(f"[gc]    {int((~ok).sum()):,} peaks had no sequence, dropped",
              file=sys.stderr)
        d = d[ok].reset_index(drop=True)
    print(f"[gc]    window GC over {2*a.gc_half} nt: "
          f"canonical {d.loc[d.is_canonical,'window_gc'].mean():.3f}, "
          f"{CCLAB} {d.loc[~d.is_canonical,'window_gc'].mean():.3f}",
          file=sys.stderr)

    y = d.is_canonical.astype(int).to_numpy()
    train = ~d.in_gene_holdout.astype(bool).to_numpy()
    test = ~train
    print(f"[split] train {int(train.sum()):,}, held-out genes "
          f"{int(test.sum()):,}", file=sys.stderr)

    G = np.column_stack([np.ones(len(d)), d.window_gc.to_numpy()])
    Xspl = d[spl].to_numpy(float)
    def centre_by_gene(X, genes):
        out = np.empty_like(X)
        df = pd.DataFrame(X)
        df["_g"] = genes
        out[:] = (df.groupby("_g").transform(lambda s: s - s.mean())).to_numpy()
        return np.nan_to_num(out, nan=0.0)

    gvec = d.gene.to_numpy()
    Xspl_c = centre_by_gene(Xspl, gvec)
    Xgc_c = centre_by_gene(d[["window_gc"]].to_numpy(float), gvec)
    Xres = np.empty_like(Xspl)
    r2 = []
    for j in range(Xspl.shape[1]):
        v = Xspl[:, j]
        m = train & np.isfinite(v)
        if m.sum() < 10:
            Xres[:, j] = v; r2.append(np.nan); continue
        beta = np.linalg.lstsq(G[m], v[m], rcond=None)[0]
        pred = G @ beta
        Xres[:, j] = v - pred
        ss_res = np.nansum((v[m] - pred[m]) ** 2)
        ss_tot = np.nansum((v[m] - np.nanmean(v[m])) ** 2)
        r2.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    W("gc_r2_per_feature.tsv",
      pd.DataFrame(dict(feature=spl, r2_on_gc=np.round(r2, 4)))
        .sort_values("r2_on_gc", ascending=False))
    if np.isfinite(r2).any():
        print(f"[gc]    splicing-feature variance explained by GC: median R2 "
              f"{np.nanmedian(r2):.3f}, max {np.nanmax(r2):.3f} "
              f"({spl[int(np.nanargmax(r2))]})", file=sys.stderr)

    MODELS = {
        "gc_model":       (d[GC_MODEL].to_numpy(float), GC_MODEL,
                           "MaxEnt 3'ss + GC contrast"),
        "splicing_only":  (Xspl, spl, "splicing features only"),
        "splicing_gcres": (Xres, spl, "splicing, GC-residualised"),
        "gc_only":        (d[["window_gc"]].to_numpy(float), ["window_gc"],
                           "window GC alone"),
        "gc_centred":       (Xgc_c, ["window_gc_centred"],
                             "window GC, gene-centred"),
        "splicing_centred": (Xspl_c, spl, "splicing, gene-centred"),
    }

    rows, coefs = [], []
    for name, (X, cols, label) in MODELS.items():
        st = Standardizer().fit(X[train])
        w = fit_logistic(st.transform(X[train]), y[train], l2=a.l2)
        s = predict(w, st.transform(X))
        d["score_" + name] = s

        for split, m in [("train", train), ("holdout", test)]:
            au = auroc(y[m], s[m])
            apv, base = auprc(y[m], s[m])
            p1, rr, win, ns = within_gene(d[m], "score_" + name)
            lo1, hi1 = boot_ci(p1, rng, a.boot)
            lom, him = boot_ci(rr, rng, a.boot)
            low, hiw = boot_ci(win, rng, a.boot)
            rows.append(dict(
                model=name, label=label, split=split,
                n_peaks=int(m.sum()), n_genes=len(p1),
                auroc=round(au, 4), auprc=round(apv, 4),
                auprc_baseline=round(base, 4),
                p_at_1=round(float(p1.mean()), 4) if len(p1) else np.nan,
                p_at_1_lo=round(lo1, 4), p_at_1_hi=round(hi1, 4),
                mrr=round(float(rr.mean()), 4) if len(rr) else np.nan,
                mrr_lo=round(lom, 4), mrr_hi=round(him, 4),
                win_rate=round(float(win.mean()), 4) if len(win) else np.nan,
                win_rate_lo=round(low, 4), win_rate_hi=round(hiw, 4),
                median_candidates=int(np.median(ns)) if len(ns) else 0))
        for c, b in zip(cols, w[1:]):
            coefs.append(dict(model=name, feature=c, coef=round(float(b), 4)))
        r = rows[-1]
        print(f"[fit]   {name:<15} holdout AUROC {r['auroc']:.3f}  "
              f"AUPRC {r['auprc']:.3f} (chance {r['auprc_baseline']:.3f})  "
              f"P@1 {r['p_at_1']:.3f}  MRR {r['mrr']:.3f}", file=sys.stderr)

    groups = [v for v in d.groupby("gene").indices.values()]
    perm_rows = []
    for name in ("gc_model", "splicing_only", "splicing_centred", "gc_only"):
        X, cols, label = MODELS[name]
        aus, wins = [], []
        for _ in range(a.n_perm):
            Xp = X.copy()
            for ix in groups:
                if len(ix) > 1:
                    Xp[ix] = X[rng.permutation(ix)]
            st = Standardizer().fit(Xp[train])
            w = fit_logistic(st.transform(Xp[train]), y[train], l2=a.l2)
            sp = predict(w, st.transform(Xp))
            aus.append(auroc(y[test], sp[test]))
            sub = d[test].copy(); sub["_sp"] = sp[test]
            _, _, wr, _ = within_gene(sub, "_sp")
            wins.append(float(wr.mean()) if len(wr) else np.nan)
        aus = np.array(aus, float); wins = np.array(wins, float)
        perm_rows.append(dict(model=name, label=label, n_perm=a.n_perm,
                              null_auroc_mean=round(float(np.nanmean(aus)), 4),
                              null_auroc_lo=round(float(np.nanpercentile(aus, 2.5)), 4),
                              null_auroc_hi=round(float(np.nanpercentile(aus, 97.5)), 4),
                              null_win_rate_mean=round(float(np.nanmean(wins)), 4),
                              null_win_rate_lo=round(float(np.nanpercentile(wins, 2.5)), 4),
                              null_win_rate_hi=round(float(np.nanpercentile(wins, 97.5)), 4)))
        print(f"[perm]  {name:<15} null AUROC "
              f"{perm_rows[-1]['null_auroc_mean']:.3f}, "
              f"null win rate {perm_rows[-1]['null_win_rate_mean']:.3f} "
              f"(should be ~0.500)", file=sys.stderr)
    W("perm_null.tsv", pd.DataFrame(perm_rows))

    W("metrics.tsv", pd.DataFrame(rows))
    W("coefficients.tsv", pd.DataFrame(coefs))

    loco = []
    for name, (X, cols, label) in MODELS.items():
        for c in sorted(d.loco_chromosome.dropna().unique()):
            te = (d.loco_chromosome == c).to_numpy()
            tr = ~te
            if te.sum() < 50 or y[tr].sum() < 10 or y[te].sum() < 5:
                continue
            st = Standardizer().fit(X[tr])
            w = fit_logistic(st.transform(X[tr]), y[tr], l2=a.l2)
            s = predict(w, st.transform(X))
            au = auroc(y[te], s[te])
            sub = d[te].copy(); sub["_s"] = s[te]
            p1, rr, _, _ = within_gene(sub, "_s")
            loco.append(dict(model=name, label=label, chrom=str(c),
                             n_peaks=int(te.sum()), n_genes=len(p1),
                             auroc=round(au, 4),
                             auroc_excess=round(au - 0.5, 4),
                             p_at_1=round(float(p1.mean()), 4) if len(p1) else np.nan,
                             mrr=round(float(rr.mean()), 4) if len(rr) else np.nan))
    W("loco.tsv", pd.DataFrame(loco))
    L = pd.DataFrame(loco)
    for name in MODELS:
        s_ = L[L.model == name]
        if len(s_):
            print(f"[loco]  {name:<15} median AUROC across {len(s_)} "
                  f"chromosomes: {s_.auroc.median():.3f}", file=sys.stderr)

    keep = ["gene", "chrom", "center", "strand", "is_canonical", "window_gc",
            "in_gene_holdout"] + ["score_" + m for m in MODELS]
    sc = d[keep]
    if len(sc) > 60000:
        sc = sc.sample(60000, random_state=0)
    W("gc_scatter.tsv", sc)

    cor = []
    for name in MODELS:
        for lab, m in [("Canonical", d.is_canonical.to_numpy()),
                       (CCLAB, ~d.is_canonical.to_numpy())]:
            v = d.loc[m, ["window_gc", "score_" + name]].dropna()
            r = float(np.corrcoef(v.window_gc, v["score_" + name])[0, 1]) \
                if len(v) > 2 else np.nan
            cor.append(dict(model=name, cls=lab, r=round(r, 4),
                            r2=round(r * r, 4), n=len(v)))
    W("gc_correlation.tsv", pd.DataFrame(cor))
    for r_ in cor:
        if r_["model"] in ("gc_model", "splicing_only"):
            print(f"[gc]    {r_['model']:<15} {r_['cls']:<12} "
                  f"r(score, window GC) = {r_['r']:+.3f}", file=sys.stderr)

    TOP = 6
    imp = (pd.DataFrame(coefs).query("model == 'splicing_only'")
           .assign(absc=lambda x: x.coef.abs())
           .nlargest(TOP, "absc").feature.tolist())
    dens = []
    for f in imp + GC_MODEL:
        if f not in d.columns:
            continue
        for lab, m in [("Canonical", d.is_canonical.to_numpy()),
                       (CCLAB, ~d.is_canonical.to_numpy())]:
            v = pd.to_numeric(d.loc[m, f], errors="coerce").dropna().to_numpy()
            if len(v) > 20000:
                v = rng.choice(v, 20000, replace=False)
            dens.append(pd.DataFrame(dict(feature=f, cls=lab, value=v)))
    if dens:
        pd.concat(dens, ignore_index=True).to_csv(
            os.path.join(a.out, "feature_values.tsv"), sep="\t", index=False)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_canonical=n_can, n_ccpa=n_cc,
                       n_genes=int(d.gene.nunique()),
                       n_features_splicing=len(spl),
                       feature_window_up=a.up, feature_window_down=a.down,
                       gc_window=2 * a.gc_half), fh, indent=2)

    print(f"\n[out]   {a.out}/metrics.tsv, loco.tsv, coefficients.tsv, "
          f"gc_scatter.tsv, gc_correlation.tsv, gc_r2_per_feature.tsv, "
          f"perm_null.tsv, feature_values.tsv", file=sys.stderr)
    print("\nRead gc_only first. If splicing_only does not clearly beat it, the "
          "splicing features are a GC detector. Then splicing_gcres: what "
          "survives residualisation is the part that is not GC.", file=sys.stderr)


if __name__ == "__main__":
    main()