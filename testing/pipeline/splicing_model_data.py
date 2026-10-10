#!/usr/bin/env python3
# data for a figure making one claim: splice-site features discriminate real 3' ends
import argparse, os, sys, json
import numpy as np, pandas as pd

KEY = ["gene", "chrom", "center", "strand"]
CCLAB = "cCPA site"
GC_PAIR = ["spl_upstream_max_me3", "gc_contrast"]
FIGURE_MODELS = ["gc_only", "splicing_centred", "minimal_centred",
                 "splicing_no_u1"]


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
            return w_new
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
        return np.nan_to_num((X - self.mu) / self.sd, nan=0.0,
                             posinf=0.0, neginf=0.0)


def standardize_by_fold(s, folds):
    s = np.asarray(s, float)
    out = np.full(len(s), np.nan)
    for f in pd.unique(pd.Series(folds).dropna()):
        m = (folds == f).to_numpy() & np.isfinite(s)
        if m.sum() < 3:
            continue
        sd = np.nanstd(s[m])
        out[m] = (s[m] - np.nanmean(s[m])) / (sd if sd > 0 else 1.0)
    return out


def centre_by_gene(X, genes):
    X = np.asarray(X, float)
    out = X.copy()
    order = np.argsort(genes, kind="mergesort")
    gs = np.asarray(genes)[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for i in range(len(bounds) - 1):
        ix = order[bounds[i]:bounds[i + 1]]
        out[ix] -= np.nanmean(X[ix], axis=0)
    return np.nan_to_num(out, nan=0.0)


# metrics
def auroc(y, s):
    m = np.isfinite(s)
    y = np.asarray(y)[m].astype(int); s = np.asarray(s)[m]
    npos, nneg = int(y.sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    ranks = ((start + csum + 1) / 2.0)[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def auprc(y, s):
    m = np.isfinite(s)
    y = np.asarray(y)[m].astype(int); s = np.asarray(s)[m]
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return np.nan, np.nan
    o = np.argsort(-s, kind="mergesort"); ys = y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / npos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec)), npos / len(y)


def within_gene_fast(gene_codes, is_can, score):
    order = np.argsort(gene_codes, kind="mergesort")
    gc_ = gene_codes[order]; ic = is_can[order]; sc = score[order]
    bounds = np.flatnonzero(np.r_[True, gc_[1:] != gc_[:-1], True])
    win, p1, rr = [], [], []
    for i in range(len(bounds) - 1):
        sl = slice(bounds[i], bounds[i + 1])
        c = ic[sl]; v = sc[sl]
        if c.sum() != 1:
            continue
        cs = v[c][0]; o = v[~c]
        o = o[np.isfinite(o)]
        if not np.isfinite(cs) or o.size == 0:
            continue
        nt = int((o == cs).sum())
        win.append(((o < cs).sum() + 0.5 * nt) / o.size)
        rank = 1 + int((o > cs).sum()) + 0.5 * nt
        p1.append(1.0 if rank <= 1.0 else 0.0)
        rr.append(1.0 / rank)
    return np.array(win), np.array(p1), np.array(rr)


def within_gene_table(df, col):
    out = []
    for g, sub in df.groupby("gene", sort=False):
        can = sub[sub.is_canonical]; oth = sub[~sub.is_canonical]
        if len(can) != 1 or len(oth) == 0:
            continue
        cs = float(can[col].iloc[0])
        v = pd.to_numeric(oth[col], errors="coerce").dropna().to_numpy()
        if not np.isfinite(cs) or v.size == 0:
            continue
        nt = int((v == cs).sum())
        auc = ((v < cs).sum() + 0.5 * nt) / v.size
        rank = 1 + int((v > cs).sum()) + 0.5 * nt
        out.append(dict(gene=g, auc=float(auc), rank=float(rank),
                        rr=1.0 / rank, p1=1.0 if rank <= 1.0 else 0.0,
                        n_ccpa=int(v.size),
                        rank_pct=float((rank - 1) / max(v.size, 1)),
                        sep=float(cs - v.mean())))
    return pd.DataFrame(out)


def boot_ci(v, rng, n=1000):
    if len(v) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def evaluate(X, y, train, test, gene_codes, is_can, l2, rng, boot):
    st = Standardizer().fit(X[train])
    w = fit_logistic(st.transform(X[train]), y[train], l2=l2)
    s = predict(w, st.transform(X))
    out = {}
    for split, m in [("train", train), ("holdout", test)]:
        au = auroc(y[m], s[m])
        apv, base = auprc(y[m], s[m])
        win, p1, rr = within_gene_fast(gene_codes[m], is_can[m], s[m])
        lo, hi = boot_ci(win, rng, boot)
        out[split] = dict(
            auroc=au, auprc=apv, auprc_baseline=base,
            win_rate=float(win.mean()) if len(win) else np.nan,
            win_lo=lo, win_hi=hi,
            p_at_1=float(p1.mean()) if len(p1) else np.nan,
            mrr=float(rr.mean()) if len(rr) else np.nan,
            n_genes=len(win), n_peaks=int(m.sum()))
    return out, w, s


def loco_scores(X, y, loco_chrom, l2):
    s = np.full(len(y), np.nan)
    for c in pd.unique(loco_chrom.dropna()):
        te = (loco_chrom == c).to_numpy()
        tr = ~te
        if te.sum() == 0 or y[tr].sum() < 10 or (1 - y[tr]).sum() < 10:
            continue
        st = Standardizer().fit(X[tr])
        w = fit_logistic(st.transform(X[tr]), y[tr], l2=l2)
        s[te] = predict(w, st.transform(X[te]))
    return s


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--up", type=int, default=2000)
    ap.add_argument("--down", type=int, default=1000)
    ap.add_argument("--gc_half", type=int, default=250)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--sel_max", type=int, default=14)
    ap.add_argument("--sel_tol", type=float, default=0.001)
    ap.add_argument("--n_gc_bins", type=int, default=10)
    ap.add_argument("--n_rank_bins", type=int, default=10)
    ap.add_argument("--out", default="splmodel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)

    pk = pd.read_parquet(a.peaks)
    pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    ft = pd.read_parquet(a.features)
    spl = [c for c in ft.columns if c.startswith("spl_")]
    dup = [c for c in spl if c in pk.columns]
    if dup:
        pk = pk.drop(columns=dup)
    d = pk.merge(ft[KEY + spl], on=KEY, how="left")
    d = d[d[spl].notna().any(axis=1)].reset_index(drop=True)
    print(f"[load] {len(d):,} peaks, {len(spl)} splicing features", file=sys.stderr)

    DIST = [c for c in spl if "dist" in c]
    SENT = float(max(a.up, a.down))
    for c in DIST:
        f = float(d[c].isna().mean())
        if f > 0:
            ind = c + "_missing"
            d[ind] = d[c].isna().astype(float)
            d[c] = d[c].fillna(SENT)
            spl.append(ind)
            print(f"[nan]  {c}: {100*f:.1f}% no site in range -> sentinel + "
                  f"indicator", file=sys.stderr)
        v = d[c].to_numpy(float)
        d[c] = np.clip(v, -a.up, a.down) if np.nanmin(v) < 0 else np.clip(v, 0, SENT)
    d[spl] = d[spl].apply(lambda s: s.fillna(s.median()))

    gen = Genome(a.genome)
    d["window_gc"] = [window_gc(gen, c, ce, a.gc_half)
                      for c, ce in zip(d.chrom.astype(str), d.center)]
    d = d[np.isfinite(d.window_gc)].reset_index(drop=True)

    y = d.is_canonical.astype(int).to_numpy()
    is_can = d.is_canonical.to_numpy(bool)
    train = ~d.in_gene_holdout.astype(bool).to_numpy()
    test = ~train
    genes = d.gene.to_numpy()
    gene_codes = pd.factorize(genes)[0]
    print(f"[set]  {int(y.sum()):,} real 3' ends, {int((1-y).sum()):,} "
          f"competing sites, {d.gene.nunique():,} genes", file=sys.stderr)

    per_gene = []
    for g, sub in d.groupby("gene", sort=False):
        can = sub[sub.is_canonical]; oth = sub[~sub.is_canonical]
        if len(can) != 1 or len(oth) == 0:
            continue
        per_gene.append(dict(gene=g,
                             gc_canonical=float(can.window_gc.iloc[0]),
                             gc_ccpa_mean=float(oth.window_gc.mean()),
                             delta_gc=float(can.window_gc.iloc[0] -
                                            oth.window_gc.mean()),
                             n_ccpa=len(oth)))
    PG = pd.DataFrame(per_gene)
    W("gc_within_gene.tsv", PG)

    W("gc_pooled.tsv", pd.DataFrame(dict(
        cls=np.where(is_can, "Canonical", CCLAB),
        window_gc=d.window_gc.to_numpy())))
    print(f"[simp] pooled GC: real {d.loc[is_can,'window_gc'].mean():.4f}, "
          f"competing {d.loc[~is_can,'window_gc'].mean():.4f}", file=sys.stderr)
    print(f"[simp] within gene: mean {PG.delta_gc.mean():+.4f}, "
          f"{100*(PG.delta_gc<0).mean():.1f}% of genes have the real site "
          f"BELOW its own competitors ({len(PG):,} genes)", file=sys.stderr)

    Xspl = d[spl].to_numpy(float)
    Xgc = d[["window_gc"]].to_numpy(float)
    Xpair = d[GC_PAIR].to_numpy(float)

    G = np.column_stack([np.ones(len(d)), d.window_gc.to_numpy()])
    Xres = np.empty_like(Xspl); r2 = []
    for j in range(Xspl.shape[1]):
        v = Xspl[:, j]; m = train & np.isfinite(v)
        if m.sum() < 10:
            Xres[:, j] = v; r2.append(np.nan); continue
        beta = np.linalg.lstsq(G[m], v[m], rcond=None)[0]
        Xres[:, j] = v - G @ beta
        ssr = np.nansum((v[m] - (G @ beta)[m]) ** 2)
        sst = np.nansum((v[m] - np.nanmean(v[m])) ** 2)
        r2.append(1 - ssr / sst if sst > 0 else np.nan)
    R2 = pd.DataFrame(dict(feature=spl, r2_on_gc=np.round(r2, 4)))
    W("gc_r2_per_feature.tsv", R2.sort_values("r2_on_gc", ascending=False))
    print(f"[gc]   feature variance explained by GC: median R2 "
          f"{np.nanmedian(r2):.3f}, max {np.nanmax(r2):.3f}", file=sys.stderr)

    Xspl_c = centre_by_gene(Xspl, genes)
    Xgc_c = centre_by_gene(Xgc, genes)
    Xpair_c = centre_by_gene(Xpair, genes)
    Xcomb_c = np.column_stack([Xspl_c, Xgc_c])

    tr_genes = np.unique(genes[train])
    rng_sel = np.random.default_rng(1)
    inner_te_genes = set(rng_sel.choice(tr_genes,
                                        max(1, int(0.25 * len(tr_genes))),
                                        replace=False))
    inner_te = train & np.array([g in inner_te_genes for g in genes])
    inner_tr = train & ~inner_te

    chosen, curve, remaining = [], [], list(range(len(spl)))
    best_so_far = 0.5
    while len(chosen) < a.sel_max and remaining:
        scored = []
        for j in remaining:
            Xc = Xspl_c[:, chosen + [j]]
            st = Standardizer().fit(Xc[inner_tr])
            w = fit_logistic(st.transform(Xc[inner_tr]), y[inner_tr], l2=a.l2)
            s = predict(w, st.transform(Xc))
            win, _, _ = within_gene_fast(gene_codes[inner_te], is_can[inner_te],
                                         s[inner_te])
            scored.append((float(win.mean()) if len(win) else 0.0, j))
        scored.sort(reverse=True)
        gain = scored[0][0] - best_so_far
        curve.append(dict(step=len(chosen) + 1, feature=spl[scored[0][1]],
                          inner_win_rate=round(scored[0][0], 4),
                          gain=round(gain, 4)))
        if gain < a.sel_tol and len(chosen) >= 1:
            curve = curve[:-1]
            break
        best_so_far = scored[0][0]
        chosen.append(scored[0][1])
        remaining.remove(scored[0][1])
    W("selection_curve.tsv", pd.DataFrame(curve))
    minimal = [spl[j] for j in chosen]
    print(f"[sel]  minimal set ({len(minimal)}): {', '.join(minimal)}",
          file=sys.stderr)
    Xmin_c = Xspl_c[:, chosen] if chosen else Xspl_c[:, :1]
    Xminp_c = np.column_stack([Xmin_c, Xgc_c])

    if len(minimal) >= 2:
        Z = Standardizer().fit(Xmin_c).transform(Xmin_c)
        C = np.corrcoef(Z, rowvar=False)
        W("minimal_corr.tsv", pd.DataFrame(
            [dict(f1=fi, f2=fj, r=round(float(C[i, j]), 4))
             for i, fi in enumerate(minimal) for j, fj in enumerate(minimal)]))
        vifs = []
        for i, fi in enumerate(minimal):
            others = [k for k in range(len(minimal)) if k != i]
            A = np.column_stack([np.ones(len(Z)), Z[:, others]])
            beta = np.linalg.lstsq(A, Z[:, i], rcond=None)[0]
            resid = Z[:, i] - A @ beta
            r2i = 1 - np.sum(resid ** 2) / np.sum((Z[:, i] - Z[:, i].mean()) ** 2)
            vifs.append(dict(feature=fi, r2_on_others=round(float(r2i), 4),
                             vif=round(float(1 / max(1 - r2i, 1e-9)), 3)))
        W("minimal_vif.tsv", pd.DataFrame(vifs))

    drop_ix = [i for i, c in enumerate(spl) if c != "spl_pas_prox_u1"]

    MODELS = {
        "gc_only":          (Xgc, ["window_gc"], "Window GC"),
        "gc_centred":       (Xgc_c, ["window_gc"], "Window GC, gene-centred"),
        "gc_model":         (Xpair, GC_PAIR, "Published pair"),
        "gc_model_centred": (Xpair_c, GC_PAIR, "Published pair, gene-centred"),
        "splicing_only":    (Xspl, spl, "All splicing"),
        "splicing_centred": (Xspl_c, spl, "Splicing (gene-centred)"),
        "splicing_gcres":   (Xres, spl, "All splicing, GC-residualised"),
        "combined_centred": (Xcomb_c, spl + ["window_gc"],
                             "All splicing + GC, gene-centred"),
        "minimal_centred":  (Xmin_c, minimal if minimal else spl[:1],
                             f"Minimal ({len(minimal)}), gene-centred"),
        "minimal_plus_gc":  (Xminp_c, (minimal if minimal else spl[:1]) + ["window_gc"],
                             f"Minimal ({len(minimal)}) + GC, gene-centred"),
        "splicing_no_u1":   (Xspl_c[:, drop_ix], [spl[i] for i in drop_ix],
                             "Splicing minus pas_prox_u1"),
    }

    rows, coefs = [], []
    for name, (X, cols, label) in MODELS.items():
        res, w, s = evaluate(X, y, train, test, gene_codes, is_can,
                             a.l2, rng, a.boot)
        d["score_" + name] = s
        for split, r in res.items():
            rows.append(dict(model=name, label=label, split=split,
                             **{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in r.items()}))
        for c, b in zip(cols, w[1:]):
            coefs.append(dict(model=name, feature=c, coef=round(float(b), 4)))
    W("coefficients.tsv", pd.DataFrame(coefs))

    print(f"\n[loco] assembling out-of-fold scores over "
          f"{d.loco_chromosome.nunique()} chromosomes", file=sys.stderr)
    for name, (X, cols, label) in MODELS.items():
        d["oof_" + name] = loco_scores(X, y, d.loco_chromosome, a.l2)

    for name, (X, cols, label) in MODELS.items():
        s = d["oof_" + name].to_numpy()
        m = np.isfinite(s)
        au = auroc(y[m], s[m])
        apv, base = auprc(y[m], s[m])
        win, p1, rr = within_gene_fast(gene_codes[m], is_can[m], s[m])
        lo, hi = boot_ci(win, rng, a.boot)
        lo1, hi1 = boot_ci(p1, rng, a.boot)
        lom, him = boot_ci(rr, rng, a.boot)
        rows.append(dict(model=name, label=label, split="loco_oof",
                         auroc=round(au, 4), auprc=round(apv, 4),
                         auprc_baseline=round(base, 4),
                         win_rate=round(float(win.mean()), 4),
                         win_lo=round(lo, 4), win_hi=round(hi, 4),
                         p_at_1=round(float(p1.mean()), 4),
                         p_at_1_lo=round(lo1, 4), p_at_1_hi=round(hi1, 4),
                         mrr=round(float(rr.mean()), 4),
                         mrr_lo=round(lom, 4), mrr_hi=round(him, 4),
                         n_genes=len(win), n_peaks=int(m.sum())))
        if name in FIGURE_MODELS:
            print(f"[oof]  {label:<28} within-gene AUROC {win.mean():.3f} "
                  f"[{lo:.3f},{hi:.3f}]  MRR {rr.mean():.3f}  "
                  f"P@1 {p1.mean():.3f}  n={len(win):,} genes", file=sys.stderr)
    ROWS = pd.DataFrame(rows)
    W("metrics.tsv", ROWS)

    PER = None
    for name in FIGURE_MODELS:
        t = within_gene_table(d.assign(_s=d["oof_" + name]), "_s")
        t = t.rename(columns={c: name + "_" + c for c in
                              ("auc", "rank", "rr", "p1", "n_ccpa",
                               "rank_pct", "sep")})
        PER = t if PER is None else PER.merge(t, on="gene", how="inner")
    PER = PER.merge(PG[["gene", "gc_canonical", "gc_ccpa_mean", "delta_gc"]],
                    on="gene", how="left")
    PER["gc_bin"] = pd.qcut(PER.gc_canonical.rank(method="first"),
                            a.n_gc_bins, labels=False) + 1
    W("per_gene.tsv", PER)
    print(f"[gene] {len(PER):,} genes with a complete out-of-fold statistic",
          file=sys.stderr)

    disc = d[["gene", "is_canonical", "window_gc", "loco_chromosome"] +
             ["oof_" + m for m in FIGURE_MODELS]].copy()
    disc["cls"] = np.where(disc.is_canonical, "Canonical", CCLAB)
    for name in FIGURE_MODELS:
        disc["z_" + name] = standardize_by_fold(d["oof_" + name].to_numpy(),
                                                d.loco_chromosome)
    W("discriminant.tsv", disc)

    fits = []
    for name in FIGURE_MODELS:
        for lab in ("Canonical", CCLAB):
            s = disc[disc.cls == lab]
            v = s[["window_gc", "z_" + name]].dropna()
            if len(v) < 10:
                continue
            sl, ic = np.polyfit(v.window_gc, v["z_" + name], 1)
            r = float(np.corrcoef(v.window_gc, v["z_" + name])[0, 1])
            fits.append(dict(model=name, label=MODELS[name][2], cls=lab,
                             slope=round(float(sl), 4),
                             intercept=round(float(ic), 4),
                             r=round(r, 4), r2=round(r * r, 4), n=len(v)))
    FITS = pd.DataFrame(fits)
    W("discriminant_fits.tsv", FITS)
    print("\n[disc] score vs window GC, out-of-fold:", file=sys.stderr)
    for f_ in fits:
        print(f"       {f_['label']:<28} {f_['cls']:<12} "
              f"r {f_['r']:+.3f}  R2 {f_['r2']:.3f}", file=sys.stderr)

    v = PG[["gc_ccpa_mean", "gc_canonical"]].dropna()
    sl, ic = np.polyfit(v.gc_ccpa_mean, v.gc_canonical, 1)
    r = float(np.corrcoef(v.gc_ccpa_mean, v.gc_canonical)[0, 1])
    W("paradox_fit.tsv", pd.DataFrame([dict(
        slope=round(float(sl), 4), intercept=round(float(ic), 4),
        r=round(r, 4), n_genes=len(v),
        frac_below_diagonal=round(float((PG.delta_gc < 0).mean()), 4),
        mean_delta=round(float(PG.delta_gc.mean()), 4))]))

    met = []
    for name in FIGURE_MODELS:
        for b in sorted(PER.gc_bin.unique()):
            s = PER[PER.gc_bin == b]
            if len(s) < 10:
                continue
            for metric, col in [("Within-gene AUROC", name + "_auc"),
                                ("MRR", name + "_rr"), ("P@1", name + "_p1")]:
                vv = s[col].to_numpy(float)
                lo, hi = boot_ci(vv, rng, 400)
                met.append(dict(model=name, label=MODELS[name][2],
                                metric=metric, gc_bin=int(b), n_genes=len(s),
                                mean_gc=round(float(s.gc_canonical.mean()), 4),
                                value=round(float(vv.mean()), 4),
                                lo=round(lo, 4), hi=round(hi, 4)))
    MET = pd.DataFrame(met)
    W("metrics_by_decile.tsv", MET)
    for name in FIGURE_MODELS:
        s = MET[(MET["model"] == name) & (MET["metric"] == "Within-gene AUROC")]
        if len(s):
            lo_ = s[s.gc_bin == s.gc_bin.min()].iloc[0]
            hi_ = s[s.gc_bin == s.gc_bin.max()].iloc[0]
            print(f"[dec]  {MODELS[name][2]:<28} GC {lo_.mean_gc:.3f}: "
                  f"{lo_.value:.3f}   GC {hi_.mean_gc:.3f}: {hi_.value:.3f}",
                  file=sys.stderr)

    PAIR_FEATS = [f for f in (minimal if minimal else spl[:9])
                  if not f.endswith("_missing")]
    pair_rows = []
    for g, sub in d.groupby("gene", sort=False):
        can = sub[sub.is_canonical]
        oth = sub[~sub.is_canonical]
        if len(can) != 1 or len(oth) == 0:
            continue
        rec = dict(gene=g, gc_canonical=float(can.window_gc.iloc[0]),
                   n_ccpa=int(len(oth)))
        for f in PAIR_FEATS:
            rec["real__" + f] = float(can[f].iloc[0])
            rec["comp__" + f] = float(oth[f].mean())
        pair_rows.append(rec)
    PP = pd.DataFrame(pair_rows)

    for f in PAIR_FEATS:
        both = np.concatenate([PP["real__" + f].to_numpy(float),
                               PP["comp__" + f].to_numpy(float)])
        mu_, sd_ = np.nanmean(both), np.nanstd(both)
        sd_ = sd_ if sd_ > 0 else 1.0
        PP["real__" + f] = (PP["real__" + f] - mu_) / sd_
        PP["comp__" + f] = (PP["comp__" + f] - mu_) / sd_

    PP = PP.sort_values("gc_canonical").reset_index(drop=True)
    PP["row"] = np.arange(len(PP)) + 1
    long = PP.melt(id_vars=["row", "gene", "gc_canonical", "n_ccpa"],
                   var_name="key", value_name="z")
    long[["block", "feature"]] = long.key.str.split("__", expand=True)
    long["block"] = long.block.map({"real": "real 3' end",
                                    "comp": "competing sites"})
    W("gene_feature_paired.tsv", long.drop(columns=["key"]))
    print(f"\n[mat]  paired matrix: {len(PP):,} genes x {len(PAIR_FEATS)} "
          f"features x 2 blocks, unsmoothed", file=sys.stderr)
    print(f"       features shown: {', '.join(PAIR_FEATS)}", file=sys.stderr)

    groups = [v for v in d.groupby("gene").indices.values()]
    perm = []
    for name in MODELS:
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
            wr, _, _ = within_gene_fast(gene_codes[test], is_can[test], sp[test])
            wins.append(float(wr.mean()) if len(wr) else np.nan)
        perm.append(dict(model=name, label=label, n_perm=a.n_perm,
                         null_auroc=round(float(np.nanmean(aus)), 4),
                         null_win=round(float(np.nanmean(wins)), 4),
                         null_win_lo=round(float(np.nanpercentile(wins, 2.5)), 4),
                         null_win_hi=round(float(np.nanpercentile(wins, 97.5)), 4)))
    W("perm_null.tsv", pd.DataFrame(perm))

    loco = []
    for name in MODELS:
        for c in sorted(d.loco_chromosome.dropna().unique()):
            te = (d.loco_chromosome == c).to_numpy()
            s = d["oof_" + name].to_numpy()
            m = te & np.isfinite(s)
            if m.sum() < 50:
                continue
            wr, _, _ = within_gene_fast(gene_codes[m], is_can[m], s[m])
            if not len(wr):
                continue
            loco.append(dict(model=name, label=MODELS[name][2], chrom=str(c),
                             n_peaks=int(m.sum()), n_genes=len(wr),
                             win_rate=round(float(wr.mean()), 4),
                             auroc=round(auroc(y[m], s[m]), 4)))
    W("loco.tsv", pd.DataFrame(loco))

    mini = (pd.DataFrame(coefs).query("model == 'minimal_centred'")
            .merge(R2, on="feature", how="left"))
    W("minimal_features.tsv", mini)

    keep = ["gene", "chrom", "center", "strand", "is_canonical", "window_gc",
            "in_gene_holdout", "loco_chromosome"] + \
           ["score_" + m for m in MODELS] + ["oof_" + m for m in MODELS]
    W("scores.tsv", d[keep])

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_canonical=int(y.sum()), n_ccpa=int((1 - y).sum()),
                       n_peaks=int(len(d)), n_genes=int(d.gene.nunique()),
                       n_genes_scored=int(len(PER)),
                       n_features=len(spl), minimal=minimal,
                       paired_features=PAIR_FEATS,
                       window_up=a.up, window_down=a.down,
                       gc_window=2 * a.gc_half), fh, indent=2)

    print(f"\n[out]  {a.out}/  ({len(PER):,} genes, out-of-fold by chromosome)",
          file=sys.stderr)


if __name__ == "__main__":
    main()