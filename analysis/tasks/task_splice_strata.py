#!/usr/bin/env python3
# splice_arch performance stratified by competitor context
import csv, sys, numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

NFOLD, RIDGE = 5, 1.0
NBOOT_ROW, NBOOT_GENE, NPERM = 300, 1000, 500
rng = np.random.default_rng(42)

# master, features, annotation
R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t")); NP = len(R)
gene = np.array([r["gene"] for r in R])
isc = np.array([int(float(r["is_canonical"])) for r in R])
key = [(r["chrom"], int(r["center"]), r["strand"], r["gene"]) for r in R]
F1 = np.load("hypothesis_test_features.npy")
SPL = F1[:, 66:70]
FEATNAMES = ["spl_acc", "spl_dpa", "spl_strong5", "spl_max5"]
have = np.isfinite(SPL).all(1)

FINE_CLASSES = ["first exon", "internal exon", "last exon-3'UTR",
                "first intron", "internal intron", "last intron", "intergenic-other"]
fc_all = np.load("fine_compartment.npy")
Lb = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
fmap = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): int(fc_all[i]) for i, r in enumerate(Lb)}
fine = np.array([fmap.get(k, 6) for k in key], np.int8)
assert all(k in fmap for k in key), "fine_compartment join incomplete"

At = np.load("atlas_peak_cache.npz"); a_dist = At["dist"]; a_tpm = At["tpm"]
matched = a_dist <= 25
gmap = defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)
ufrac = np.zeros(NP)
for g, ix in gmap.items():
    ix = np.array(ix); s = a_tpm[ix].sum()
    if s > 0: ufrac[ix] = a_tpm[ix] / s

K = isc == 0
INTRON = np.isin(fine, [3, 4, 5]); EXON = np.isin(fine, [1, 2])
STRATA = [
    ("all competitors",            K),
    ("intronic",                   K & INTRON),
    ("first intron",               K & (fine == 3)),
    ("internal intron",            K & (fine == 4)),
    ("last intron",                K & (fine == 5)),
    ("exonic",                     K & EXON),
    ("terminal exon",              K & (fine == 2)),
    ("atlas annotated",            K & matched),
    ("not annotated",              K & ~matched),
    ("annotated, usage >= 10%",    K & matched & (ufrac >= 0.10)),
    ("unannotated + intronic",     K & ~matched & INTRON),
]

# cv machinery (no sklearn)
allg = sorted(set(gene)); permf = np.random.default_rng(7).permutation(len(allg))
fold_of = {g: permf[i] % NFOLD for i, g in enumerate(allg)}

def fit_ridge(X, yv, lam):
    P = np.eye(X.shape[1]) * lam; P[0, 0] = 0.0; beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yv - mu) / w; XtW = X.T * w
        try: new = np.linalg.solve(XtW @ X + P, XtW @ z + P @ beta)
        except np.linalg.LinAlgError: new = np.linalg.lstsq(XtW @ X + P, XtW @ z + P @ beta, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-9: beta = new; break
        beta = new
    return beta

def cv_pred(rows):
    yv = isc[rows].astype(float); fold = np.array([fold_of[gene[i]] for i in rows])
    Xr = SPL[rows].astype(float); pred = np.full(len(rows), np.nan)
    for f in range(NFOLD):
        tr, te = fold != f, fold == f
        if tr.sum() == 0 or te.sum() == 0: continue
        Xtr, Xte = Xr[tr].copy(), Xr[te].copy()
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
        Xtr = (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd
        Xte = (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd
        Xtr = np.column_stack([np.ones(len(Xtr)), Xtr]); Xte = np.column_stack([np.ones(len(Xte)), Xte])
        pred[te] = Xte @ fit_ridge(Xtr, yv[tr], RIDGE)
    return yv, pred

def auroc(yv, s):
    order = np.argsort(s, kind="mergesort"); sv = s[order]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[order] = rk; npos = yv.sum(); nneg = n - npos
    return np.nan if npos == 0 or nneg == 0 else (rr[yv == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def auprc(yv, s):
    o = np.argsort(-s, kind="mergesort"); y = yv[o]; sv = s[o]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    last = np.r_[sv[1:] != sv[:-1], True]
    idx = np.where(last)[0]
    blk = np.searchsorted(idx, np.arange(len(y)))
    prec = (tp[idx] / k[idx])[blk]
    npos = y.sum()
    return np.nan if npos == 0 else float((prec * y).sum() / npos)

def rank_metrics(yv, pred, genes):
    by = defaultdict(list)
    for k in range(len(yv)): by[genes[k]].append(k)
    p1_obs, rr_obs, n_i, p1_mem, rr_mem = [], [], [], [], []
    for g, ix in by.items():
        ix = np.array(ix)
        if (yv[ix] == 1).sum() != 1 or len(ix) < 2: continue
        sc = pred[ix]
        if not np.isfinite(sc).all(): continue
        n = len(ix)
        o = np.argsort(-sc, kind="mergesort"); ss = -np.sort(-sc); rk = np.empty(n); i = 0
        while i < n:
            j = i
            while j + 1 < n and ss[j + 1] == ss[i]: j += 1
            rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
        r = np.empty(n); r[o] = rk
        mx = sc.max(); am = sc == mx
        p1 = np.where(am, 1.0 / am.sum(), 0.0)
        rrv = 1.0 / r
        c = int(np.where(yv[ix] == 1)[0][0])
        p1_obs.append(p1[c]); rr_obs.append(rrv[c]); n_i.append(n)
        p1_mem.append(p1); rr_mem.append(rrv)
    return (np.array(p1_obs), np.array(rr_obs), np.array(n_i, float),
            p1_mem, rr_mem)

def flat(mems):
    sizes = np.array([len(m) for m in mems]); starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return np.concatenate(mems), starts, sizes

def perm_chance(mems, nperm=NPERM):
    fl, st, sz = flat(mems)
    draws = st[None, :] + (rng.random((nperm, len(sz))) * sz[None, :]).astype(int)
    m = fl[draws].mean(1)
    return float(m.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def boot_mean_excess(obs, chance_per_gene, nboot=NBOOT_GENE):
    d = obs - chance_per_gene; n = len(d)
    b = d[rng.integers(0, n, size=(nboot, n))].mean(1)
    r = obs[rng.integers(0, n, size=(nboot, n))].mean(1)
    return (float(obs.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)),
            float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))

# run strata
OUT = open("splice_arch_strata.tsv", "w")
W = lambda *a: OUT.write("\t".join(str(x) for x in a) + "\n")
W("section", "stratum", "n_peaks", "n_genes", "n_genes_ranked", "metric", "value", "lo", "hi",
  "chance", "chance_perm", "chance_perm_lo", "chance_perm_hi", "excess", "exc_lo", "exc_hi", "base_rate")

hdrfmt = ("{:<24s} {:>7s} {:>6s} {:>6s} | {:>20s} {:>20s} | {:>18s} {:>9s} {:>9s} | {:>18s} {:>9s} {:>9s}")
print("\n=========  SPLICE_arch (4 features) by competitor stratum; canonical fixed  =========")
print(hdrfmt.format("stratum", "nPeaks", "nGene", "nRank", "AUROC [95% CI]", "AUPRC [95% CI] (base)",
                    "P@1 [CI]", "chance", "excess", "MRR [CI]", "chance", "excess"))
STORE = {}
for nm, m in STRATA:
    rows = np.where(((isc == 1) | m) & have)[0]
    npk = int((isc[rows] == 0).sum()); ncan = int((isc[rows] == 1).sum())
    ng = len({gene[i] for i in rows[isc[rows] == 0]})
    yv, pred = cv_pred(rows); genes = gene[rows]
    ar = auroc(yv, pred); ap = auprc(yv, pred); base = float(yv.mean())
    gpool = defaultdict(list)
    for k in range(len(yv)): gpool[genes[k]].append(k)
    gk = list(gpool); gidx = [np.array(gpool[g]) for g in gk]
    arb = np.empty(NBOOT_ROW); apb = np.empty(NBOOT_ROW)
    for b in range(NBOOT_ROW):
        sel = rng.integers(0, len(gk), len(gk))
        ii = np.concatenate([gidx[j] for j in sel])
        arb[b] = auroc(yv[ii], pred[ii]); apb[b] = auprc(yv[ii], pred[ii])
    alo, ahi = np.percentile(arb, [2.5, 97.5]); plo, phi = np.percentile(apb, [2.5, 97.5])
    p1o, rro, ni, p1m, rrm = rank_metrics(yv, pred, genes)
    ch_p1 = float(np.mean(1.0 / ni))
    Hn = np.array([np.sum(1.0 / np.arange(1, int(n) + 1)) for n in ni])
    ch_mrr = float(np.mean(Hn / ni))
    pp1, pp1lo, pp1hi = perm_chance(p1m); pmr, pmrlo, pmrhi = perm_chance(rrm)
    p1v, p1lo, p1hi, p1e, p1elo, p1ehi = boot_mean_excess(p1o, 1.0 / ni)
    mrv, mrlo, mrhi, mre, mrelo, mrehi = boot_mean_excess(rro, Hn / ni)
    STORE[nm] = dict(n=npk, ng=ng, ngr=len(ni))
    print(("{:<24s} {:>7d} {:>6d} {:>6d} | {:>6.4f}[{:.4f},{:.4f}] {:>6.4f}[{:.4f},{:.4f}]({:.3f}) | "
           "{:6.4f}[{:.4f},{:.4f}] {:9.4f} {:+9.4f} | {:6.4f}[{:.4f},{:.4f}] {:9.4f} {:+9.4f}").format(
        nm, npk, ng, len(ni), ar, alo, ahi, ap, plo, phi, base,
        p1v, p1lo, p1hi, ch_p1, p1e, mrv, mrlo, mrhi, ch_mrr, mre))
    NA = ""
    W("metric", nm, npk, ng, len(ni), "AUROC", f"{ar:.5f}", f"{alo:.5f}", f"{ahi:.5f}",
      "0.5", NA, NA, NA, f"{ar-0.5:.5f}", f"{alo-0.5:.5f}", f"{ahi-0.5:.5f}", NA)
    W("metric", nm, npk, ng, len(ni), "AUPRC", f"{ap:.5f}", f"{plo:.5f}", f"{phi:.5f}",
      f"{base:.5f}", NA, NA, NA, f"{ap-base:.5f}", f"{plo-base:.5f}", f"{phi-base:.5f}", f"{base:.5f}")
    W("metric", nm, npk, ng, len(ni), "P@1", f"{p1v:.5f}", f"{p1lo:.5f}", f"{p1hi:.5f}",
      f"{ch_p1:.5f}", f"{pp1:.5f}", f"{pp1lo:.5f}", f"{pp1hi:.5f}", f"{p1e:.5f}", f"{p1elo:.5f}", f"{p1ehi:.5f}", NA)
    W("metric", nm, npk, ng, len(ni), "MRR", f"{mrv:.5f}", f"{mrlo:.5f}", f"{mrhi:.5f}",
      f"{ch_mrr:.5f}", f"{pmr:.5f}", f"{pmrlo:.5f}", f"{pmrhi:.5f}", f"{mre:.5f}", f"{mrelo:.5f}", f"{mrehi:.5f}", NA)
    log(f"  [{nm}] done  perm-vs-analytic  P@1 {pp1:.4f} vs {ch_p1:.4f}   MRR {pmr:.4f} vs {ch_mrr:.4f}")

# per-feature within-gene win-rate
def winrate(val, m):
    obs, mems = [], []
    for g, ixs in gmap.items():
        ix = np.array(ixs)
        c = ix[(isc[ix] == 1) & have[ix]]; k = ix[(isc[ix] == 0) & m[ix] & have[ix]]
        if len(c) != 1 or len(k) < 1: continue
        pool = np.r_[val[c], val[k]]
        if not np.isfinite(pool).all(): continue
        M = len(pool); sp = np.sort(pool)
        less = np.searchsorted(sp, pool, "left"); eq = np.searchsorted(sp, pool, "right") - less
        wm = (less + 0.5 * (eq - 1)) / (M - 1)
        obs.append(wm[0]); mems.append(wm)
    obs = np.array(obs); n = len(obs)
    b = obs[rng.integers(0, n, size=(NBOOT_GENE, n))].mean(1)
    fl, st, sz = flat(mems)
    draws = st[None, :] + (rng.random((NPERM, len(sz))) * sz[None, :]).astype(int)
    nul = fl[draws].mean(1)
    wr = float(obs.mean())
    p = (np.sum(np.abs(nul - 0.5) >= abs(wr - 0.5)) + 1) / (NPERM + 1)
    return wr, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(nul.mean()), p, n

W("#")
W("section", "stratum", "feature", "winrate", "ci_lo", "ci_hi", "perm_null", "perm_p", "excess", "n_genes")
print("\n=========  Single-feature within-gene win-rate (canonical higher), by stratum  =========")
print("{:<24s} {:<12s} {:>8s} {:>18s} {:>9s} {:>9s} {:>8s} {:>7s}".format(
    "stratum", "feature", "win-rate", "95% CI", "excess", "perm null", "perm p", "nGenes"))
for nm, m in STRATA:
    for fi, fn in enumerate(FEATNAMES):
        wr, lo, hi, nul, p, ng = winrate(SPL[:, fi], m)
        print("{:<24s} {:<12s} {:>8.4f} [{:>7.4f},{:>7.4f}] {:>+9.4f} {:>9.4f} {:>8.4f} {:>7d}".format(
            nm, fn, wr, lo, hi, wr - 0.5, nul, p, ng))
        W("feature", nm, fn, f"{wr:.5f}", f"{lo:.5f}", f"{hi:.5f}", f"{nul:.5f}",
          f"{p:.5f}", f"{wr-0.5:+.5f}", ng)
    log(f"  [winrate {nm}] done")
OUT.close()
log("wrote splice_arch_strata.tsv  DONE_SPLICE_STRATA")
