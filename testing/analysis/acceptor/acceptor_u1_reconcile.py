#!/usr/bin/env python3
# is the acceptor a proxy for u1 depletion?
import sys, csv, math
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
import pyarrow.parquet as pq
U = list(csv.DictReader(open("fig3_ucube_meta.tsv"), delimiter="\t"))
F = list(csv.DictReader(open("fig3_peak_features.tsv"), delimiter="\t"))
assert len(U) == len(F) and [r["gene"] for r in U] == [r["gene"] for r in F], "row order differs"
assert [r["is_canonical"] for r in U] == [r["is_canonical"] for r in F]
P = pq.read_table("cpa_peaks_final.parquet").to_pandas()
mkey = {(g, c, int(x)): v for g, c, x, v in
        zip(P["gene"], P["chrom"], P["center"], P["spl_upstream_max_me3"])}
akey = {(g, c, int(x)): bool(k) for g, c, x, k in
        zip(P["gene"], P["chrom"], P["center"], P["in_analysis_set"])}
Z = np.load("cnn_wg_scores.npz", allow_pickle=True)
zkey = {(g, c, int(x)): i for i, (g, c, x) in enumerate(zip(Z["gene"], Z["chrom"], Z["center"]))}

def f(x):
    try:
        v = float(x); return v if np.isfinite(v) else np.nan
    except Exception: return np.nan
K = [(r["gene"], r["chrom"], int(r["center"])) for r in U]
G = np.array([r["gene"] for r in U])
Y = np.array([r["is_canonical"] == "1" for r in U])
ACC = np.array([mkey.get(k, np.nan) for k in K])
INSET = np.array([akey.get(k, False) for k in K])
LOG = np.array([Z["logit"][zkey[k]] if k in zkey else np.nan for k in K])
U1 = {"u1_dw_up": np.array([f(r["u1_dw_up"]) for r in U]),
      "u1_dw_dn": np.array([f(r["u1_dw_dn"]) for r in U]),
      "u1_dens_win": np.array([f(r["u1_dens_win"]) for r in F]),
      "u1_dens_strong": np.array([f(r["u1_dens_strong"]) for r in F]),
      "u1_dens_3p": np.array([f(r["u1_dens_3p"]) for r in F]),
      "interv_u1_density": np.array([f(r["interv_u1_density"]) for r in F])}
log(f"[data] {len(U):,} peaks; {Y.sum():,} real 3' ends; {INSET.sum():,} in the analysis set; "
    f"acceptor score present for {np.isfinite(ACC).sum():,}")

def pear(x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = len(x)
    r = float(np.corrcoef(x, y)[0, 1]); z = 0.5*math.log((1+r)/(1-r)); se = 1/math.sqrt(n-3)
    return r, math.tanh(z-1.96*se), math.tanh(z+1.96*se), n
def auroc(s, y):
    m = np.isfinite(s); s, y = s[m], y[m]
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); r[o] = np.arange(1, len(s)+1)
    ss = s[o]; i = 0
    while i < len(ss):
        j = i
        while j+1 < len(ss) and ss[j+1] == ss[i]: j += 1
        if j > i: r[o[i:j+1]] = (i+1+j+1)/2
        i = j+1
    n1 = int(y.sum()); n0 = len(y)-n1
    return (r[y].sum() - n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
def wingene(s, y, g):
    m = np.isfinite(s); s, y, g = s[m], y[m], g[m]
    out = []
    for gg in np.unique(g):
        k = g == gg; a = s[k & y] if (k & y).any() else None; b = s[k & ~y]
        if a is None or not len(a) or not len(b): continue
        w = (a[:, None] > b[None, :]).mean() + 0.5*(a[:, None] == b[None, :]).mean()
        out.append(w)
    return float(np.mean(out)), len(out)
def irls(Xd, y, iters=60):
    Xd = np.column_stack([np.ones(len(Xd)), Xd]); b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        e = Xd @ b; p = 1/(1+np.exp(-np.clip(e, -30, 30))); w = np.clip(p*(1-p), 1e-9, None)
        z = e + (y - p)/w
        b_new = np.linalg.solve(Xd.T @ (Xd*w[:, None]) + 1e-9*np.eye(Xd.shape[1]), Xd.T @ (w*z))
        if np.max(np.abs(b_new-b)) < 1e-9: b = b_new; break
        b = b_new
    e = Xd @ b; p = 1/(1+np.exp(-np.clip(e, -30, 30)))
    w = np.clip(p*(1-p), 1e-9, None)
    cov = np.linalg.inv(Xd.T @ (Xd*w[:, None]) + 1e-9*np.eye(Xd.shape[1]))
    return b, np.sqrt(np.diag(cov)), Xd @ b

log("\nTEST 1  is a strong acceptor found in U1-depleted context?")
log(f"  {'U1 feature':<20} {'r with acceptor':>16} {'95% CI':>20} {'n':>8}")
rows = []
for k, v in U1.items():
    r, lo, hi, n = pear(ACC, v)
    log(f"  {k:<20} {r:>+16.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>20} {n:>8,}")
    rows.append(("test1", k, "r_with_acceptor", r, lo, hi, n))

BEST = "u1_dw_up"
m = INSET & np.isfinite(ACC) & np.isfinite(U1[BEST])
a = ACC[m]; u = U1[BEST][m]; y = Y[m]; g = G[m]; lg = LOG[m]
za = (a-a.mean())/a.std(); zu = (u-u.mean())/u.std()
log(f"\nTEST 2  analysis set, complete cases: {m.sum():,} peaks, {y.sum():,} real 3' ends, "
    f"{len(np.unique(g)):,} genes.  U1 feature = {BEST}")
log(f"  {'model':<26} {'pooled AUROC':>13} {'within-gene':>12} {'genes':>7}")
for nm, s in (("acceptor alone", a), ("U1 depletion alone (-U1)", -u)):
    wg, ng = wingene(s, y, g)
    log(f"  {nm:<26} {auroc(s, y):>13.4f} {wg:>12.4f} {ng:>7,}")
    rows.append(("test2", nm, "auroc", auroc(s, y), wg, np.nan, int(m.sum())))
FITS = {}
for nm, Xd in (("acceptor", za[:, None]), ("U1", zu[:, None]),
               ("acceptor + U1", np.column_stack([za, zu]))):
    b, se, sc = irls(Xd, y.astype(float))
    FITS[nm] = (b, se, sc)
    wg, ng = wingene(sc, y, g)
    co = "  ".join(f"{n2} {b[i+1]:+.4f} +/- {se[i+1]:.4f} (z {b[i+1]/se[i+1]:+.1f})"
                   for i, n2 in enumerate(nm.split(" + ")))
    log(f"  {'logistic ' + nm:<26} {auroc(sc, y):>13.4f} {wg:>12.4f} {ng:>7,}   {co}")
    rows.append(("test2", "logistic " + nm, "auroc", auroc(sc, y), wg, np.nan, int(m.sum())))
ba, sa, _ = FITS["acceptor"]; bb, sb, _ = FITS["acceptor + U1"]
bu, su, _ = FITS["U1"]
log(f"\n  acceptor coefficient per SD:  alone {ba[1]:+.4f} +/- {sa[1]:.4f}   "
    f"with U1 in {bb[1]:+.4f} +/- {sb[1]:.4f}   retained {100*bb[1]/ba[1]:.0f}%")
log(f"  U1 coefficient per SD:        alone {bu[1]:+.4f} +/- {su[1]:.4f}   "
    f"with acceptor in {bb[2]:+.4f} +/- {sb[2]:.4f}   retained {100*bb[2]/bu[1]:.0f}%")
log(f"  AUROC gained by adding U1 to the acceptor: "
    f"{auroc(FITS['acceptor + U1'][2], y) - auroc(FITS['acceptor'][2], y):+.4f}")
log(f"  AUROC gained by adding the acceptor to U1: "
    f"{auroc(FITS['acceptor + U1'][2], y) - auroc(FITS['U1'][2], y):+.4f}")
rows.append(("test2", "acceptor coef retained with U1 in", "frac", bb[1]/ba[1], np.nan, np.nan, 0))
rows.append(("test2", "U1 coef retained with acceptor in", "frac", bb[2]/bu[1], np.nan, np.nan, 0))

log("\nTEST 3  split the acceptor into the part U1 predicts and the residual")
UM = np.column_stack([np.ones(m.sum())] + [((U1[k][m]-np.nanmean(U1[k][m]))/np.nanstd(U1[k][m]))
                                            for k in U1 if np.isfinite(U1[k][m]).all()])
coef, *_ = np.linalg.lstsq(UM, a, rcond=None)
shared = UM @ coef; resid = a - shared
log(f"  U1 features used: {[k for k in U1 if np.isfinite(U1[k][m]).all()]}")
log(f"  they explain R2 = {1-((a-shared)**2).sum()/((a-a.mean())**2).sum():.4f} of acceptor strength")
lgm = np.isfinite(lg)
for nm, v in (("raw acceptor", a), ("its U1-predicted part", shared), ("its residual", resid)):
    r, lo, hi, n = pear(v[lgm], lg[lgm])
    ra, la_, ha, _ = pear(v[lgm & y], lg[lgm & y])
    log(f"  CNN logit vs {nm:<22} all r = {r:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
        f"within real ends r = {ra:+.4f} [{la_:+.4f}, {ha:+.4f}]")
    rows.append(("test3", nm, "r_with_cnn", r, lo, hi, n))
    log(f"  {'':13} {'':22}   AUROC real vs competitor = {auroc(v, y):.4f}")
    rows.append(("test3", nm, "auroc", auroc(v, y), np.nan, np.nan, int(m.sum())))
with open("acceptor_u1_reconcile.tsv", "w") as fh:
    fh.write("test\titem\tstat\tvalue\tlo\thi\tn\n")
    for r_ in rows: fh.write("\t".join(str(x) for x in r_) + "\n")
log("\n[out] acceptor_u1_reconcile.tsv")
