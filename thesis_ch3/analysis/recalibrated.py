#!/usr/bin/env python3
# do the splice-feature results hold when the features are calibrated?
import os, sys, math, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
src = open(os.path.join(HERE, "stats_ch3.py")).read(); cut = src.index("# ---------------------------------------------------------------- data")
ns = {"__file__": os.path.join(HERE, "stats_ch3.py")}; exec(src[:cut], ns); wilcoxon = ns["wilcoxon"]
RNG = np.random.default_rng(1); ROWS = []
def out(sec, what, a, b="", c="", d=""):
    ROWS.append((sec, what, a, b, c, d)); log(f"[{sec}] {what:60s} {a}  {b}  {c}  {d}")
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]; split = D["split"]
Z = np.load(os.path.join(PUB, "binned_profiles.npz"), allow_pickle=True); P = Z["P"].astype(float); PS = Z["PS"].astype(float)
CH = list(Z["channels"])
gidx = collections.defaultdict(list)
for i, gg in enumerate(g): gidx[gg].append(i)
gidx = [np.array(v) for v in gidx.values()]
UP5 = slice(15, 20)
for c in ("u1_dens4", "u1_dens7", "me5_dens8"):
    j = CH.index(c)
    for lab, M in (("raw", P[:, j]), ("excess over shuffle", P[:, j] - PS[:, j])):
        for reg, sl in (("upstream 500 nt", UP5), ("whole window", slice(0, 25))):
            v = M[:, sl].mean(1)
            d = np.array([v[ii][y[ii] == 1][0] - v[ii][y[ii] == 0].mean() for ii in gidx])
            z, p, n = wilcoxon(d)
            out("1", f"{c} {lab}, {reg}: real minus mean competitor (per kb)", f"mean {d.mean():+.3f}, median {np.median(d):+.3f}",
                f"below zero {np.mean(d < 0):.3f}", f"z = {z:.2f}, r = {abs(z)/math.sqrt(n):.3f}", f"P = {p:.2g}")
def winrate(v):
    w = []; p1 = []
    for ii in gidx:
        r = v[ii][y[ii] == 1]; cc = v[ii][y[ii] == 0]
        if np.isnan(r).any(): continue
        cc = np.nan_to_num(cc, nan=-1e9)
        w.append(((r[:, None] > cc[None, :]).sum() + 0.5 * (r[:, None] == cc[None, :]).sum()) / (len(r) * len(cc)))
        p1.append(float(r.max() > cc.max()))
    return np.mean(w), np.mean(p1), len(w)
A = np.load(os.path.join(PUB, "acceptor_scalar.npz")); A4 = np.load(os.path.join(PUB, "acceptor4_scalar.npz"))
for lab, v in (("MaxEnt score3 maximum, -2000 to -126", A["acc"].astype(float)), ("structured acceptor, total", A4["raw_tot"].astype(float)),
               ("structured acceptor, AG term", A4["raw_ag"].astype(float)), ("structured acceptor, branch point", A4["raw_bp"].astype(float)),
               ("structured acceptor, polypyrimidine", A4["raw_py"].astype(float))):
    w, p1, n = winrate(v); out("2", f"acceptor score alone: {lab}", f"win-rate {w:.3f}", f"P@1 {p1:.3f}", f"genes {n}")
def centre(F):
    F = F.copy()
    for ii in gidx: F[ii] -= F[ii].mean(0)
    return F
def irls(Xm, yv, lam, it=30):
    Xb = np.column_stack([np.ones(len(Xm)), Xm]); b = np.zeros(Xb.shape[1]); R = lam * np.eye(Xb.shape[1]); R[0, 0] = 0
    npos = yv.sum(); wpos = (len(yv) - npos) / npos; sw = np.where(yv == 1, wpos, 1.0)
    for _ in range(it):
        eta = Xb @ b; pr = 1 / (1 + np.exp(-eta)); Wt = sw * pr * (1 - pr) + 1e-9
        H = Xb.T @ (Xb * Wt[:, None]) + R; gr = Xb.T @ (sw * (yv - pr)) - R @ b
        step = np.linalg.solve(H, gr); b += step
        if np.abs(step).max() < 1e-7: break
    return b
def within(sc, idx):
    sub = collections.defaultdict(list)
    for i in idx: sub[g[i]].append(i)
    w = []; p1 = []
    for ii in sub.values():
        ii = np.array(ii); r = sc[ii][y[ii] == 1]; cc = sc[ii][y[ii] == 0]
        if len(r) == 0 or len(cc) == 0: continue
        w.append(((r[:, None] > cc[None, :]).sum() + 0.5 * (r[:, None] == cc[None, :]).sum()) / (len(r) * len(cc)))
        p1.append(float(r.max() > cc.max()))
    return np.array(w), np.array(p1)
tr = np.flatnonzero(split == "train"); va = np.flatnonzero(split == "val"); te = np.flatnonzero(split == "test")
acc = np.nan_to_num(A["acc"].astype(float), nan=np.nanmin(A["acc"]))
j4, j7, j8 = CH.index("u1_dens4"), CH.index("u1_dens7"), CH.index("me5_dens8")
VAR = {
  "U1 >= 4 (reported 26-feature model)": np.column_stack([P[:, j4], acc]),
  "U1 >= 4, excess over shuffle": np.column_stack([P[:, j4] - PS[:, j4], acc]),
  "U1 >= 7": np.column_stack([P[:, j7], acc]),
  "U1 >= 7, excess over shuffle": np.column_stack([P[:, j7] - PS[:, j7], acc]),
  "MaxEnt donors >= 8": np.column_stack([P[:, j8], acc]),
  "MaxEnt donors >= 8, excess over shuffle": np.column_stack([P[:, j8] - PS[:, j8], acc]),
  "all three, raw and excess": np.column_stack([P[:, j4], P[:, j7], P[:, j8], P[:, j4] - PS[:, j4], P[:, j7] - PS[:, j7], P[:, j8] - PS[:, j8], acc]),
  "structured acceptor instead of MaxEnt": np.column_stack([P[:, j4], np.nan_to_num(A4["raw_tot"].astype(float))]),
}
for name, F in VAR.items():
    Fc = centre(F); mu = Fc[tr].mean(0); sd = Fc[tr].std(0); sd[sd == 0] = 1; Fz = (Fc - mu) / sd
    best = None
    for lam in (0.01, 0.1, 1, 10, 100, 1000):
        b = irls(Fz[tr], y[tr], lam); wv, _ = within(np.column_stack([np.ones(len(Fz)), Fz]) @ b, va)
        if best is None or wv.mean() > best[0]: best = (wv.mean(), lam, b)
    b = np.nan_to_num(irls(Fz[np.r_[tr, va]], y[np.r_[tr, va]], best[1]))
    wt, pt = within(np.column_stack([np.ones(len(Fz)), Fz]) @ b, te)
    bw = [wt[RNG.integers(0, len(wt), len(wt))].mean() for _ in range(1000)]; bp = [pt[RNG.integers(0, len(pt), len(pt))].mean() for _ in range(1000)]
    out("3", f"splice model, {name} ({F.shape[1]} features)", f"test win-rate {wt.mean():.3f} [{np.percentile(bw,2.5):.3f}, {np.percentile(bw,97.5):.3f}]",
        f"P@1 {pt.mean():.3f} [{np.percentile(bp,2.5):.3f}, {np.percentile(bp,97.5):.3f}]", f"lambda {best[1]}", f"genes {len(wt)}")
with open(os.path.join(ROOT, "data/recalibrated.tsv"), "w") as fh:
    fh.write("section\tcomparison\tresult\tresult2\tresult3\tresult4\n")
    for r in ROWS: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/recalibrated.tsv")
