#!/usr/bin/env python3
# the window sweep, redone for the model that is actually used
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NPERM = 42, 0.20, 1.0, 400
rng = np.random.default_rng(SEED)
WS = [500, 1000, 2000, 5000]; SPLITS = [200, 400, 700, 1000, 1300]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("upstream_rowmask.npz")["ROW"]; N = len(D)
Z = np.load("anchor_sweep.npz", allow_pickle=True); ZN = list(Z["names"])
ME3 = {w: Z["X"][:, ZN.index(f"me3_W{w}")] for w in WS}
PB = np.load("binned_features178.npz"); BN = [str(s) for s in PB["names"]]; BX = PB["X"][ridx].astype(float)
GCB = [BN.index(f"bin{b}_gc") for b in range(1, 26)]
CS = np.load("control_contrast_scan.npz", allow_pickle=True); CSN = list(CS["names"])
gcc_exact = CS["X"][:, CSN.index("gc_distal")] - CS["X"][:, CSN.index("gc_proximal")]
def gcc_split(s):
    k = (2000 - s) // 100
    d = np.nanmean(BX[:, [GCB[b] for b in range(0, k)]], 1)
    p = np.nanmean(BX[:, [GCB[b] for b in range(k, 19)]], 1)
    return d - p

gi = defaultdict(list)
for i in range(N): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]
HG = [(g, ix) for g, ix in GEN if is_hold[ix[0]]]
HROWS = np.concatenate([ix for _, ix in HG]); IDX = np.where(ROW)[0]

def midrank(s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk
    return rr
def auroc(yv, s):
    m = np.isfinite(s); yv = yv[m]; rr = midrank(s[m]); P = yv.sum(); n = len(yv)
    return float((rr[yv == 1].sum() - P * (P + 1) / 2) / (P * (n - P)))
def null_exact(score, G):
    gl = []
    for g, ix in G:
        f = ix[np.isfinite(score[ix])]
        if len(f) >= 2 and (y[f] == 1).sum() == 1: gl.append(f)
    rows = np.concatenate(gl); rr = midrank(score[rows]); Gn = len(gl); n = len(rows)
    ng = np.array([len(ix) for ix in gl]); off = np.r_[0, np.cumsum(ng)[:-1]]
    mr = np.array([rr[off[k]:off[k] + ng[k]].mean() for k in range(Gn)])
    return float((mr.mean() - (Gn + 1) / 2) / (n - Gn))
def winrate(v, G):
    out = np.full(len(G), np.nan)
    for t, (g, ix) in enumerate(G):
        s = v[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d): out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
    return out
def fit_ridge(Xt, yt, lam=RIDGE):
    P = np.eye(Xt.shape[1]) * lam; P[0, 0] = 0.0; b = np.zeros(Xt.shape[1])
    for _ in range(80):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yt - mu) / w; XtW = Xt.T * w
        try: nb = np.linalg.solve(XtW @ Xt + P, XtW @ z + P @ b)
        except np.linalg.LinAlgError: nb = np.linalg.lstsq(XtW @ Xt + P, XtW @ z + P @ b, rcond=None)[0]
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b
def prep(Xtr, Xte):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return (np.column_stack([np.ones(len(Xtr)), (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd]),
            np.column_stack([np.ones(len(Xte)), (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd]))
def loco_hold(X):
    pred = np.full(N, np.nan)
    for c in CHROMS:
        te = ROW & is_hold & (chrom == c); tr = ROW & ~is_hold & (chrom != c)
        if te.sum() == 0 or tr.sum() == 0: continue
        A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
    return pred

ap = gcc_split(400)
m = ROW & np.isfinite(ap) & np.isfinite(gcc_exact)
print(f"binned s=400 approximation vs the deployed -437/-126 contrast: r = "
      f"{np.corrcoef(ap[m], gcc_exact[m])[0,1]:.4f}")
print(f"\n{'acc W':>6} {'GC split':>9} {'AUROC':>7} {'null':>7} {'excess':>8} {'win-rate':>9}")
G = {}
with open("gc_sweep.tsv", "w") as fh:
    fh.write("acc_W\tgc_split\tauroc\tnull\texcess\twinrate\n")
    for w in WS:
        for s in SPLITS:
            X = np.column_stack([ME3[w], gcc_split(s)])
            p = loco_hold(X)
            a = auroc(y[HROWS], p[HROWS]); nu = null_exact(p, HG)
            wr = float(np.nanmean(winrate(p, HG)))
            G[(w, s)] = (a, a - nu, wr)
            print(f"{w:>6} {s:>9} {a:>7.4f} {nu:>7.4f} {a-nu:>+8.4f} {wr:>9.4f}")
            fh.write(f"{w}\t{s}\t{a:.5f}\t{nu:.5f}\t{a-nu:.5f}\t{wr:.5f}\n")
ex = np.array([[G[(w, s)][1] for s in SPLITS] for w in WS])
wr = np.array([[G[(w, s)][2] for s in SPLITS] for w in WS])
best = max(G, key=lambda k: G[k][1])
print(f"\nbest cell W={best[0]}, split={best[1]}: excess {G[best][1]:+.4f}, win-rate {G[best][2]:.4f}")
print(f"deployed cell W=2000, split=400: excess {G[(2000,400)][1]:+.4f}, win-rate {G[(2000,400)][2]:.4f}")
print(f"excess spread over the grid {ex.min():+.4f} to {ex.max():+.4f}; "
      f"row (W) spread {np.ptp(ex.mean(1)):.4f}, column (split) spread {np.ptp(ex.mean(0)):.4f}")
print(f"win-rate spread {wr.min():.4f} to {wr.max():.4f}")
log("DONE_GC_SWEEP")
