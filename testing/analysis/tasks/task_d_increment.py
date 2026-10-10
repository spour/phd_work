#!/usr/bin/env python3
# what the composition feature adds to the acceptor, paired and tested
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT, NPERM = 42, 0.20, 1.0, 2000, 1000
rng = np.random.default_rng(SEED)

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
ROW = np.load("upstream_rowmask.npz")["ROW"]; N = len(D)
me3 = D["spl_upstream_max_me3"].to_numpy(float)
CS = np.load("control_contrast_scan.npz", allow_pickle=True); CSN = list(CS["names"])
gcc = CS["X"][:, CSN.index("gc_distal")] - CS["X"][:, CSN.index("gc_proximal")]

gi = defaultdict(list)
for i in range(N): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]
HG = [(g, ix) for g, ix in GEN if is_hold[ix[0]]]
GIDX = [ix for _, ix in HG]; HROWS = np.concatenate(GIDX)

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
def midrank(s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk
    return rr
def auroc(idx, s):
    yv = y[idx]; rr = midrank(s[idx]); P = yv.sum(); n = len(yv)
    return float((rr[yv == 1].sum() - P * (P + 1) / 2) / (P * (n - P)))
def null_exact(score, groups):
    rows = np.concatenate(groups); rr = midrank(score[rows])
    Gn = len(groups); n = len(rows)
    ng = np.array([len(ix) for ix in groups]); off = np.r_[0, np.cumsum(ng)[:-1]]
    mr = np.array([rr[off[k]:off[k] + ng[k]].mean() for k in range(Gn)])
    return float((mr.mean() - (Gn + 1) / 2) / (n - Gn))
def winrate_vec(v, groups):
    out = np.full(len(groups), np.nan)
    for t, ix in enumerate(groups):
        s = v[ix]; c = int(np.argmax(y[ix])); d = np.delete(s, c); d = d[np.isfinite(d)]
        if np.isfinite(s[c]) and len(d): out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
    return out

P1 = loco_hold(me3[:, None])
P2 = loco_hold(np.column_stack([me3, gcc]))
w1 = winrate_vec(P1, GIDX); w2 = winrate_vec(P2, GIDX)
a1 = auroc(HROWS, P1); a2 = auroc(HROWS, P2)
n1 = null_exact(P1, GIDX); n2 = null_exact(P2, GIDX)
e1, e2 = a1 - n1, a2 - n2
print(f"\nholdout {len(GIDX):,} genes, {len(HROWS):,} peaks")
print(f"  acceptor alone           win-rate {np.nanmean(w1):.4f}  AUROC {a1:.4f}  null {n1:.4f}  excess {e1:+.4f}")
print(f"  acceptor + gc_contrast   win-rate {np.nanmean(w2):.4f}  AUROC {a2:.4f}  null {n2:.4f}  excess {e2:+.4f}")

SEL = [rng.integers(0, len(GIDX), len(GIDX)) for _ in range(NBOOT)]
dw = np.empty(NBOOT); da = np.empty(NBOOT)
for t, sel in enumerate(SEL):
    ii = np.concatenate([GIDX[j] for j in sel])
    gg = [GIDX[j] for j in sel]
    dw[t] = np.nanmean(w2[sel]) - np.nanmean(w1[sel])
    da[t] = (auroc(ii, P2) - null_exact(P2, gg)) - (auroc(ii, P1) - null_exact(P1, gg))

def report(nm, obs, b):
    lo, hi = np.percentile(b, [2.5, 97.5])
    pr = 2 * min(float(np.mean(b <= 0)), float(np.mean(b >= 0)))
    pr = max(pr, 1.0 / NBOOT)
    txt = f"< {1/NBOOT:.0e}" if float(np.mean(b <= 0)) == 0 or float(np.mean(b >= 0)) == 0 else f"= {pr:.2e}"
    print(f"  increment in {nm:<14} {obs:+.4f} [{lo:+.4f},{hi:+.4f}]  two-sided bootstrap p {txt}"
          f"  ({NBOOT:,} gene-clustered replicates)")
    return lo, hi, pr, txt
print(f"\npaired increment of acceptor + gc_contrast over acceptor alone:")
lw, hw, pw, tw = report("win-rate", float(np.nanmean(w2) - np.nanmean(w1)), dw)
la, ha, pa, ta = report("AUROC excess", float(e2 - e1), da)
with open("d_increment.tsv", "w") as fh:
    fh.write("quantity\tacceptor_alone\tacceptor_plus_gc\tincrement\tci_lo\tci_hi\tp\tp_text\tnboot\n")
    fh.write(f"win-rate\t{np.nanmean(w1):.5f}\t{np.nanmean(w2):.5f}\t{np.nanmean(w2)-np.nanmean(w1):.5f}\t"
             f"{lw:.5f}\t{hw:.5f}\t{pw:.6g}\t{tw}\t{NBOOT}\n")
    fh.write(f"AUROC excess\t{e1:.5f}\t{e2:.5f}\t{e2-e1:.5f}\t{la:.5f}\t{ha:.5f}\t{pa:.6g}\t{ta}\t{NBOOT}\n")
log("DONE_D_INCREMENT")
