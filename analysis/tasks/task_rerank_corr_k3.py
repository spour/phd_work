#!/usr/bin/env python3
# corrected set at k = 3, for continuity with the previous primary model
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, HOLD_FRAC, RIDGE, NBOOT = 42, 0.20, 1.0, 400
rng = np.random.default_rng(SEED)
FINAL = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
X = np.column_stack([D[n].to_numpy(float) for n in FINAL])
ROW = np.load("corr_rowmask.npz")["ROW"]
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
allg = np.array(sorted(set(gene)))
hold_g = set(rng.choice(allg, size=int(round(HOLD_FRAC * len(allg))), replace=False))
is_hold = np.array([g in hold_g for g in gene])
CHROMS = sorted(set(chrom))

def fit_ridge(Xt, yt, lam=RIDGE):
    P = np.eye(Xt.shape[1]) * lam; P[0, 0] = 0.0; b = np.zeros(Xt.shape[1])
    for _ in range(80):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yt - mu) / w; XtW = Xt.T * w
        nb = np.linalg.solve(XtW @ Xt + P, XtW @ z + P @ b)
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b
def prep(Xtr, Xte):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
    return (np.column_stack([np.ones(len(Xtr)), (np.where(np.isfinite(Xtr), Xtr, mu) - mu) / sd]),
            np.column_stack([np.ones(len(Xte)), (np.where(np.isfinite(Xte), Xte, mu) - mu) / sd]))
pred = np.full(len(y), np.nan)
for c in CHROMS:
    te = ROW & is_hold & (chrom == c); tr = ROW & ~is_hold & (chrom != c)
    if te.sum() == 0 or tr.sum() == 0: continue
    A, B = prep(X[tr], X[te]); pred[te] = B @ fit_ridge(A, y[tr])
hits, ns = [], []
for g, ix in gi.items():
    if not is_hold[ix[0]]: continue
    ix = ix[ROW[ix]]
    if len(ix) < 2 or not np.isfinite(pred[ix]).all(): continue
    n = len(ix); j = np.argsort(-(pred[ix] + rng.random(n) * 1e-12), kind="mergesort")
    hits.append(1.0 if j[0] == int(np.argmax(y[ix])) else 0.0); ns.append(n)
hits = np.array(hits); ns = np.array(ns, float)
b = hits[rng.integers(0, len(hits), size=(NBOOT, len(hits)))].mean(1)
print(f"k=3 on the corrected set: holdout genes {len(hits)}, P@1 {hits.mean():.4f} "
      f"[{np.percentile(b,2.5):.4f},{np.percentile(b,97.5):.4f}] chance {np.mean(1.0/ns):.4f} "
      f"excess {hits.mean()-np.mean(1.0/ns):+.4f}")
dev = np.where(ROW & ~is_hold)[0]
A, _ = prep(X[dev], X[dev]); beta = fit_ridge(A, y[dev])[1:]
devg = [ix[ROW[ix]] for g, ix in gi.items() if not is_hold[ix[0]] and ROW[ix].sum() >= 2]
BB = np.empty((300, len(FINAL)))
for t in range(300):
    ii = np.concatenate([devg[j] for j in rng.integers(0, len(devg), len(devg))])
    Ab, _ = prep(X[ii], X[ii]); BB[t] = fit_ridge(Ab, y[ii])[1:]
with open("rerank_corr_k3_coefs.tsv", "w") as fh:
    fh.write("feature\tbeta_std\tci_lo\tci_hi\tsign_flip_frac\n")
    for a, nm in enumerate(FINAL):
        lo, hi = np.percentile(BB[:, a], [2.5, 97.5]); fl = float(np.mean(np.sign(BB[:, a]) != np.sign(beta[a])))
        print(f"  {nm:<24} {beta[a]:>+8.4f} [{lo:>+8.4f},{hi:>+8.4f}]  flip {fl:.3f}")
        fh.write(f"{nm}\t{beta[a]:.5f}\t{lo:.5f}\t{hi:.5f}\t{fl:.4f}\n")
log("DONE_CORR_K3")
