#!/usr/bin/env python3
# leave-one-chromosome-out scores for the 26-feature splice model
import sys
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
U1 = np.asarray(L.U1, float); POW = np.array([int(L.POW[p]) for p in range(9)], np.int64)
THR = 4.0; UP = 2000
SM = np.load("splice_model_26.npz", allow_pickle=True)
BINLO = SM["bin_lo"].astype(int); LAM = float(SM["lam"])
names = [str(x) for x in SM["names"]]
log(f"[model] 26 features, ridge lambda {LAM}, refit per fold")

A = np.load("analysis_columns.npz", allow_pickle=True)
gene = np.array([str(x) for x in A["gene"]]); chrom = np.array([str(x) for x in A["chrom"]])
center = A["center"].astype(np.int64); strand = np.array([str(x) for x in A["strand"]])
y = A["is_canonical"].astype(int); me3 = A["spl_upstream_max_me3"].astype(float)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]
assert np.array_equal(np.array([str(x) for x in D["gene"][ok]]), gene)
N, W = X.shape
log(f"[data] {N:,} peaks, {len(set(gene)):,} genes, {len(set(chrom))} chromosomes")

log("[feat] rebuilding the 25 U1 density bins")
Xi = X.astype(np.int64); bad = Xi > 3; Xi[bad] = 0
K = W-8
code = np.zeros((N, K), np.int64)
for p in range(9): code += Xi[:, p:p+K]*POW[p]
hit = U1[code] >= THR
inv = np.zeros((N, K), bool)
for p in range(9): inv |= bad[:, p:p+K]
hit &= ~inv
pos = np.arange(K)-UP
b = np.clip((pos-BINLO[0])//100, 0, 24)
F = np.zeros((N, 26))
for j in range(25): F[:, j] = hit[:, b == j].sum(1)*10.0
F[:, 25] = me3
del code, hit, inv, Xi
log(f"[feat] {F.shape} built")

import collections
gi = collections.defaultdict(list)
for i, g in enumerate(gene): gi[g].append(i)
Fc = F.copy()
for g, idx in gi.items():
    idx = np.array(idx); Fc[idx] -= F[idx].mean(0, keepdims=True)

def irls(Xd, yy, lam, it=80):
    Z = np.column_stack([np.ones(len(Xd)), Xd]); bt = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-np.clip(Z@bt, -30, 30)))
        w = np.clip(p*(1-p), 1e-6, None)
        z = Z@bt+(yy-p)/w
        R = np.eye(Z.shape[1])*lam; R[0, 0] = 0
        bt = np.linalg.solve(Z.T@(Z*w[:, None])+R, Z.T@(w*z))
    return bt

score = np.full(N, np.nan)
chroms = sorted(set(chrom))
log(f"[loco] {len(chroms)} folds")
for ci, c in enumerate(chroms, 1):
    te = chrom == c; tr = ~te
    if te.sum() == 0 or y[tr].sum() < 10: continue
    mu = Fc[tr].mean(0); sd = Fc[tr].std(0); sd[sd < 1e-9] = 1.0
    bt = irls((Fc[tr]-mu)/sd, y[tr].astype(float), LAM)
    score[te] = bt[0]+((Fc[te]-mu)/sd)@bt[1:]
    if ci % 6 == 0 or ci == len(chroms):
        log(f"   {ci}/{len(chroms)} folds, {int(np.isfinite(score).sum()):,} peaks scored")
assert np.isfinite(score).all(), "some peaks were never in a held-out fold"

def wr_p1(s, yy):
    if yy.sum() != 1 or len(yy) < 2: return None, None
    t = s[yy == 1][0]; o = s[yy == 0]
    return float((t > o).mean()+0.5*(t == o).mean()), int((o < t).all())
res = [wr_p1(score[np.array(idx)], y[np.array(idx)]) for idx in gi.values()]
res = [r for r in res if r[0] is not None]
log(f"\n[check] LOCO within-gene over all {len(res):,} analysis genes: "
    f"win rate {np.mean([r[0] for r in res]):.4f}   P@1 {np.mean([r[1] for r in res]):.4f}")
old = np.load("splice26_scores.npz", allow_pickle=True)
sp = np.array([str(x) for x in old["split"]])
m = sp == "test"
resT = [wr_p1(score[np.array(idx)], y[np.array(idx)]) for g, idx in gi.items()
        if sp[idx[0]] == "test"]
resT = [r for r in resT if r[0] is not None]
log(f"        on the {len(resT):,} test genes: win rate {np.mean([r[0] for r in resT]):.4f}   "
    f"P@1 {np.mean([r[1] for r in resT]):.4f}")
log(f"        locked-coefficient model on the same test genes reported 0.6993 / 0.4720")
log(f"        correlation with the locked scores: r = {np.corrcoef(score, old['score'])[0,1]:+.4f}")
np.savez_compressed("splice26_loco_scores.npz", score=score, gene=gene, chrom=chrom,
                    center=center, strand=strand, y=y, split=sp)
log("[out] splice26_loco_scores.npz")
