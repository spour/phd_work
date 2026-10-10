#!/usr/bin/env python3
# score the locked 26-feature splicing model on every analysis-set peak
import sys
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

M = np.load("splice_model_26.npz", allow_pickle=True)
beta = M["beta"]; mu = M["mu"]; sd = M["sd"]; names = [str(x) for x in M["names"]]
BINLO = M["bin_lo"].astype(int); UP = int(M["up"]); DN = int(M["dn"])
assert len(names) == 26 and len(beta) == 27 and len(BINLO) == 25
assert names[-1] == "spl_upstream_max_me3"
log(f"[model] {len(names)} features, up {UP} dn {DN}, lambda {float(M['lam'])}")

U1 = np.asarray(L.U1, float); POW = np.array([int(L.POW[p]) for p in range(9)], np.int64)
THR = 4.0
log(f"[u1] {U1.size:,} 9-mers, {100*np.mean(U1>=THR):.2f}% at or above the threshold {THR}")

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; gene = np.array([str(x) for x in D["gene"][ok]])
y = D["y"][ok].astype(int); split = np.array([str(x) for x in D["split"][ok]])
A = np.load("analysis_columns.npz", allow_pickle=True)
assert np.array_equal(np.array([str(x) for x in A["gene"]]), gene), "row order differs"
me3 = A["spl_upstream_max_me3"].astype(float)
N, W = X.shape
log(f"[data] {N:,} peaks over {len(set(gene)):,} genes, window {W} nt")

log("[u1] encoding 9-mers and counting hits per 100 nt bin")
Xi = X.astype(np.int64)
bad = (Xi > 3)
Xi[bad] = 0
K = W-8
code = np.zeros((N, K), np.int64)
for p in range(9): code += Xi[:, p:p+K]*POW[p]
hit = (U1[code] >= THR)
if bad.any():
    invalid = np.zeros((N, K), bool)
    for p in range(9): invalid |= bad[:, p:p+K]
    hit &= ~invalid
    log(f"[u1] {int(invalid.sum()):,} 9-mers voided by an N token")
pos = np.arange(K)-UP
b = np.clip((pos-BINLO[0])//100, 0, 24)
F = np.zeros((N, 26))
for j in range(25):
    F[:, j] = hit[:, b == j].sum(1)*10.0
F[:, 25] = me3
log(f"[feat] built {F.shape}, mean density over bins {F[:,:25].mean():.2f} per kb")

G = {}
for g in np.unique(gene): G[g] = np.flatnonzero(gene == g)
Fc = F.copy()
for g, idx in G.items(): Fc[idx] -= F[idx].mean(0, keepdims=True)
Z = (Fc-mu)/sd
score = beta[0]+Z@beta[1:]
log(f"[score] {score.shape}, mean {score.mean():+.4f}, sd {score.std():.4f}")

def winrate(s, lab):
    if lab.sum() != 1 or len(s) < 2: return np.nan
    t = s[lab == 1][0]; o = s[lab == 0]
    return float((t > o).mean()+0.5*(t == o).mean())
te = split == "test"
wr = [winrate(score[i][te[i]], y[i][te[i]]) for i in [np.array(v) for v in G.values()]]
wr = np.array([v for v in wr if np.isfinite(v)])
log(f"[check] within-gene win rate on the test split: {wr.mean():.4f} over {len(wr):,} genes")
log( "        acceptor_options_metrics.tsv reports 0.69928 for this model; a close match confirms "
     "the rebuilt features and the locked coefficients agree")
np.savez_compressed("splice26_scores.npz", score=score, gene=gene, y=y, split=split,
                    chrom=A["chrom"], center=A["center"], strand=A["strand"])
log("[out] splice26_scores.npz")
