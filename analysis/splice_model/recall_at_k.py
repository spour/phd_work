#!/usr/bin/env python3
# within-gene recall@k for the 26-feature model on the analysis set
import sys
import numpy as np
import binned_common as K
from binned_common import log

KMAX = 8
Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]
X = np.column_stack([P[:, CH.index("u1_dens4"), :].astype(float), K.ACC])
assert X.shape[1] == 26
gene = K.gene; y = K.y; chrom = K.C0["chrom"][K.ok]
N = len(y)
GENES = np.array(sorted(set(gene)))
GI = {g: np.flatnonzero(gene == g) for g in GENES}
PAIR = {g: (ix[y[ix] == 1], ix[y[ix] == 0]) for g, ix in GI.items()}
PAIR = {g: v for g, v in PAIR.items() if len(v[0]) and len(v[1])}
GL = np.array(sorted(PAIR))
assert len(GL) == 4847, f"expected 4,847 analysis-set genes, got {len(GL)}"
log(f"[scope] {len(GL)} analysis-set genes, {N} peaks, {len(set(chrom))} chromosomes")

def groups(ii):
    by = {}
    for k_, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k_] == 1 else c).append(k_)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]

Cg = K.centre_by_gene(X)
def fe(fi, ei, lam):
    m = np.nanmean(Cg[fi], 0); m = np.where(np.isfinite(m), m, 0.0)
    s = np.nanstd(Cg[fi], 0); s[~np.isfinite(s) | (s == 0)] = 1.0
    Zm = (np.where(np.isfinite(Cg), Cg, m) - m) / s
    b = K.fit_ridge(np.column_stack([np.ones(len(fi)), Zm[fi]]), y[fi], lam)
    return np.column_stack([np.ones(len(ei)), Zm[ei]]) @ b

sc = np.full(N, np.nan); lams = []
for ch in sorted(set(chrom)):
    te = np.flatnonzero(chrom == ch); tr = np.flatnonzero(chrom != ch)
    trg = np.array(sorted(set(gene[tr]))); rs = np.random.default_rng(K.SEED)
    sh = trg.copy(); rs.shuffle(sh); ivg = set(sh[:int(round(0.25 * len(sh)))])
    iv = tr[np.array([g in ivg for g in gene[tr]])]
    it = tr[np.array([g not in ivg for g in gene[tr]])]
    gi = groups(iv); best = (-1.0, 1.0)
    for lam in K.LAMBDAS:
        w = K.winrate(fe(it, iv, lam), gi)
        if w > best[0]: best = (w, lam)
    sc[te] = fe(tr, te, best[1]); lams.append(best[1])
assert np.isfinite(sc).all(), "LOCO left peaks unscored"
scored = len({g for g in GL})
log(f"[loco] {len(set(chrom))} folds, every one of the {scored} genes scored out of fold; "
    f"lambda per fold ranged {min(lams):g} to {max(lams):g}")

rk = np.empty(len(GL), int); nn = np.empty(len(GL), float)
for j, g in enumerate(GL):
    r, c = PAIR[g]
    rk[j] = 1 + int((sc[c] >= sc[r]).sum()); nn[j] = len(c) + 1
log(f"[rank] candidates per gene: median {np.median(nn):.0f}, mean {nn.mean():.2f}, "
    f"max {int(nn.max())}")

mod = [float((rk <= k).mean()) for k in range(1, KMAX + 1)]
chc = [float(np.mean(np.minimum(k, nn) / nn)) for k in range(1, KMAX + 1)]
def ci(k, nb=2000):
    v = (rk <= k).astype(float)
    bs = np.random.default_rng(K.SEED).integers(0, len(v), size=(nb, len(v)))
    m = v[bs].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
log("[recall] " + "  ".join(f"k{k}: {mod[k-1]:.3f} vs {chc[k-1]:.3f}" for k in (1, 2, 3, 5)))
with open("recall_at_k.tsv", "w") as fh:
    fh.write("scope\tk\trecall\tci_lo\tci_hi\tchance\tlift\tn_genes\n")
    for k in range(1, KMAX + 1):
        lo, hi = ci(k)
        fh.write(f"analysis set\t{k}\t{mod[k-1]:.6f}\t{lo:.6f}\t{hi:.6f}\t{chc[k-1]:.6f}\t"
                 f"{mod[k-1]-chc[k-1]:+.6f}\t{len(GL)}\n")
log("[out] recall_at_k.tsv")
