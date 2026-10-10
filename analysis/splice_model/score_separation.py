#!/usr/bin/env python3
# the 26-feature model's scores at true 3' ends against their own competitors, within gene
import sys, math
import numpy as np
import binned_common as K
from binned_common import log

Z = np.load("binned_profiles.npz", allow_pickle=True)
P = Z["P"]; CH = [str(c) for c in Z["channels"]]
X = np.column_stack([P[:, CH.index("u1_dens4"), :].astype(float), K.ACC])
assert X.shape[1] == 26
gene = K.gene; y = K.y; chrom = K.C0["chrom"][K.ok]; N = len(y)
GI = {g: np.flatnonzero(gene == g) for g in sorted(set(gene))}
PAIR = {g: (ix[y[ix] == 1], ix[y[ix] == 0]) for g, ix in GI.items()}
PAIR = {g: v for g, v in PAIR.items() if len(v[0]) and len(v[1])}
GL = np.array(sorted(PAIR)); assert len(GL) == 4847
log(f"[scope] {len(GL)} analysis-set genes, {N} peaks")

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
sc = np.full(N, np.nan)
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
    sc[te] = fe(tr, te, best[1])
assert np.isfinite(sc).all()
log("[loco] leave-one-chromosome-out complete, lambda retuned per fold")

cen = sc.copy()
for g in GL:
    ix = GI[g]; cen[ix] = sc[ix] - sc[ix].mean()

d = np.empty(len(GL)); wr = np.empty(len(GL))
for j, g in enumerate(GL):
    r, c = PAIR[g]
    d[j] = sc[r][0] - sc[c].mean()
    wr[j] = float((sc[r][0] > sc[c]).mean() + 0.5 * (sc[r][0] == sc[c]).mean())
WR = float(wr.mean())
bs = np.random.default_rng(K.SEED).integers(0, len(wr), size=(2000, len(wr)))
WLO, WHI = np.percentile(wr[bs].mean(1), [2.5, 97.5])
log(f"[winrate] LOCO within-gene win-rate over {len(GL)} genes: {WR:.4f} [{WLO:.4f},{WHI:.4f}]")
log(f"[winrate] the locked model's held-out figure is 0.6993 on 1,623 genes, a different "
    f"estimate of the same quantity")

def wilcoxon(v):
    nz = v[v != 0]; n = len(nz); a = np.abs(nz)
    o = np.argsort(a, kind="stable"); sa = a[o]
    _, inv, cnt = np.unique(sa, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); mid = ((cs - cnt) + cs + 1) / 2.0
    rank = np.empty(n); rank[o] = mid[inv]
    V = float(rank[nz > 0].sum()); mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(cnt ** 3 - cnt)) / 48.0
    z = (V - mu + (0.5 if V < mu else -0.5)) / math.sqrt(var)
    pv = math.erfc(abs(z) / math.sqrt(2))
    logp = (math.log(2) - z * z / 2 - math.log(abs(z) * math.sqrt(2 * math.pi))) / math.log(10)
    return V, n, z, ("< 2.2e-16" if pv < 2.2e-16 else f"{pv:.3g}"), logp, len(v) - n
V, n, z, pstr, logp, nz0 = wilcoxon(d)
log(f"[wilcoxon] PAIRED within-gene signed-rank on the per-gene delta (true minus its own "
    f"competitors' mean)")
log(f"[wilcoxon] V = {V:.0f}, n = {n} genes ({nz0} exact-zero deltas dropped), z = {z:+.4f}, "
    f"p {pstr} (log10 p ~ {logp:.1f})")
log(f"[wilcoxon] median paired delta {np.median(d):+.4f}, mean {d.mean():+.4f}, "
    f"{100*(d>0).mean():.1f}% of genes positive")
log("[wilcoxon] no pooled two-sample test was computed; the pooled contrast is the confounded "
    "one this figure exists to avoid")

with open("score_separation.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    for k_, v_ in (("n_genes", len(GL)), ("winrate_loco", f"{WR:.5f}"),
                   ("winrate_lo", f"{WLO:.5f}"), ("winrate_hi", f"{WHI:.5f}"),
                   ("winrate_heldout_locked", "0.69930"),
                   ("wilcoxon_V", f"{V:.0f}"), ("wilcoxon_n", n), ("zero_deltas", nz0),
                   ("wilcoxon_z", f"{z:.4f}"), ("wilcoxon_p", pstr),
                   ("wilcoxon_log10p", f"{logp:.1f}"),
                   ("median_delta", f"{np.median(d):.5f}"), ("mean_delta", f"{d.mean():.5f}"),
                   ("frac_positive", f"{(d>0).mean():.5f}"), ("test", "paired within-gene"),
                   ("oof", "leave-one-chromosome-out")):
        fh.write(f"{k_}\t{v_}\n")
with open("score_separation_sites.tsv", "w") as fh:
    fh.write("gene\tclass\tcentred_score\n")
    for i in range(N):
        fh.write(f"{gene[i]}\t{'true' if y[i]==1 else 'competitor'}\t{cen[i]:.5f}\n")
log(f"[dist] gene-centred score: true median {np.median(cen[y==1]):+.4f}, "
    f"competitor median {np.median(cen[y==0]):+.4f}")
log("[out] score_separation.tsv score_separation_sites.tsv")
