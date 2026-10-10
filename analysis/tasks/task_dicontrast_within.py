#!/usr/bin/env python3
# test the mid-window [-500,0] strand-asymmetric dinucleotide contrasts directly
import sys, csv, math
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
POS = "fig3_new_positional_features.tsv"
R = list(csv.DictReader(open(POS), delimiter="\t")); NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
F1 = np.load("hypothesis_test_features.npy")
DI0 = 24
IDX = {d: i for i, d in enumerate([a+b for a in "ACGT" for b in "ACGT"])}
def di(d): return F1[:, DI0 + IDX[d]]
d_CA = di("CA") - di("AC"); d_TA = di("TA") - di("AT"); d_GA = di("GA") - di("AG")

gmap = defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)
groups = []
for g, idxs in gmap.items():
    idx = np.array(idxs); c = idx[isc[idx] == 1]; k = idx[isc[idx] == 0]
    if len(c) == 1 and len(k) >= 1: groups.append((int(c[0]), k, idx))
NGg = len(groups); log(f"[within] {NGg} genes (1 canonical + >=1 competitor)")

def auroc(pos, neg):
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]; n1, n0 = len(pos), len(neg)
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort"); sa = a[o]
    ng = np.r_[True, sa[1:] != sa[:-1]]; grp = np.cumsum(ng); ranks = np.empty(len(a))
    st = np.r_[np.nonzero(ng)[0], len(a)]; av = (st[:-1] + st[1:] + 1) / 2.0; ranks[o] = av[grp - 1]
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

rng = np.random.default_rng(19); NPERM = 2000

def within_test(val, name):
    wr_half, wr_excl, pdiff = [], [], []
    wg_list = []
    for c, ks, idx in groups:
        v = val[idx]; n = len(idx)
        gt = (v[:, None] > v[None, :]).sum(1)
        eq = (v[:, None] == v[None, :]).sum(1) - 1
        wg_list.append((gt + 0.5 * eq) / (n - 1))
        vc = val[c]; vk = val[ks]
        wr_half.append(np.mean((vc > vk) * 1.0 + (vc == vk) * 0.5))
        info = vk != vc
        if info.sum() > 0: wr_excl.append(np.mean(vc > vk[info]))
        pdiff.append(vc - vk.mean())
    wr_half = np.array(wr_half); wr_excl = np.array(wr_excl); pdiff = np.array(pdiff)
    obs = wr_half.mean()
    bs = wr_half[rng.integers(0, len(wr_half), size=(2000, len(wr_half)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    null = np.empty((NPERM, len(wg_list)))
    for j, gi in enumerate(wg_list): null[:, j] = rng.choice(gi, size=NPERM)
    null = null.mean(1)
    pval = (np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (NPERM + 1)
    pos = int(np.sum(wr_half > 0.5)); neg = int(np.sum(wr_half < 0.5)); tie = int(np.sum(wr_half == 0.5))
    ne = pos + neg
    def binom_two(k, n):
        if n == 0: return float("nan")
        mu = n / 2.0; sd = math.sqrt(n / 4.0)
        z = (abs(k - mu) - 0.5) / sd
        return math.erfc(z / math.sqrt(2))
    sgn_excl_p = binom_two(pos, ne)
    sgn_half_stat = (pos + 0.5 * tie) / len(wr_half)
    pooled = auroc(val[isc == 1], val[isc == 0])
    return dict(name=name, genes=len(wr_half), ga_half=obs, ga_excl=wr_excl.mean(),
                excess=obs - 0.5, ci_lo=lo, ci_hi=hi, null_mean=null.mean(),
                null_lo=np.percentile(null, 2.5), null_hi=np.percentile(null, 97.5), perm_p=pval,
                pooled_auroc=pooled, sign_pos=pos, sign_neg=neg, sign_tie=tie,
                sign_excl_p=sgn_excl_p, sign_half=sgn_half_stat, pdiff=pdiff.mean())

print("\n==== MID-WINDOW [-500,0] DINUCLEOTIDE CONTRASTS: within-gene canonical-vs-competitor ====")
print("win-rate/excess/AUROC oriented so >0.5 means the CANONICAL (real 3' end) carries MORE of the contrast.")
print(f"{'contrast':8s} {'genes':>6s} {'GA_ties=0.5':>11s} {'GA_excl':>8s} {'excess':>8s} {'boot 95% CI':>18s} "
      f"{'perm_null_mean':>14s} {'perm_p':>8s} {'pooled_AUROC':>12s} {'sign +/-/tie':>16s} {'sign_p':>8s}")
for v, nm in [(d_CA, "d_CA"), (d_TA, "d_TA"), (d_GA, "d_GA")]:
    r = within_test(v, nm)
    ci = f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]"; sg = f"{r['sign_pos']}/{r['sign_neg']}/{r['sign_tie']}"
    print(f"{r['name']:8s} {r['genes']:>6d} {r['ga_half']:>11.4f} {r['ga_excl']:>8.4f} {r['excess']:>+8.4f} {ci:>18s} "
          f"{r['null_mean']:>14.4f} {r['perm_p']:>8.4g} {r['pooled_auroc']:>12.4f} {sg:>16s} {r['sign_excl_p']:>8.2g}")
print("\nnull mean ~0.5 confirms the permutation null; excess = GA(ties=0.5) - 0.5.")
print("If excess ~0, CI spans 0.5, AUROC ~0.5, perm_p large -> the equal-and-opposite mid-window di")
print("coefficients were the near-null-direction artifact and are dropped from interpretation.")
