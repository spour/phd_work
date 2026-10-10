#!/usr/bin/env python3
# item 2: within-gene test on d_ga = freq(ga)-freq(ag) [mid window -500,0]
import sys, csv
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
POS = "fig3_new_positional_features.tsv"
R = list(csv.DictReader(open(POS), delimiter="\t")); NP = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
F1 = np.load("hypothesis_test_features.npy"); comp = np.load("peak_compartment.npy")
dGA = F1[:, 24 + 8] - F1[:, 24 + 2]

gmap = defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)
GENES = []
for g, idxs in gmap.items():
    idx = np.array(idxs); c = idx[isc[idx] == 1]; k = idx[isc[idx] == 0]
    if len(c) == 1 and len(k) >= 1: GENES.append((int(c[0]), idx))
NG = len(GENES)
rng = np.random.default_rng(19); NPERM = 2000

def auroc(pos, neg):
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort"); sa = a[o]
    ng = np.r_[True, sa[1:] != sa[:-1]]; grp = np.cumsum(ng); ranks = np.empty(len(a))
    st = np.r_[np.nonzero(ng)[0], len(a)]; av = (st[:-1] + st[1:] + 1) / 2.0; ranks[o] = av[grp - 1]
    n1 = len(pos); return (ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(neg))

def perpeak_winrate(v, cmp_, mode):
    n = len(v); wg = np.full(n, np.nan)
    for i in range(n):
        if mode == "all": sel = np.ones(n, bool)
        elif mode == "same": sel = cmp_ == cmp_[i]
        else: sel = (cmp_ == 0) & (cmp_[i] == 0)
        sel[i] = False
        m = sel.sum()
        if m == 0: continue
        vo = v[sel]; wg[i] = np.mean((v[i] > vo) * 1.0 + (v[i] == vo) * 0.5)
    return wg

WG = {"all": [], "same": [], "term": []}; CANpos = []
for c, idx in GENES:
    v = dGA[idx]; cm = comp[idx]; cpos = int(np.where(idx == c)[0][0]); CANpos.append(cpos)
    for mode in ("all", "same", "term"): WG[mode].append(perpeak_winrate(v, cm, mode))
log("per-peak win-rates precomputed")

def stat(mode):
    obs_vals = [];
    for g in range(NG):
        w = WG[mode][g][CANpos[g]]
        if np.isfinite(w): obs_vals.append(w)
    obs_vals = np.array(obs_vals); obs = obs_vals.mean()
    bs = obs_vals[rng.integers(0, len(obs_vals), size=(4000, len(obs_vals)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    null = np.empty(NPERM)
    for p in range(NPERM):
        vals = []
        for g in range(NG):
            w = WG[mode][g]; fin = np.where(np.isfinite(w))[0]
            if len(fin) == 0: continue
            vals.append(w[fin[rng.integers(len(fin))]])
        null[p] = np.mean(vals)
    pval = (np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (NPERM + 1)
    return obs, lo, hi, null.mean(), pval, len(obs_vals)

def pooled_auroc(mode):
    cpos_dGA = []; comp_dGA = []
    for c, idx in GENES:
        v = dGA[idx]; cm = comp[idx]; cpos = int(np.where(idx == c)[0][0])
        if mode == "same": sel = (cm == cm[cpos])
        elif mode == "term": sel = (cm == 0) & (cm[cpos] == 0)
        else: sel = np.ones(len(idx), bool)
        sel[cpos] = False
        if sel.sum() == 0 or (mode == "term" and cm[cpos] != 0): continue
        cpos_dGA.append(v[cpos]); comp_dGA.extend(v[sel].tolist())
    if not cpos_dGA or not comp_dGA: return np.nan
    return auroc(np.array(cpos_dGA), np.array(comp_dGA))

print("\n==== ITEM 2: d_GA within-gene, compartment-restricted (0=TERM, 2=INTRON) ====")
print("orientation: >0.5 / +excess = canonical carries MORE d_GA than competitor")
print(f"{'stratum':20s} {'win-rate':>9s} {'excess':>8s} {'boot 95% CI':>18s} {'perm null':>10s} {'perm_p':>9s} {'pooled AUROC':>12s} {'n_genes':>8s}")
for mode, lab in [("all", "marginal (all pairs)"), ("same", "same-compartment"), ("term", "TERM-TERM (both TERM)")]:
    obs, lo, hi, nm, pv, ng = stat(mode); au = pooled_auroc(mode)
    print(f"{lab:20s} {obs:>9.4f} {obs-0.5:>+8.4f} [{lo:.4f},{hi:.4f}] {nm:>10.4f} {pv:>9.4g} {au:>12.4f} {ng:>8d}")
