#!/usr/bin/env python3
# kinetic-competition stats: canonical (6,583) vs competitor (64,140)
import math, numpy as np
from collections import defaultdict
def log(*a):
    import sys; print(*a, file=sys.stderr, flush=True)

K = np.load("kinetic_scan_cache.npz", allow_pickle=True)
isc = K["is_canonical"].astype(int); gene = K["gene"]
_, gid = np.unique(gene, return_inverse=True)
canon = isc == 1; comp = isc == 0
RNG = np.random.default_rng(11)

METRICS = [("intron_len", "host intron length (nt)"),
           ("frac_pos", "position in intron (donor=0, acceptor=1)"),
           ("up_donor_dist", "distance to upstream donor (nt)"),
           ("dn_acc_dist", "distance to downstream acceptor (nt)"),
           ("up_donor_score", "upstream donor score (MaxEnt 5'ss)"),
           ("dn_acc_score", "downstream acceptor score (MaxEnt 3'ss)")]

def auroc_fast(a, b):
    b2 = np.sort(b); nB = len(b2)
    lo = np.searchsorted(b2, a, "left"); hi = np.searchsorted(b2, a, "right")
    return float(np.mean(lo + 0.5 * (hi - lo)) / nB)

def _memof(ids):
    o = np.argsort(ids, kind="mergesort"); ids_s = ids[o]
    cut = np.r_[0, np.flatnonzero(ids_s[1:] != ids_s[:-1]) + 1, len(ids_s)]
    return [o[cut[j]:cut[j + 1]] for j in range(len(cut) - 1)]

def auroc_between(xa, ida, xb, idb, B=400):
    ma = np.isfinite(xa); mb = np.isfinite(xb)
    xa, ida = xa[ma], ida[ma]; xb, idb = xb[mb], idb[mb]
    obs = auroc_fast(xa, xb)
    mema = _memof(ida); memb = _memof(idb); na = len(mema); nb = len(memb)
    dr = np.empty(B)
    for k in range(B):
        a = xa[np.concatenate([mema[p] for p in RNG.integers(0, na, na)])]
        b = xb[np.concatenate([memb[p] for p in RNG.integers(0, nb, nb)])]
        dr[k] = auroc_fast(a, b)
    return obs, float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))

def mannwhitney(xa, xb):
    xa = xa[np.isfinite(xa)]; xb = xb[np.isfinite(xb)]
    n1, n2 = len(xa), len(xb); N = n1 + n2
    allv = np.concatenate([xa, xb]); order = np.argsort(allv, kind="mergesort")
    sv = allv[order]; rk = np.empty(N); i = 0
    tie_term = 0.0
    while i < N:
        j = i
        while j + 1 < N and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0
        t = j - i + 1; tie_term += t ** 3 - t; i = j + 1
    rr = np.empty(N); rr[order] = rk
    R1 = rr[:n1].sum(); U1 = R1 - n1 * (n1 + 1) / 2.0
    mU = n1 * n2 / 2.0
    sU = math.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie_term / (N * (N - 1))))
    z = (U1 - mU) / sU if sU > 0 else 0.0
    if abs(z) > 37: pstr = "p<1e-300"
    else:
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        pstr = f"p={p:.2e}" if p >= 1e-300 else "p<1e-300"
    return z, pstr

_by = defaultdict(lambda: {"c": [], "k": []})
for _i in range(len(isc)):
    _by[gid[_i]]["c" if canon[_i] else "k"].append(_i)
GENES = [(np.array(d["c"]), np.array(d["k"])) for d in _by.values()
         if len(d["c"]) == 1 and len(d["k"]) >= 1]

def within_gene_perm(val, nperm=500):
    obs = []; wr_mem = []; sizes = []; can_at = []
    for ci, ki in GENES:
        cv = val[ci[0]]; kv = val[ki]; m = np.isfinite(kv)
        if not np.isfinite(cv) or m.sum() < 1: continue
        kv = kv[m]; pool = np.r_[cv, kv]; M = len(pool); mm = M - 1
        sp = np.sort(pool)
        less = np.searchsorted(sp, pool, "left")
        eq = np.searchsorted(sp, pool, "right") - less
        wm = (less + 0.5 * (eq - 1)) / mm
        wr_mem.append(wm); sizes.append(M); can_at.append(0); obs.append(wm[0])
    obs = np.array(obs); wr = float(obs.mean()); ng = len(obs)
    bs = obs[RNG.integers(0, ng, size=(1000, ng))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    rs = np.random.default_rng(7)
    nul = np.empty(nperm)
    for t in range(nperm):
        nul[t] = np.mean([wm[rs.integers(0, s)] for wm, s in zip(wr_mem, sizes)])
    p = (np.sum(np.abs(nul - 0.5) >= abs(wr - 0.5)) + 1) / (nperm + 1)
    return wr, float(lo), float(hi), p, ng

rows = []
fv = open("fig3env_kinetic_vals.tsv", "w"); fv.write("metric\tgroup\tvalue\n")
print("\n================  ITEM 1: kinetic competition, canonical vs competitor  ================")
print(f"{'quantity':<34} {'canon':>9} {'comp':>9}  {'AUROC[95% CI]':>22}  {'win-rate[CI] (perm p)':>26}  MannWhitney")
for key, lab in METRICS:
    v = K[key]; va = v[canon]; vb = v[comp]
    ca_m = np.nanmedian(va); cb_m = np.nanmedian(vb); ca_mean = np.nanmean(va); cb_mean = np.nanmean(vb)
    na = int(np.isfinite(va).sum()); nb = int(np.isfinite(vb).sum())
    au, alo, ahi = auroc_between(v[canon], gid[canon], v[comp], gid[comp])
    z, pstr = mannwhitney(va, vb)
    wr, wlo, whi, pp, ng = within_gene_perm(v)
    rows.append((key, lab, ca_mean, cb_mean, ca_m, cb_m, na, nb, au, alo, ahi, wr, wlo, whi, pp, ng, z, pstr))
    print(f"{lab:<34} {ca_m:>9.2f} {cb_m:>9.2f}  {au:>6.3f}[{alo:.3f},{ahi:.3f}]  "
          f"{wr:>6.3f}[{wlo:.3f},{whi:.3f}] p={pp:.4f}  {pstr} (z={z:.1f})")
    for gtag, arr in [("canonical", va), ("competitor", vb)]:
        a = arr[np.isfinite(arr)]
        for x in a: fv.write(f"{key}\t{gtag}\t{x:.4f}\n")
fv.close()

with open("fig3env_kinetic_stats.tsv", "w") as fh:
    fh.write("metric\tlabel\tcanon_mean\tcomp_mean\tcanon_med\tcomp_med\tn_canon\tn_comp\t"
             "auroc\tau_lo\tau_hi\twinrate\twr_lo\twr_hi\tperm_p\tng\tmw_z\tmw_p\n")
    for r in rows:
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.5f}" for x in r) + "\n")

print("\nseparation ranking (|AUROC-0.5|, distance/length/frac/acceptor-score; donor score -> panel d):")
rank = sorted([r for r in rows if r[0] != "up_donor_score"], key=lambda r: -abs(r[8] - 0.5))
for r in rank: print(f"  {r[1]:<34} AUROC {r[8]:.3f}  |sep| {abs(r[8]-0.5):.3f}")
log("wrote fig3env_kinetic_stats.tsv / vals; DONE_KINETIC_STATS")
