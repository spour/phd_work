#!/usr/bin/env python3
# b4.1 + b4.3 (cache-only, no rescan)
import sys
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)

Z = np.load("u1_window_cache.npz")
isc = Z["isc"]; gi = Z["gene_idx"]; LADDER = [4.0, 5.25, 6.5, 7.0, 8.0]
DENS = {t: Z[f"dens_{t}"] for t in LADDER}

order = np.argsort(gi, kind="stable"); gs = gi[order]
cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
gmemb, gcanon = [], []
for j in range(len(cut) - 1):
    idx = order[cut[j]:cut[j + 1]]
    c = idx[isc[idx] == 1]; k = idx[isc[idx] == 0]
    if len(c) == 1 and len(k) >= 1: gmemb.append(idx); gcanon.append(int(c[0]))
NG = len(gmemb); log(f"[within] {NG:,} genes")

def per_gene_excl(val, canon_of):
    out = np.full(NG, np.nan)
    for i, (memb, c) in enumerate(zip(gmemb, canon_of)):
        vc = val[c]; vk = val[memb[memb != c]]
        d = vk - vc; nz = d != 0
        if nz.any(): out[i] = (d[nz] > 0).mean()
    return out

def ga(val, canon_of):
    pg = per_gene_excl(val, canon_of); pg = pg[np.isfinite(pg)]
    return pg.mean(), len(pg)

def info_count(val, canon_of):
    out = np.zeros(NG, int)
    for i, (memb, c) in enumerate(zip(gmemb, canon_of)):
        vc = val[c]; vk = val[memb[memb != c]]; out[i] = int((vk != vc).sum())
    return out

# b4.1 gene-averaged permutation null
print("\n=== B4.1  gene-averaged label-permutation null ===")
print("observed and null are BOTH gene-averaged tie-excluded (headline estimator)\n")
print(f"{'thr':>5} {'observed_GA':>12} {'null_mean':>10} {'null 95% CI':>20}")
for t in (5.25, 7.0, 8.0):
    val = DENS[t]
    obs, _ = ga(val, gcanon)
    null = np.array([ga(val, [int(m[rng.integers(len(m))]) for m in gmemb])[0] for _ in range(200)])
    print(f"{t:>5} {obs:>12.4f} {null.mean():>10.4f}   [{np.percentile(null,2.5):.4f},{np.percentile(null,97.5):.4f}]")

# b4.3 reweighting check
print("\n=== B4.3  is the sharpening a reweighting artifact? ===")
ic = {t: info_count(DENS[t], gcanon) for t in LADDER}
print("\ninformative pairs per gene (quantiles):")
for t in (5.25, 7.0, 8.0):
    q = np.percentile(ic[t], [50, 75, 90, 99]); n1 = int((ic[t] == 1).sum())
    print(f"  @{t:<5} median {q[0]:.0f}  q75 {q[1]:.0f}  q90 {q[2]:.0f}  max {ic[t].max()}  ; genes with exactly 1: {n1:,}")

print("\ngene-averaged tie-excluded win-rate, stratified by informative-pair count:")
bins = [("1", lambda x: x == 1), ("2-3", lambda x: (x >= 2) & (x <= 3)),
        ("4-10", lambda x: (x >= 4) & (x <= 10)), (">10", lambda x: x > 10)]
print(f"{'stratum':>8}" + "".join(f"{f'@{t:g}':>12}" for t in (5.25, 7.0, 8.0)))
for lab, f in bins:
    row = f"{lab:>8}"
    for t in (5.25, 7.0, 8.0):
        pg = per_gene_excl(DENS[t], gcanon)
        m = f(ic[t]) & np.isfinite(pg)
        row += f"{(pg[m].mean() if m.any() else np.nan):>12.4f}"
    print(row + f"   (n@8.0 {int((f(ic[8.0])).sum())})")

fixed = np.ones(NG, bool)
for t in LADDER: fixed &= (ic[t] >= 4)
log(f"[fixed] {int(fixed.sum()):,} genes with >=4 informative pairs at ALL thresholds")
print(f"\nfixed gene set ({int(fixed.sum()):,} genes, >=4 informative pairs at every threshold), like-for-like ladder:")
print(f"{'thr':>6} {'GA_tie_excl':>12} {'95% CI':>20}")
for t in LADDER:
    pg = per_gene_excl(DENS[t], gcanon)[fixed]
    pg = pg[np.isfinite(pg)]
    b = pg[rng.integers(0, len(pg), size=(2000, len(pg)))].mean(1)
    print(f"{t:>6} {pg.mean():>12.4f}   [{np.percentile(b,2.5):.4f},{np.percentile(b,97.5):.4f}]")
