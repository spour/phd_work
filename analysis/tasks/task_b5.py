#!/usr/bin/env python3
# b5: excess-over-null becomes the reported statistic for every within-gene win-rate
import sys
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)
NPERM = 1000

Z = np.load("u1_window_cache.npz")
isc = Z["isc"]; gi = Z["gene_idx"]; LADDER = [4.0, 5.25, 6.5, 7.0, 8.0]
DENS = {t: Z[f"dens_{t}"] for t in LADDER}
u1d525 = Z["dens_5.25"]; excess525 = Z["excess_525"]
zc = np.load("fig3_ucube.npz"); mc = zc["me5cube"]; edges = zc["edges"]; mthr = list(zc["me5_thrs"])
lo, hi = int(np.searchsorted(edges, -500)), int(np.searchsorted(edges, 0))
me5c8 = mc[:, lo:hi, 3 + mthr.index(8.0)].sum(1)

# gene structure
order = np.argsort(gi, kind="stable"); gs = gi[order]
cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
gmemb, gcanon = [], []
for j in range(len(cut) - 1):
    idx = order[cut[j]:cut[j + 1]]
    c = idx[isc[idx] == 1]; k = idx[isc[idx] == 0]
    if len(c) == 1 and len(k) >= 1: gmemb.append(idx); gcanon.append(int(c[0]))
NG = len(gmemb)
comp_flat = np.concatenate([m[m != c] for m, c in zip(gmemb, gcanon)])
comp_gene = np.concatenate([np.full((m != c).sum(), i) for i, (m, c) in enumerate(zip(gmemb, gcanon))])
canon_arr = np.array(gcanon)
memb_flat = np.concatenate(gmemb)
sizes = np.array([len(m) for m in gmemb]); starts = np.r_[0, np.cumsum(sizes)[:-1]]
memb_gene = np.repeat(np.arange(NG), sizes)
log(f"[within] {NG:,} genes ; {len(comp_flat):,} competitor pairs")

def obs_pergene(val):
    vc = val[canon_arr][comp_gene]; vk = val[comp_flat]
    win = vk > vc; loss = vk < vc; tie = vk == vc
    wins = np.bincount(comp_gene, win, NG); loss_g = np.bincount(comp_gene, loss, NG)
    tie_g = np.bincount(comp_gene, tie, NG); n = np.bincount(comp_gene, None, NG)
    info = wins + loss_g
    wr = np.where(info > 0, wins / np.maximum(info, 1), np.nan)
    half = (wins + 0.5 * tie_g) / n
    return wr, half

def null_pergene(val):
    nu = np.full(NG, np.nan)
    for i, m in enumerate(gmemb):
        v = val[m]; sv = np.sort(v); mm = len(v)
        less = np.searchsorted(sv, v, "left"); greater = mm - np.searchsorted(sv, v, "right")
        info = (less + greater) > 0
        if info.any():
            nu[i] = (greater[info] / (less[info] + greater[info])).mean()
    return nu

def perm_null_mean(val, nperm):
    mv = val[memb_flat]; out = np.empty(nperm)
    for p in range(nperm):
        off = (rng.random(NG) * sizes).astype(int); ci = starts + off
        cv = mv[ci][memb_gene]
        nc = np.ones(len(mv), bool); nc[ci] = False
        win = (mv > cv) & nc; loss = (mv < cv) & nc
        w = np.bincount(memb_gene, win, NG); l = np.bincount(memb_gene, loss, NG)
        info = w + l; wr = np.where(info > 0, w / np.maximum(info, 1), np.nan)
        out[p] = np.nanmean(wr)
    return out

def report(val, name, nperm_confirm=0):
    wr, half = obs_pergene(val); nu = null_pergene(val)
    ok = np.isfinite(wr)
    obs = wr[ok].mean(); null = nu[ok].mean(); exc = obs - null
    exg = (wr - nu)[ok]
    b = exg[rng.integers(0, len(exg), size=(2000, len(exg)))].mean(1)
    lo_, hi_ = np.percentile(b, [2.5, 97.5])
    half_obs = np.nanmean(half)
    conf = perm_null_mean(val, nperm_confirm).mean() if nperm_confirm else np.nan
    return dict(name=name, obs=obs, null=null, exc=exc, lo=lo_, hi=hi_,
                half=half_obs, genes=int(ok.sum()), conf=conf)

print("\n============  B5.1  within-gene win-rate as excess over the permutation null  ============")
print("gene-averaged tie-excluded sign test; null = exact per-gene permutation expectation\n")
h = f"{'feature':<24} {'observed':>9} {'null':>7} {'excess':>8} {'95% CI (excess)':>18} {'genes':>6} {'ties=0.5':>9}"
print(h); print("-" * len(h))
rows = []
for t in LADDER:
    r = report(DENS[t], f"U1 density @ {t:g}", nperm_confirm=(NPERM if t in (5.25, 7.0, 8.0) else 0))
    rows.append(r)
rows.append(report(excess525, "u1_excess @ 5.25"))
rows.append(report(me5c8, "MaxEnt5 count >= 8"))
for r in rows:
    ci = f"[{r['lo']:+.3f},{r['hi']:+.3f}]"
    print(f"{r['name']:<24} {r['obs']:>9.4f} {r['null']:>7.4f} {r['exc']:>+8.4f} {ci:>18} {r['genes']:>6} {r['half']:>9.4f}")
print("\nclosed-form null vs 1,000-perm empirical (confirmation):")
for r in rows:
    if np.isfinite(r["conf"]):
        print(f"  {r['name']:<24} closed-form {r['null']:.4f}  empirical(1000) {r['conf']:.4f}")

print("\n============  B5.2  null of the other two estimators  ============")
def perm_half_mean(val, nperm):
    mv = val[memb_flat]; out = np.empty(nperm)
    for p in range(nperm):
        off = (rng.random(NG) * sizes).astype(int); ci = starts + off
        cv = mv[ci][memb_gene]; nc = np.ones(len(mv), bool); nc[ci] = False
        win = (mv > cv) & nc; tie = (mv == cv) & nc
        w = np.bincount(memb_gene, win, NG); ti = np.bincount(memb_gene, tie, NG)
        n = np.bincount(memb_gene, nc.astype(float), NG)
        out[p] = np.nanmean(np.where(n > 0, (w + 0.5 * ti) / np.maximum(n, 1), np.nan))
    return out
ph = perm_half_mean(u1d525, 1000)
print(f"  ties=0.5 (U1@5.25) permutation null: {ph.mean():.4f}  [{np.percentile(ph,2.5):.4f},{np.percentile(ph,97.5):.4f}]  (expect 0.500, unbiased)")
ne = perm_null_mean(excess525, 1000)
print(f"  u1_excess tie-excluded permutation null: {ne.mean():.4f}  (continuous, expect ~0.500; obs 0.547 stands if so)")

print("\n============  B5.3  reciprocal conditioning (null-corrected increments)  ============")
def residualize(y, x):
    ok = np.isfinite(y) & np.isfinite(x); bta = np.polyfit(x[ok], y[ok], 1)
    return y - (bta[0] * x + bta[1])
ru1 = residualize(u1d525, me5c8); rme5 = residualize(me5c8, u1d525)
for val, nm in [(u1d525, "U1@5.25 (marginal)"), (ru1, "U1@5.25 | MaxEnt5"),
                (me5c8, "MaxEnt5>=8 (marginal)"), (rme5, "MaxEnt5>=8 | U1@5.25")]:
    r = report(val, nm)
    print(f"  {nm:<24} obs {r['obs']:.4f}  null {r['null']:.4f}  excess {r['exc']:+.4f}  [{r['lo']:+.3f},{r['hi']:+.3f}]")

import csv
with open("u1_within_null.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["feature", "observed", "null", "excess", "excess_lo", "excess_hi", "ties_half", "genes"])
    for r in rows:
        w.writerow([r["name"], f"{r['obs']:.6f}", f"{r['null']:.6f}", f"{r['exc']:.6f}",
                    f"{r['lo']:.6f}", f"{r['hi']:.6f}", f"{r['half']:.6f}", r["genes"]])
print("\n[out] u1_within_null.tsv")
