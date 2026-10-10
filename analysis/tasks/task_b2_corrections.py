#!/usr/bin/env python3
# task b2 corrections: recompute the tie decomposition gene-averaged
import sys, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)

Z = np.load("u1_window_cache.npz")
isc = Z["isc"]; gi = Z["gene_idx"]; LADDER = list(Z["ladder"])
feats = {f"U1 density @ {t:g}": Z[f"dens_{t}"] for t in LADDER}
feats["u1_excess @ 5.25"] = Z["excess_525"]
NP = len(isc); log(f"[cache] {NP:,} peaks ; ladder {LADDER}")

order = np.argsort(gi, kind="stable"); gs = gi[order]
cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
groups = []
for j in range(len(cut) - 1):
    idx = order[cut[j]:cut[j + 1]]
    c = idx[isc[idx] == 1]; k = idx[isc[idx] == 0]
    if len(c) == 1 and len(k) >= 1: groups.append((int(c[0]), k))
log(f"[within] {len(groups):,} qualifying genes")

def gene_stats(val, nboot=2000):
    pg_half, pg_excl = [], []
    tot_pairs = tie_pairs = bz_pairs = info_pairs = win_info = 0
    for c_idx, k_idxs in groups:
        vc = val[c_idx]; vk = val[k_idxs]
        fin = np.isfinite(vk) & np.isfinite(vc)
        vk = vk[fin]
        if not np.isfinite(vc) or vk.size == 0: continue
        wins = vk > vc; ties = vk == vc; info = ~ties
        pg_half.append(float((wins * 1.0 + ties * 0.5).mean()))
        if info.sum() > 0: pg_excl.append(float((vk[info] > vc).mean()))
        tot_pairs += vk.size; tie_pairs += int(ties.sum()); bz_pairs += int(((vk == 0) & (vc == 0)).sum())
        info_pairs += int(info.sum()); win_info += int((vk[info] > vc).sum())
    pg_half = np.array(pg_half); pg_excl = np.array(pg_excl)
    def ci(a):
        b = a[rng.integers(0, len(a), size=(nboot, len(a)))].mean(1)
        return float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    h_m, h_lo, h_hi = ci(pg_half)
    e_m, e_lo, e_hi = ci(pg_excl)
    return dict(genes_all=len(pg_half), genes_info=len(pg_excl),
                ga_half=h_m, ga_half_lo=h_lo, ga_half_hi=h_hi,
                ga_excl=e_m, ga_excl_lo=e_lo, ga_excl_hi=e_hi,
                frac_tie=tie_pairs / tot_pairs, frac_bz=bz_pairs / tot_pairs,
                pp_excl=win_info / max(info_pairs, 1), n_info_pairs=info_pairs)

R = {name: gene_stats(v) for name, v in feats.items()}

print("\n============  TASK B2 (corrected): GENE-AVERAGED win-rate, ties excluded vs ties=0.5  ============")
print("estimator = per-gene win-rate averaged over genes (Section 3); 95% CI = bootstrap over genes\n")
h = f"{'feature':<22} {'frac_tie':>8} {'both0':>7} {'GA_excl_ties':>13} {'95% CI':>18} {'genes':>6} {'GA_ties=0.5':>12} {'95% CI':>18}"
print(h); print("-" * len(h))
for name in feats:
    r = R[name]
    ci_e = f"[{r['ga_excl_lo']:.3f},{r['ga_excl_hi']:.3f}]"
    ci_h = f"[{r['ga_half_lo']:.3f},{r['ga_half_hi']:.3f}]"
    print(f"{name:<22} {r['frac_tie']:>8.4f} {r['frac_bz']:>7.4f} {r['ga_excl']:>13.4f} {ci_e:>18} "
          f"{r['genes_info']:>6} {r['ga_half']:>12.4f} {ci_h:>18}")

print("\n-- pair-pooled companion (64,140 pairs; NOT the headline estimator) --")
for name in feats:
    r = R[name]
    print(f"  {name:<22} tie-excluded(pooled) {r['pp_excl']:.4f}  informative pairs {r['n_info_pairs']:,}")

d65 = R["U1 density @ 6.5"]; d52 = R["U1 density @ 5.25"]; d70 = R["U1 density @ 7"]
dip_below_525 = d65["ga_excl_hi"] < d52["ga_excl_lo"]
dip_below_70  = d65["ga_excl_hi"] < d70["ga_excl_lo"]
print(f"\n[dip@6.5] tie-excl GA = {d65['ga_excl']:.4f} CI [{d65['ga_excl_lo']:.4f},{d65['ga_excl_hi']:.4f}]")
print(f"          vs 5.25 CI [{d52['ga_excl_lo']:.4f},{d52['ga_excl_hi']:.4f}]  vs 7.0 CI [{d70['ga_excl_lo']:.4f},{d70['ga_excl_hi']:.4f}]")
print(f"          CI-separated from 5.25? {dip_below_525}   from 7.0? {dip_below_70}")
print(f"\n[conditioning] tie-excluded n genes: 4.0={R['U1 density @ 4']['genes_info']}, "
      f"8.0={R['U1 density @ 8']['genes_info']} of {len(groups)} "
      f"(high-threshold subset is feature-selected: genes with >=1 informative, strong-site pair)")

with open("u1_tie_analysis_geneavg.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["feature", "frac_tie", "frac_bothzero", "ga_winrate_ties_excluded",
                "ga_excl_lo", "ga_excl_hi", "genes_informative",
                "ga_winrate_ties_half", "ga_half_lo", "ga_half_hi", "pairpooled_excl", "n_info_pairs"])
    for name in feats:
        r = R[name]
        w.writerow([name, f"{r['frac_tie']:.6f}", f"{r['frac_bz']:.6f}", f"{r['ga_excl']:.6f}",
                    f"{r['ga_excl_lo']:.6f}", f"{r['ga_excl_hi']:.6f}", r["genes_info"],
                    f"{r['ga_half']:.6f}", f"{r['ga_half_lo']:.6f}", f"{r['ga_half_hi']:.6f}",
                    f"{r['pp_excl']:.6f}", r["n_info_pairs"]])
print("\n[out] u1_tie_analysis_geneavg.tsv")
