#!/usr/bin/env python3
# b3: (1) label-permutation null confirming the informative-pair selection
import sys, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)

Z = np.load("u1_window_cache.npz")
isc = Z["isc"]; gi = Z["gene_idx"]
DENS = {5.25: Z["dens_5.25"], 7.0: Z["dens_7.0"], 8.0: Z["dens_8.0"]}

order = np.argsort(gi, kind="stable"); gs = gi[order]
cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
genes = []
gmemb = []; gcanon = []
for j in range(len(cut) - 1):
    idx = order[cut[j]:cut[j + 1]]
    c = idx[isc[idx] == 1]; k = idx[isc[idx] == 0]
    if len(c) == 1 and len(k) >= 1:
        gmemb.append(idx); gcanon.append(int(c[0]))
log(f"[within] {len(gmemb):,} genes")

# part 1: label-permutation null (pooled tie-excluded win-rate)
def pooled_excl_winrate(val, canon_of):
    win = info = 0
    for memb, c in zip(gmemb, canon_of):
        v = val[memb]; vc = val[c]
        vk = v[memb != c]
        d = vk - vc; nz = d != 0
        win += int((d[nz] > 0).sum()); info += int(nz.sum())
    return win / max(info, 1)

print("\n=== B3.1  label-permutation null (informative-pair selection is label-symmetric) ===")
print("under the null the tie-excluded win-rate is 0.5 by construction: the filter tests only")
print("value inequality, never the canonical label. Empirical confirmation (200 within-gene")
print("label permutations); observed value shown for contrast.\n")
for t in (5.25, 7.0, 8.0):
    val = DENS[t]
    obs = pooled_excl_winrate(val, gcanon)
    null = []
    for _ in range(200):
        canon_perm = [int(memb[rng.integers(len(memb))]) for memb in gmemb]
        null.append(pooled_excl_winrate(val, canon_perm))
    null = np.array(null)
    print(f"  @{t:<5} observed {obs:.4f}   null mean {null.mean():.4f}  95% [{np.percentile(null,2.5):.4f},{np.percentile(null,97.5):.4f}]")

# part 2: one-zero vs both-nonzero-unequal
print("\n=== B3.2  decomposition of informative pairs ===")
print("one-zero = exactly one side carries a strong site; win-rate here = P(the strong-site")
print("side is the competitor). both>0 = both sides carry sites, unequal density.\n")
rows2 = []
for t in (5.25, 7.0, 8.0):
    val = DENS[t]
    n_info = n_onezero = n_both = win_onezero = win_both = 0
    for memb, c in zip(gmemb, gcanon):
        vc = val[c]; vk = val[memb[memb != c]]
        d = vk - vc; info = d != 0
        vk_i = vk[info]; d_i = d[info]
        onezero = ((vk_i == 0) | (vc == 0)) & ~((vk_i == 0) & (vc == 0))
        both = (vk_i > 0) & (vc > 0)
        n_info += info.sum(); n_onezero += int(onezero.sum()); n_both += int(both.sum())
        win_onezero += int((d_i[onezero] > 0).sum()); win_both += int((d_i[both] > 0).sum())
    fr_oz = n_onezero / n_info; fr_both = n_both / n_info
    wr_oz = win_onezero / max(n_onezero, 1); wr_both = win_both / max(n_both, 1)
    rows2.append((t, n_info, fr_oz, wr_oz, fr_both, wr_both))
    print(f"  @{t:<5} informative {n_info:,}")
    print(f"        one-zero  frac {fr_oz:.3f}  win-rate {wr_oz:.4f}  (= P competitor is the strong-site side)")
    print(f"        both>0    frac {fr_both:.3f}  win-rate {wr_both:.4f}")

with open("u1_pair_decomp.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["threshold","n_informative","frac_one_zero","winrate_one_zero","frac_both_nonzero","winrate_both_nonzero"])
    for r in rows2: w.writerow([f"{r[0]:.2f}", r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}"])
print("[out] u1_pair_decomp.tsv")
