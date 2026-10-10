#!/usr/bin/env python3
# the distributions behind the two features: pooled and paired, for both
import csv, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
NBIN, NPERM = 60, 10000
rng = np.random.default_rng(42)

T = pd.read_parquet("cpa_peaks_final.parquet", engine="fastparquet")
T = T[T.in_analysis_set].reset_index(drop=True)
y = T.is_canonical.to_numpy(); gene = T.gene.to_numpy()
print(f"analysis set {len(T):,} peaks, {int(y.sum()):,} canonical, {int((~y).sum()):,} competitor")
gi = defaultdict(list)
for i in range(len(T)): gi[gene[i]].append(i)
GIX = [np.array(v) for v in gi.values()]

def midrank(s):
    o = np.argsort(s, kind="mergesort"); sv = s[o]; n = len(s); rk = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]: j += 1
        rk[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    rr = np.empty(n); rr[o] = rk
    return rr

def paired(v, lab):
    out = []
    for ix in GIX:
        c = ix[lab[ix]]; k = ix[~lab[ix]]
        if len(c) != 1: continue
        kk = v[k][np.isfinite(v[k])]
        if not np.isfinite(v[c[0]]) or len(kk) == 0: continue
        out.append(v[c[0]] - kk.mean())
    return np.array(out)

def run(name, tag, v):
    ok = np.isfinite(v)
    rr = midrank(v[ok]); yy = y[ok].astype(float); P = yy.sum(); n = len(yy)
    auroc = float((rr[yy == 1].sum() - P * (P + 1) / 2) / (P * (n - P)))
    mcan = float(np.median(v[ok & y])); mcom = float(np.median(v[ok & ~y]))
    print(f"\n===== {name} =====")
    print(f"pooled: AUROC {auroc:.4f}   median canonical {mcan:+.4f}   "
          f"median competitor {mcom:+.4f}   finite {int(ok.sum()):,}")
    lo, hi = np.percentile(v[ok], [0.5, 99.5])
    edges = np.linspace(lo, hi, NBIN + 1); ctr = 0.5 * (edges[1:] + edges[:-1]); w = edges[1] - edges[0]
    with open(f"{tag}_hist.tsv", "w") as fh:
        fh.write("class\tcentre\tdensity\n")
        for cl, m in [("canonical", ok & y), ("competitor", ok & ~y)]:
            c, _ = np.histogram(v[m], bins=edges)
            for x, dd in zip(ctr, c / (c.sum() * w)): fh.write(f"{cl}\t{x:.6f}\t{dd:.6f}\n")
        fh.write(f"#auroc\t{auroc:.5f}\t\n#median_canonical\t{mcan:.5f}\t\n")
        fh.write(f"#median_competitor\t{mcom:.5f}\t\n#n_canonical\t{int((ok&y).sum())}\t\n")
        fh.write(f"#n_competitor\t{int((ok&~y).sum())}\t\n")
    d = paired(v, y)
    frac = float(np.mean(d > 0)); med = float(np.median(d))
    b = np.array([np.mean(d[rng.integers(0, len(d), len(d))] > 0) for _ in range(300)])
    print(f"paired: {len(d):,} genes, fraction above zero {frac:.4f} "
          f"[{np.percentile(b,2.5):.4f},{np.percentile(b,97.5):.4f}], median {med:+.4f}")
    flat = []; sizes = []; obs_pos = []
    for ix in GIX:
        f = ix[np.isfinite(v[ix])]
        if len(f) < 2 or (y[f]).sum() != 1: continue
        a = (v[f] > v[f].mean()).astype(np.int8)
        obs_pos.append(int(a[np.argmax(y[f])])); flat.append(a); sizes.append(len(f))
    flat = np.concatenate(flat); ng = np.array(sizes); off = np.r_[0, np.cumsum(ng)[:-1]]
    fobs = float(np.mean(obs_pos))
    assert abs(fobs - frac) < 1e-12, (fobs, frac)
    pick = off[None, :] + (rng.random((NPERM, len(ng))) * ng[None, :]).astype(int)
    null = flat[pick].mean(1)
    ge = int((null >= frac).sum())
    p = (1 + ge) / (1 + NPERM)
    ptxt = f"< {1/NPERM:.0e}" if ge == 0 else f"= {p:.2e}"
    print(f"        permutation null {null.mean():.4f} [{np.percentile(null,2.5):.4f},"
          f"{np.percentile(null,97.5):.4f}]; p {ptxt} "
          f"({ge} of {NPERM:,} shuffles at or above the observed fraction)")
    with open(f"{tag}_paired.tsv", "w") as fh:
        fh.write("rank\tdiff\n")
        for i, x in enumerate(np.sort(d)): fh.write(f"{i+1}\t{x:.6f}\n")
        fh.write(f"#n_genes\t{len(d)}\n#frac_above\t{frac:.5f}\n")
        fh.write(f"#frac_lo\t{np.percentile(b,2.5):.5f}\n#frac_hi\t{np.percentile(b,97.5):.5f}\n")
        fh.write(f"#median_diff\t{med:.5f}\n#nperm\t{NPERM}\n#p_exceed\t{ge}\n#p_value\t{p:.6g}\n")
        fh.write(f"#null_mean\t{null.mean():.5f}\n#null_lo\t{np.percentile(null,2.5):.5f}\n")
        fh.write(f"#null_hi\t{np.percentile(null,97.5):.5f}\n")
    return frac, med

run("gc_contrast", "gcc", T.gc_contrast.to_numpy(float))
run("spl_upstream_max_me3", "me3", T.spl_upstream_max_me3.to_numpy(float))

# sample sizes behind the gc profile
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("upstream_rowmask.npz")["ROW"]
yy = D.is_canonical.to_numpy().astype(bool)
PB = np.load("binned_features178.npz"); BN = [str(s) for s in PB["names"]]; BX = PB["X"][ridx].astype(float)
print("\n===== GC PROFILE SAMPLE SIZES =====")
with open("gc_profile_n.tsv", "w") as fh:
    fh.write("class\tposition\tn\n")
    mins = {}
    for cl, m in [("canonical", ROW & yy), ("competitor", ROW & ~yy)]:
        ns = []
        for b in range(25):
            v = BX[:, BN.index(f"bin{b+1}_gc")]
            k = int((m & np.isfinite(v)).sum()); ns.append(k)
            fh.write(f"{cl}\t{-2000 + 100*b + 50}\t{k}\n")
        mins[cl] = (min(ns), max(ns))
        print(f"  {cl:<12} n per 100 nt bin: min {min(ns):,} max {max(ns):,}"
              + ("  (constant)" if min(ns) == max(ns) else ""))
    fh.write(f"#n_canonical_min\t{mins['canonical'][0]}\t\n#n_canonical_max\t{mins['canonical'][1]}\t\n")
    fh.write(f"#n_competitor_min\t{mins['competitor'][0]}\t\n#n_competitor_max\t{mins['competitor'][1]}\t\n")
log("DONE_DIST_PANELS")
