#!/usr/bin/env python3
# join scanned peaks to polyasite atlas clusters to get measured per-site usage
import sys, numpy as np, pandas as pd

PREFIX = sys.argv[1]
ATLAS = sys.argv[2] if len(sys.argv) > 2 else "../train/data/atlas.chr.bed"
WIN = int(sys.argv[3]) if len(sys.argv) > 3 else 50

M = pd.read_csv(f"{PREFIX}_ucube_meta.tsv", sep="\t")
A = pd.read_csv(ATLAS, sep="\t", header=None,
                names=["chrom", "start", "end", "name", "score", "strand",
                       "frac_samples", "nr_prots", "score2", "annotation", "signal"])
A = A[np.isfinite(A.score)]
print(f"[atlas] {len(A):,} clusters ; [peaks] {len(M):,}", file=sys.stderr)

out_score = np.full(len(M), np.nan)
out_frac = np.full(len(M), np.nan)
out_nprot = np.full(len(M), np.nan)
out_annot = np.array([None] * len(M), dtype=object)

for (ch, st), grp in A.groupby(["chrom", "strand"], sort=False):
    sel = (M.chrom.values == ch) & (M.strand.values == st)
    if not sel.any(): continue
    idx = np.where(sel)[0]
    lo = grp.start.values - WIN; hi = grp.end.values + WIN
    o = np.argsort(lo); lo = lo[o]; hi = hi[o]
    sc = grp.score.values[o]; fr = grp.frac_samples.values[o]
    npr = grp.nr_prots.values[o]; ann = grp.annotation.values[o]
    hi_max = np.maximum.accumulate(hi)
    ctr = M.center.values[idx]
    j = np.searchsorted(lo, ctr, side="right")
    for k, (c, jj) in enumerate(zip(ctr, j)):
        if jj == 0 or hi_max[jj - 1] < c: continue
        best = -1.0; bi = -1
        for t in range(jj - 1, -1, -1):
            if hi_max[t] < c: break
            if hi[t] >= c and sc[t] > best: best = sc[t]; bi = t
        if bi >= 0:
            r = idx[k]
            out_score[r] = sc[bi]; out_frac[r] = fr[bi]
            out_nprot[r] = npr[bi]; out_annot[r] = ann[bi]

M["atlas_score"] = out_score
M["atlas_frac_samples"] = out_frac
M["atlas_nr_prots"] = out_nprot
M["atlas_annotation"] = out_annot
M.to_csv(f"{PREFIX}_ucube_meta_atlas.tsv", sep="\t", index=False)

hit = np.isfinite(out_score)
print(f"[join] matched {hit.sum():,}/{len(M):,} peaks ({hit.mean():.1%}) within {WIN}nt", file=sys.stderr)
for c in (1, 0):
    m = (M.is_canonical.values == c) & hit
    print(f"   is_canonical={c}: {m.sum():,} matched ; median atlas score "
          f"{np.median(out_score[m]):.3f}", file=sys.stderr)
print("\n[annotation x class]", file=sys.stderr)
print(pd.crosstab(M.atlas_annotation, M.is_canonical).to_string(), file=sys.stderr)
