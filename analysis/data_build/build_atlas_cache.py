#!/usr/bin/env python3
# cache polyasite 2.0 per-site usage for each of the 70723 peaks (nearest cluster
import csv, numpy as np, bisect
from collections import defaultdict
ATL = "../train/data/atlas.clusters.2.0.GRCh38.96.tsv"
R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
chrom = [r["chrom"] for r in R]; cen = [int(r["center"]) for r in R]; strd = [r["strand"] for r in R]
NP = len(R)

recs = defaultdict(list)
with open(ATL) as fh:
    fh.readline()
    for line in fh:
        p = line.rstrip("\n").split("\t")
        ch = p[0]
        if not ch.startswith("chr"): ch = "chr" + ch
        st = p[5]; rep = int(p[6]); tpm = float(p[4]); frac = float(p[7]); nprot = int(p[8])
        samp = p[13:]
        nsamp = sum(1 for x in samp if x not in ("0", "0.0", "", "0.000000"))
        recs[(ch, st)].append((rep, tpm, frac, nprot, nsamp))
for k in recs: recs[k].sort()
print("atlas clusters:", sum(len(v) for v in recs.values()))

dist = np.full(NP, 1e9); tpm = np.zeros(NP); frac = np.zeros(NP); nprot = np.zeros(NP, int); nsamp = np.zeros(NP, int)
for i in range(NP):
    a = recs.get((chrom[i], strd[i]))
    if not a: continue
    reps = [x[0] for x in a]; c = cen[i]
    j = bisect.bisect_left(reps, c); best = None; bd = 1e9
    for k in (j - 1, j):
        if 0 <= k < len(a):
            dd = abs(reps[k] - c)
            if dd < bd: bd = dd; best = a[k]
    dist[i] = bd
    if best is not None:
        tpm[i] = best[1]; frac[i] = best[2]; nprot[i] = best[3]; nsamp[i] = best[4]
np.savez("atlas_peak_cache.npz", dist=dist, tpm=tpm, frac=frac, nprot=nprot, nsamp=nsamp)
print("saved atlas_peak_cache.npz  matched@25nt:", int((dist <= 25).sum()))
