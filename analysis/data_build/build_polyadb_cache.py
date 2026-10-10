#!/usr/bin/env python3
# cache polya_db v4 nearest-site per peak (same strand), parallel to build_atlas_cache.py
import csv, numpy as np, bisect
from collections import defaultdict
DB = "../train/data/polyadb_v4/HumanPas/hg38.PAS.main.tsv"
R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
chrom = [r["chrom"] for r in R]; cen = [int(r["center"]) for r in R]; strd = [r["strand"] for r in R]
NP = len(R)

def fv(x):
    try: return float(x)
    except (ValueError, TypeError): return np.nan

recs = defaultdict(list)
with open(DB) as fh:
    rd = csv.DictReader(fh, delimiter="\t")
    for p in rd:
        pid = p["PAS_ID"].split(":")
        if len(pid) != 3: continue
        ch, st, pos = pid[0], pid[1], int(pid[2])
        if not ch.startswith("chr"): ch = "chr" + ch
        recs[(ch, st)].append((pos, fv(p["PolyaStrength"]), fv(p["PolyaStrength_percentile"]),
                               fv(p["AvgRPM.3READS"]), fv(p["PSE.3READS"])))
for k in recs: recs[k].sort()
print("polyadb sites:", sum(len(v) for v in recs.values()))

dist = np.full(NP, 1e9); strength = np.full(NP, np.nan); pctile = np.full(NP, np.nan)
rpm = np.full(NP, np.nan); pse = np.full(NP, np.nan)
for i in range(NP):
    a = recs.get((chrom[i], strd[i]))
    if not a: continue
    pos = [x[0] for x in a]; c = cen[i]
    j = bisect.bisect_left(pos, c); best = None; bd = 1e9
    for k in (j - 1, j):
        if 0 <= k < len(a):
            dd = abs(pos[k] - c)
            if dd < bd: bd = dd; best = a[k]
    dist[i] = bd
    if best is not None:
        strength[i] = best[1]; pctile[i] = best[2]; rpm[i] = best[3]; pse[i] = best[4]
np.savez("polyadb_peak_cache.npz", dist=dist, strength=strength, pctile=pctile, rpm=rpm, pse=pse)
print("saved polyadb_peak_cache.npz  matched@250nt:", int((dist <= 250).sum()))
