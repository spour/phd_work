#!/usr/bin/env python3
# take every competitor peak that is further downstream than its gene's canonical
import numpy as np, pyarrow.parquet as pq
from collections import defaultdict
import bisect

PARQ = "rerank_genomewide_scored.parquet"
BED = "/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/clean/train/data/Dominant_CPA_baseline_total.bed"

t = pq.read_table(PARQ, columns=["gene", "chrom", "strand", "center", "is_canonical"])
gene = np.array(t["gene"].to_pylist()); chrom = np.array(t["chrom"].to_pylist())
strd = np.array(t["strand"].to_pylist()); cen = np.array(t["center"].to_pylist(), dtype=np.int64)
isc = np.array(t["is_canonical"].to_pylist()).astype(int)
NK = len(gene)

gmap = defaultdict(list)
for i in range(NK): gmap[gene[i]].append(i)
tc = np.where(strd == "+", cen, -cen).astype(float)

dn_idx = []
for g, idxs in gmap.items():
    idx = np.array(idxs); c = idx[isc[idx] == 1]
    if len(c) != 1 or len(idx) < 2: continue
    c = int(c[0])
    if tc[c] == tc[idx].max(): continue
    for i in idx:
        if isc[i] == 0 and tc[i] > tc[c]: dn_idx.append(i)
dn_idx = np.array(dn_idx)
print(f"downstream-of-canonical competitor peaks (in {len({gene[i] for i in dn_idx})} not-3'-most genes): {len(dn_idx)}")

by_c = defaultdict(list); by_cs = defaultdict(list)
for L in open(BED):
    p = L.rstrip("\n").split("\t")
    if len(p) < 3: continue
    ch, s, e = p[0], int(p[1]), int(p[2]); st = p[5] if len(p) > 5 else "."
    by_c[ch].append((s, e)); by_cs[(ch, st)].append((s, e))
for d in (by_c, by_cs):
    for k in d: d[k].sort()

def hit(table, ch_key, pos):
    iv = table.get(ch_key)
    if not iv: return False
    starts = [s for s, _ in iv]
    j = bisect.bisect_right(starts, pos) - 1
    for k in range(max(0, j - 2), min(len(iv), j + 3)):
        s, e = iv[k]
        if s <= pos < e: return True
    return False

n_any = sum(hit(by_c, chrom[i], int(cen[i])) for i in dn_idx)
n_str = sum(hit(by_cs, (chrom[i], strd[i]), int(cen[i])) for i in dn_idx)
print(f"  center inside a baseline interval (chrom match only):        {n_any}/{len(dn_idx)} = {n_any/len(dn_idx):.3f}")
print(f"  center inside a baseline interval (chrom + strand match):     {n_str}/{len(dn_idx)} = {n_str/len(dn_idx):.3f}")

can_idx = []
for g, idxs in gmap.items():
    idx = np.array(idxs); c = idx[isc[idx] == 1]
    if len(c) != 1 or len(idx) < 2: continue
    c = int(c[0])
    if tc[c] == tc[idx].max(): continue
    can_idx.append(c)
can_idx = np.array(can_idx)
nc = sum(hit(by_cs, (chrom[i], strd[i]), int(cen[i])) for i in can_idx)
print(f"  (ref) canonicals of the same not-3'-most genes, strand match: {nc}/{len(can_idx)} = {nc/len(can_idx):.3f}")
