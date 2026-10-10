#!/usr/bin/env python3
# same hand-check as task_notmost_handcheck.py, but sourced from the parquet peak master
import numpy as np, pyarrow.parquet as pq
from collections import defaultdict

PARQ = "rerank_genomewide_scored.parquet"
t = pq.read_table(PARQ, columns=["gene", "chrom", "strand", "center", "is_canonical"])
gene = np.array(t["gene"].to_pylist()); chrom = np.array(t["chrom"].to_pylist())
strd = np.array(t["strand"].to_pylist()); cen = np.array(t["center"].to_pylist(), dtype=np.int64)
isc = np.array(t["is_canonical"].to_pylist()).astype(int)
NK = len(gene); print(f"# source: {PARQ}  peaks={NK}")

gmap = defaultdict(list)
for i in range(NK): gmap[gene[i]].append(i)

# exact 3'-end rank (transcript orientation): + ranks by descending center, - by ascending
tc = np.where(strd == "+", cen, -cen).astype(float)
rank3p = np.full(NK, -1, np.int64)
for g, idxs in gmap.items():
    idx = np.array(idxs); t_ = tc[idx]
    order = np.argsort(-t_, kind="mergesort")
    r = np.empty(len(idx), np.int64); r[order] = np.arange(1, len(idx) + 1)
    rank3p[idx] = r

GENES = {}
for g, idxs in gmap.items():
    idx = np.array(idxs); c = idx[isc[idx] == 1]
    if len(c) == 1 and len(idx) >= 2: GENES[g] = (int(c[0]), idx)

not3 = {g: v for g, v in GENES.items() if rank3p[v[0]] != 1}
plus = sorted([g for g in not3 if strd[not3[g][0]] == "+"], key=lambda g: len(not3[g][1]))
minus = sorted([g for g in not3 if strd[not3[g][0]] == "-"], key=lambda g: len(not3[g][1]))

def show(glist, lab):
    print(f"\n==== {lab} strand: 5 genes where canonical is NOT 3'-most ====")
    print(f"{'gene':14s} {'chrom':6s} {'st':2s} {'center':>10s} {'is_can':>6s} {'rank3p':>6s}")
    for g in glist[:5]:
        c, idx = not3[g]
        for i in idx[np.argsort(rank3p[idx])]:
            mark = " <== CANONICAL" if isc[i] == 1 else ""
            print(f"{gene[i]:14s} {chrom[i]:6s} {strd[i]:2s} {cen[i]:>10d} {isc[i]:>6d} {rank3p[i]:>6d}{mark}")
        print()

show(plus, "+"); show(minus, "-")

print("==== EXACT 3'-END RANK CODE ====")
print('  tc = np.where(strd == "+", cen, -cen)   # + strand: descending center = downstream;')
print('                                          # - strand: negate -> ascending center = downstream')
print('  order = np.argsort(-tc[idx])            # most downstream first')
print('  rank[order] = np.arange(1, n+1)         # rank 1 = 3\'-most peak')

print("\n==== SUMMARY 1: fraction canonical is 3'-most, by strand ====")
fr = {}
for st in ("+", "-"):
    gs = [g for g in GENES if strd[GENES[g][0]] == st]
    n3 = sum(rank3p[GENES[g][0]] == 1 for g in gs); fr[st] = n3 / len(gs)
    print(f"  {st} strand: {n3}/{len(gs)} = {fr[st]:.4f} 3'-most  ({len(gs)-n3} not = {(len(gs)-n3)/len(gs):.4f})")
n3all = sum(rank3p[GENES[g][0]] == 1 for g in GENES)
print(f"  overall : {n3all}/{len(GENES)} = {n3all/len(GENES):.4f} 3'-most  ({len(GENES)-n3all} not)")
gap = abs(fr["+"] - fr["-"]) * 100
print(f"  + vs - gap = {gap:.2f} pts  {'<== FLAG: >2 pts, possible strand-handling bug' if gap>2 else '(within a couple of points: strand flip consistent, not a bug)'}")

print("\n==== SUMMARY 2: among not-3'-most genes, downstream peaks past the canonical ====")
ndown, alldist, maxdist = [], [], []
for g, (c, idx) in not3.items():
    dn = idx[tc[idx] > tc[c]]; ndown.append(len(dn))
    d = tc[dn] - tc[c]; alldist.extend(d.tolist()); maxdist.append(d.max())
ndown = np.array(ndown); alldist = np.array(alldist); maxdist = np.array(maxdist)
print(f"  genes where canonical is not 3'-most: {len(not3)}")
print(f"  # downstream peaks per such gene: median {np.median(ndown):.0f}  mean {ndown.mean():.2f}  max {ndown.max()}")
print(f"  canonical->each downstream peak (nt): median {np.median(alldist):.0f}  "
      f"25/75 {np.percentile(alldist,25):.0f}/{np.percentile(alldist,75):.0f}  max {alldist.max():.0f}")
print(f"  canonical->3'-most peak (nt):        median {np.median(maxdist):.0f}  "
      f"25/75 {np.percentile(maxdist,25):.0f}/{np.percentile(maxdist,75):.0f}  max {maxdist.max():.0f}")
