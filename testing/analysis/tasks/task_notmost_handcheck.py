#!/usr/bin/env python3
# pull genes where the canonical is not the 3'-most peak
import csv
import numpy as np
from collections import defaultdict

R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
NK = len(R)
gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
chrom = np.array([r["chrom"] for r in R]); cen = np.array([int(r["center"]) for r in R])
strd = np.array([r["strand"] for r in R])

gmap = defaultdict(list)
for i in range(NK): gmap[gene[i]].append(i)

# exact 3'-end rank (transcript orientation), copied from task_position_increment.py
tc = np.where(strd == "+", cen, -cen).astype(float)
rank3p = np.full(NK, -1, np.int64)
for g, idxs in gmap.items():
    idx = np.array(idxs); t = tc[idx]
    order = np.argsort(-t, kind="mergesort")
    r = np.empty(len(idx), np.int64); r[order] = np.arange(1, len(idx) + 1)
    rank3p[idx] = r

GENES = {}
for g, idxs in gmap.items():
    idx = np.array(idxs); c = idx[isc[idx] == 1]
    if len(c) == 1 and len(idx) >= 2: GENES[g] = (int(c[0]), idx)

def canon_is_3most(cpos): return rank3p[cpos] == 1

not3 = {g: v for g, v in GENES.items() if not canon_is_3most(v[0])}

plus = sorted([g for g in not3 if strd[not3[g][0]] == "+"], key=lambda g: len(not3[g][1]))
minus = sorted([g for g in not3 if strd[not3[g][0]] == "-"], key=lambda g: len(not3[g][1]))

def show(glist, lab):
    print(f"\n==== {lab} strand: 5 genes where canonical is NOT 3'-most ====")
    print(f"{'gene':14s} {'chrom':6s} {'st':2s} {'center':>10s} {'is_can':>6s} {'rank3p':>6s}")
    for g in glist[:5]:
        c, idx = not3[g]
        order = idx[np.argsort(rank3p[idx])]
        for i in order:
            mark = " <== CANONICAL" if isc[i] == 1 else ""
            print(f"{gene[i]:14s} {chrom[i]:6s} {strd[i]:2s} {cen[i]:>10d} {isc[i]:>6d} {rank3p[i]:>6d}{mark}")
        print()

show(plus, "+")
show(minus, "-")

print("==== EXACT 3'-END RANK CODE (from task_position_increment.py:34-40) ====")
print('  tc = np.where(strd == "+", cen, -cen)   # + strand: larger center = downstream;')
print('                                          # - strand: negate so larger = downstream')
print('  order = np.argsort(-tc[idx])            # most downstream first')
print('  rank[order] = arange(1, n+1)            # rank 1 = 3\'-most peak')

# summary 1: fraction canonical-is-3'-most, split by strand
print("\n==== SUMMARY 1: fraction of genes where canonical is 3'-most, by strand ====")
for st in ("+", "-"):
    gs = [g for g, (c, idx) in GENES.items() if strd[c] == st]
    n3 = sum(canon_is_3most(GENES[g][0]) for g in gs)
    print(f"  {st} strand: {n3}/{len(gs)} = {n3/len(gs):.4f} 3'-most  ({len(gs)-n3} not, {(len(gs)-n3)/len(gs):.4f})")
allg = list(GENES)
n3all = sum(canon_is_3most(GENES[g][0]) for g in allg)
print(f"  overall : {n3all}/{len(allg)} = {n3all/len(allg):.4f} 3'-most  ({len(allg)-n3all} not)")
fplus = sum(canon_is_3most(GENES[g][0]) for g in GENES if strd[GENES[g][0]] == "+") / max(1, sum(strd[GENES[g][0]] == "+" for g in GENES))
fminus = sum(canon_is_3most(GENES[g][0]) for g in GENES if strd[GENES[g][0]] == "-") / max(1, sum(strd[GENES[g][0]] == "-" for g in GENES))
print(f"  + minus - gap = {abs(fplus-fminus)*100:.2f} pts  {'<== FLAG: >2 pts, possible strand-handling bug' if abs(fplus-fminus)>0.02 else '(within a couple of points: strand flip looks consistent)'}")

# summary 2: among not-3'-most genes, how far past the canonical
print("\n==== SUMMARY 2: among not-3'-most genes, downstream peaks past the canonical ====")
ndown = []; maxdist = []; alldist = []
for g, (c, idx) in not3.items():
    dn = idx[tc[idx] > tc[c]]
    ndown.append(len(dn))
    d = (tc[dn] - tc[c])
    alldist.extend(d.tolist()); maxdist.append(d.max())
ndown = np.array(ndown); alldist = np.array(alldist); maxdist = np.array(maxdist)
print(f"  genes where canonical is not 3'-most: {len(not3)}")
print(f"  # downstream peaks per such gene: median {np.median(ndown):.0f}  mean {ndown.mean():.2f}  max {ndown.max()}")
print(f"  distance canonical->each downstream peak (nt): median {np.median(alldist):.0f}  "
      f"25/75 pct {np.percentile(alldist,25):.0f}/{np.percentile(alldist,75):.0f}  max {alldist.max():.0f}")
print(f"  distance canonical->3'-most peak (nt): median {np.median(maxdist):.0f}  "
      f"25/75 pct {np.percentile(maxdist,25):.0f}/{np.percentile(maxdist,75):.0f}  max {maxdist.max():.0f}")
