#!/usr/bin/env python3
# rebuild the oh 2017 peak call at three cut sizes, so the top-n choice can be tested
import sys
import numpy as np
import pandas as pd

def log(*a): print(*a, file=sys.stderr, flush=True)
BW = {"camo":  "external/geo/GSE103252/GSE103252_cAMO_15nmol_odT_selection.polyA_sites.bw",
      "u1amo": "external/geo/GSE103252/GSE103252_U1AMO_15nmol_odT_selection.polyA_sites.bw"}
PUB = {"camo": "external/peaks/oh2017_cAMO_hg19.tsv",
       "u1amo": "external/peaks/oh2017_U1AMO_hg19.tsv"}
NS = [20000, 40000, 60000]
MERGE = 25

def call_blocks(path, merge_within=MERGE):
    import pyBigWig
    bw = pyBigWig.open(path); rows = []
    for chrom in bw.chroms():
        try: ivs = bw.intervals(chrom)
        except RuntimeError: continue
        if not ivs: continue
        keep = [(s, e, abs(v)) for s, e, v in ivs if abs(v) > 0.0]
        if not keep: continue
        b_end, best_v, best_p = keep[0][1], keep[0][2], keep[0][0]
        for s, e, v in keep[1:]:
            if s - b_end <= merge_within:
                b_end = max(b_end, e)
                if v > best_v: best_v, best_p = v, s
            else:
                rows.append((chrom, best_p + 1, best_v)); b_end, best_v, best_p = e, v, s
        rows.append((chrom, best_p + 1, best_v))
    bw.close()
    return pd.DataFrame(rows, columns=["chrom", "pos", "value"])

for cond in ["camo", "u1amo"]:
    B = call_blocks(BW[cond])
    pub = pd.read_csv(PUB[cond], sep="\t")
    pub_p = pub[pub.strand == "+"]
    top40 = B.nlargest(40000, "value")
    same = len(set(zip(top40.chrom, top40.pos)) & set(zip(pub_p.seqnames, pub_p.start)))
    print(f"{cond}: {len(B):,} merged blocks   reproduces the published 40,000: "
          f"{same:,}/40,000   {'OK' if same == 40000 else 'MISMATCH'}")
    assert same == 40000, f"{cond}: could not reproduce the published peak call"
    nfloor = int((B.value == B.value.min()).sum())
    print(f"   floor value {B.value.min():.4f} on {nfloor:,} blocks; "
          f"value at rank 40,000 is {top40.value.min():.4f}")
    for N in NS:
        t = B.nlargest(N, "value")
        out = pd.concat([t.assign(strand="+"), t.assign(strand="-")], ignore_index=True)
        f = f"ohN_{cond}_{N}_hg19.tsv"
        out[["chrom", "pos", "strand"]].to_csv(f, sep="\t", index=False)
        print(f"   N={N:>6,}  cut value {t.value.min():.4f}  wrote {f} ({len(out):,} rows)")
log("DONE_OH_TOPN")
