#!/usr/bin/env python3
# where the true 3' end sits among its gene's candidates, before and after
import sys, collections
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
import pyarrow.parquet as pq
S = np.load("splice26_loco_scores.npz", allow_pickle=True)
sg = np.array([str(x) for x in S["gene"]]); sc = S["center"].astype(np.int64)
sv = S["score"].astype(float); sy = S["y"].astype(int)
Z = np.load("cnn_wg_scores.npz", allow_pickle=True)
zmap = {(str(g), int(c)): i for g, c, i in zip(Z["gene"], Z["center"], range(len(Z["gene"])))}
zl = Z["logit"].astype(float)
B = pq.read_table("shortlist_feat_all.parquet",
                  columns=["gene", "center", "rf_prob"]).to_pandas()
bmap = {(g, int(c)): v for g, c, v in zip(B.gene, B.center, B.rf_prob)}
log(f"[in] splice LOCO {len(sg):,} peaks, CNN LOCO {len(zl):,}, baseline rf_prob {len(bmap):,}")
gi = collections.defaultdict(list)
for i in range(len(sg)): gi[sg[i]].append(i)
def zs(v):
    s = v.std(); return (v-v.mean())/(s if s > 1e-9 else 1.0)
def rank_of_true(v, y):
    t = v[y == 1][0]
    return int((v > t).sum())+1
rows = []
skip = 0
for g, idx in gi.items():
    idx = np.array(idx); y = sy[idx]
    if y.sum() != 1 or len(idx) < 2: continue
    keys = [(g, int(sc[i])) for i in idx]
    if not all(k in zmap for k in keys) or not all(k in bmap for k in keys):
        skip += 1; continue
    rf = np.array([bmap[k] for k in keys], float)
    cn = zl[np.array([zmap[k] for k in keys])]
    sp = sv[idx]
    combo = zs(sp)+zs(cn)
    rows.append(dict(gene=g, n=len(idx),
                     rank_baseline=rank_of_true(rf, y),
                     rank_splice=rank_of_true(sp, y),
                     rank_cnn=rank_of_true(cn, y),
                     rank_combo=rank_of_true(combo, y)))
log(f"[rank] {len(rows):,} genes ({skip:,} skipped for missing scores)")
R = rows
def dist(k, lab):
    v = np.array([r[k] for r in R])
    log(f"  {lab:<16} rank 1 {np.mean(v==1):.4f}   rank<=2 {np.mean(v<=2):.4f}   "
        f"rank<=3 {np.mean(v<=3):.4f}   median {np.median(v):.0f}   max {v.max()}")
log("\n[rank of the true 3' end among its gene's candidates]")
for k, lab in (("rank_baseline","baseline RF"),("rank_splice","splice 26"),
               ("rank_cnn","extended CNN"),("rank_combo","combo z-sum")):
    dist(k, lab)
v = np.array([r["rank_baseline"] for r in R])
log(f"\n  baseline rank 1 fraction is {np.mean(v==1):.4f}, which must be 0 by construction")
with open("rescue_ranks.tsv", "w") as f:
    f.write("gene\tn_candidates\trank_baseline\trank_splice\trank_cnn\trank_combo\n")
    for r in R: f.write(f"{r['gene']}\t{r['n']}\t{r['rank_baseline']}\t{r['rank_splice']}\t"
                        f"{r['rank_cnn']}\t{r['rank_combo']}\n")
log("[out] rescue_ranks.tsv")
