#!/usr/bin/env python3
# p@1 and mrr for the two-feature model, holdout protocol, against their own chance rates
import numpy as np, pandas as pd
from collections import defaultdict
NPERM = 10000
rng = np.random.default_rng(42)
T = pd.read_parquet("cpa_peaks_final.parquet", engine="fastparquet")
T = T[T.in_analysis_set & T.in_gene_holdout].reset_index(drop=True)
y = T.is_canonical.to_numpy(); s = T.discriminant.to_numpy(float); gene = T.gene.to_numpy()
gi = defaultdict(list)
for i in range(len(T)): gi[gene[i]].append(i)
G = [np.array(v) for v in gi.values() if len(v) >= 2 and y[np.array(v)].sum() == 1]
print(f"holdout {len(G):,} genes, {sum(len(g) for g in G):,} peaks, "
      f"mean {np.mean([len(g) for g in G]):.1f} candidates per gene")

def stats(pos_of):
    p1 = np.empty(len(G)); mrr = np.empty(len(G))
    for t, ix in enumerate(G):
        v = s[ix]; k = pos_of(t, ix)
        gt = int((v > v[k]).sum()); eq = int((v == v[k]).sum())
        r0 = gt + 1
        p1[t] = 1.0 / eq if r0 == 1 else 0.0
        mrr[t] = float(np.mean(1.0 / np.arange(r0, r0 + eq)))
    return float(p1.mean()), float(mrr.mean())

obs_p1, obs_mrr = stats(lambda t, ix: int(np.argmax(y[ix])))
n = np.array([len(g) for g in G], float)
H = np.array([np.sum(1.0 / np.arange(1, int(k) + 1)) for k in n])
ch_p1, ch_mrr = float(np.mean(1.0 / n)), float(np.mean(H / n))
P1 = np.empty(NPERM); MR = np.empty(NPERM)
for b in range(NPERM):
    P1[b], MR[b] = stats(lambda t, ix: int(rng.integers(0, len(ix))))
q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
print(f"\n{'metric':<6} {'observed':>9} {'chance, analytic':>17} {'chance, permutation':>21} {'ratio':>7}")
for nm, o, c, perm in [("P@1", obs_p1, ch_p1, P1), ("MRR", obs_mrr, ch_mrr, MR)]:
    lo, hi = q(perm)
    print(f"{nm:<6} {o:>9.4f} {c:>17.4f} {perm.mean():>10.4f} [{lo:.4f},{hi:.4f}] {o/c:>7.2f}x")
with open("p1_mrr.tsv", "w") as fh:
    fh.write("metric\tobserved\tchance_analytic\tchance_perm\tperm_lo\tperm_hi\tnperm\tn_genes\n")
    for nm, o, c, perm in [("P@1", obs_p1, ch_p1, P1), ("MRR", obs_mrr, ch_mrr, MR)]:
        lo, hi = q(perm)
        fh.write(f"{nm}\t{o:.5f}\t{c:.5f}\t{perm.mean():.5f}\t{lo:.5f}\t{hi:.5f}\t{NPERM}\t{len(G)}\n")
