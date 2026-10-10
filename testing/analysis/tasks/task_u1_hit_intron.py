#!/usr/bin/env python3
# host-intron rank of the hit and non-hit false positives, for the figure
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

def log(*a): print(*a, file=sys.stderr, flush=True)

T = pd.read_parquet("cpa_peaks_final.parquet", engine="fastparquet")
A = T[T.in_analysis_set].reset_index(drop=True).rename(columns={"center": "pos"})
P = pd.read_csv("ovl_ana_per_peak.tsv", sep="\t")
D = A.merge(P[["chrom", "pos", "strand", "gene", "hit_oh_u1amo_50", "hit_oh_camo_50"]],
            on=["chrom", "pos", "strand", "gene"], how="left", validate="1:1")
FP = ~D.is_canonical.to_numpy()
U1 = FP & D.hit_oh_u1amo_50.astype(bool).to_numpy()
CT = FP & D.hit_oh_camo_50.astype(bool).to_numpy()
NH = FP & ~D.hit_oh_u1amo_50.astype(bool).to_numpy()

EX = pd.read_csv("gencode_canonical_exons.tsv", sep="\t", header=None,
                 names=["chrom", "start", "end", "strand", "gene", "exon"])
EX = EX[EX.gene.isin(set(D.gene))]
intr = {}
for g, sub in EX.groupby("gene", sort=False):
    sub = sub.sort_values("start")
    st = sub.start.to_numpy(); en = sub.end.to_numpy()
    iv = [(en[k], st[k + 1] - 1) for k in range(len(st) - 1) if st[k + 1] - 1 > en[k]]
    if sub.strand.iloc[0] == "-": iv = iv[::-1]
    if iv: intr[g] = iv
rank = np.full(len(D), np.nan)
gn = D.gene.to_numpy(); pos = D.pos.to_numpy()
for i in range(len(D)):
    iv = intr.get(gn[i])
    if not iv: continue
    p = pos[i]
    for r, (a, b) in enumerate(iv, 1):
        if a <= p <= b: rank[i] = r; break

def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - hw), min(1.0, c + hw)

ok = np.isfinite(rank)
rows = []
print(f"{'group':<18}{'n':>8}{'first two introns':>20}{'95% CI':>18}{'median rank':>13}")
print("-" * 77)
for lab, m in [("U1 AMO hit", U1), ("control AMO hit", CT), ("not hit", NH)]:
    s = rank[m & ok]; n = len(s); k = int((s <= 2).sum())
    lo, hi = wilson(k, n)
    print(f"{lab:<18}{n:>8,}{100*k/n:>19.2f}%{f'{100*lo:.2f}-{100*hi:.2f}':>18}{np.median(s):>13.1f}")
    for r in range(1, 11):
        c = int((s == r).sum()) if r < 10 else int((s >= 10).sum())
        rows.append(dict(group=lab, rank=r, label=str(r) if r < 10 else "10+",
                         n=c, frac=c / n, n_group=n,
                         frac_first_two=k / n, ci_lo=lo, ci_hi=hi,
                         median_rank=float(np.median(s))))
pd.DataFrame(rows).to_csv("u1_hit_intron.tsv", sep="\t", index=False)
print("\nwrote u1_hit_intron.tsv")
log("DONE_U1_HIT_INTRON")
