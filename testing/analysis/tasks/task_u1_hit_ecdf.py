#!/usr/bin/env python3
# ecdf of the held-out discriminant for the four groups, for panel a
import sys
import numpy as np
import pandas as pd

def log(*a): print(*a, file=sys.stderr, flush=True)
NPT = 400

T = pd.read_parquet("cpa_peaks_final.parquet", engine="fastparquet")
A = T[T.in_analysis_set].reset_index(drop=True).rename(columns={"center": "pos"})
P = pd.read_csv("ovl_ana_per_peak.tsv", sep="\t")
D = A.merge(P[["chrom", "pos", "strand", "gene", "hit_oh_u1amo_50", "hit_oh_camo_50"]],
            on=["chrom", "pos", "strand", "gene"], how="left", validate="1:1")
FP = ~D.is_canonical.to_numpy()
G = [("canonical", D.is_canonical.to_numpy()),
     ("U1 AMO hit", FP & D.hit_oh_u1amo_50.astype(bool).to_numpy()),
     ("control AMO hit", FP & D.hit_oh_camo_50.astype(bool).to_numpy()),
     ("not hit", FP & ~D.hit_oh_u1amo_50.astype(bool).to_numpy())]

rows = []
print(f"{'group':<18}{'n':>8}{'Q1':>9}{'median':>9}{'Q3':>9}")
print("-" * 53)
for lab, m in G:
    v = np.sort(D.discriminant.to_numpy(float)[m & np.isfinite(D.discriminant.to_numpy(float))])
    n = len(v)
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    print(f"{lab:<18}{n:>8,}{q1:>9.3f}{md:>9.3f}{q3:>9.3f}")
    k = np.unique(np.linspace(0, n - 1, min(NPT, n)).astype(int))
    for i in k:
        rows.append(dict(group=lab, value=float(v[i]), ecdf=(i + 1) / n, n=n,
                         q1=q1, median=md, q3=q3))
pd.DataFrame(rows).to_csv("u1_hit_ecdf.tsv", sep="\t", index=False)
print("\nwrote u1_hit_ecdf.tsv")
log("DONE_U1_HIT_ECDF")
