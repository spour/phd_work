#!/usr/bin/env python3
# confirm cpa_peaks_final.parquet before it is shared, and settle the apa flag check
import csv, sys
import numpy as np
import pandas as pd
from collections import defaultdict

def log(*a): print(*a, file=sys.stderr, flush=True)
ATLAS = "../train/data/atlas.clusters.2.0.GRCh38.96.tsv"
POLYADB = "../train/data/polyadb_v4/HumanPas/hg38.PAS.main.tsv"
PARQ = "cpa_peaks_final.parquet"
WINS = [0, 1, 25, 500]

import os
print("===== 1. THE FILE =====")
print(f"{PARQ}  exists: {os.path.exists(PARQ)}  "
      f"{os.path.getsize(PARQ):,} bytes  modified {pd.Timestamp(os.path.getmtime(PARQ), unit='s')}")
T = pd.read_parquet(PARQ, engine="fastparquet")
print(f"rows {len(T):,}   columns {T.shape[1]}")
for c in T.columns:
    print(f"  {c:<32}{str(T[c].dtype):<10}{int(T[c].notna().sum()):>7,} non-null")
for other in ["rerank_corr_peaks.parquet", "rerank_corr_peaks.SUPERSEDED.parquet"]:
    if os.path.exists(other):
        print(f"also present: {other}"
              + ("   <- SUPERSEDED, do not share" if "SUPERSEDED" in other else
                 "   <- the old file is back under its original name"))

print("\n===== 2. THE COUNTS =====")
A = T[T.in_analysis_set]
comp = ~T.is_canonical
checks = [("in_analysis_set peaks", len(A), 48394),
          ("  canonical", int(A.is_canonical.sum()), 4847),
          ("  competitor", int((~A.is_canonical).sum()), 43547),
          ("  genes", A.gene.nunique(), 4847),
          ("APA-annotated", int((comp & T.flag_apa_annotated).sum()), 18119),
          ("reference terminal exon, beyond the APA set",
           int((comp & T.flag_reference_terminal_exon & ~T.flag_apa_annotated).sum()), 1076),
          ("downstream of the canonical, beyond the first two",
           int((comp & ~T.flag_apa_annotated & ~T.flag_reference_terminal_exon
                & T.flag_downstream_of_canonical).sum()), 1398)]
ok = True
for lab, got, want in checks:
    good = got == want; ok &= good
    print(f"  {lab:<52}{got:>8,}   expected {want:>8,}   {'OK' if good else 'MISMATCH'}")
print(f"  ALL COUNTS {'PASS' if ok else 'FAIL'}")

def load_sites(path, kind):
    recs = defaultdict(list)
    if kind == "atlas":
        with open(path) as fh:
            fh.readline()
            for line in fh:
                p = line.rstrip("\n").split("\t")
                c = p[0] if p[0].startswith("chr") else "chr" + p[0]
                recs[(c, p[5])].append(int(p[6]))
    else:
        with open(path) as fh:
            for p in csv.DictReader(fh, delimiter="\t"):
                q = p["PAS_ID"].split(":")
                if len(q) != 3: continue
                c = q[0] if q[0].startswith("chr") else "chr" + q[0]
                recs[(c, q[1])].append(int(q[2]))
    for k in recs: recs[k] = np.sort(np.asarray(recs[k]))
    print(f"  {path}: {sum(len(v) for v in recs.values()):,} sites")
    return recs

def signed_nearest(chrom, pos, strand, recs):
    d = np.full(len(pos), np.nan)
    K = pd.DataFrame({"c": chrom, "p": pos, "s": strand})
    for (c, s), sub in K.groupby(["c", "s"], sort=False):
        arr = recs.get((c, s))
        if arr is None or arr.size == 0: continue
        p = sub["p"].to_numpy()
        i = np.searchsorted(arr, p)
        lo_i = np.clip(i - 1, 0, arr.size - 1); hi_i = np.clip(i, 0, arr.size - 1)
        lo = np.where(i > 0, arr[lo_i] - p, np.inf)
        hi = np.where(i < arr.size, arr[hi_i] - p, np.inf)
        take = np.where(np.abs(lo) <= np.abs(hi), lo, hi)
        d[sub.index.to_numpy()] = take
    return d

print("\n===== 3. THE APA FLAG CHECK =====")
CAN = T[T.is_canonical].reset_index(drop=True)
print(f"canonical peaks: {len(CAN):,}")
print(f"fraction with flag_apa_annotated (the flag as built, 500 nt): "
      f"{CAN.flag_apa_annotated.mean():.4f}  "
      f"({int(CAN.flag_apa_annotated.sum()):,} of {len(CAN):,})")

print("\n  shortlist geometry, which sets the ceiling on an exact match:")
step = np.diff(np.sort(T.center.unique()))
print(f"    distinct centres {T.center.nunique():,}; smallest gaps seen "
      f"{np.unique(step[step > 0])[:6]}")
print(f"    centre mod 50: {dict(pd.Series(T.center % 50).value_counts().head(3))}")

rows = []
for lab, path, kind in [("PolyASite 2.0", ATLAS, "atlas"), ("PolyA_DB v4", POLYADB, "db")]:
    R = load_sites(path, kind)
    d = signed_nearest(CAN.chrom.to_numpy(), CAN.center.to_numpy(), CAN.strand.to_numpy(), R)
    fin = np.isfinite(d)
    print(f"\n  --- {lab} ---")
    near = d[fin & (np.abs(d) <= 5)]
    if len(near):
        vals, cnt = np.unique(near.astype(int), return_counts=True)
        print(f"    signed offset within +/-5 nt: n={len(near):,}  mode "
              f"{vals[np.argmax(cnt)]:+d} nt   " +
              "  ".join(f"{v:+d}:{c}" for v, c in zip(vals, cnt)))
    q = np.percentile(np.abs(d[fin]), [10, 25, 50, 75, 90])
    print(f"    |offset| deciles: p10 {q[0]:,.0f}  p25 {q[1]:,.0f}  median {q[2]:,.0f}  "
          f"p75 {q[3]:,.0f}  p90 {q[4]:,.0f} nt")
    for w in WINS:
        k = int((np.abs(d[fin]) <= w).sum())
        print(f"    within +/-{w:<4} nt: {k:>6,} of {len(CAN):,}   {k/len(CAN):.4f}")
        rows.append(dict(db=lab, window_nt=w, n_canonical=len(CAN), n_matched=k,
                         frac=k / len(CAN)))
Ra = load_sites(ATLAS, "atlas"); Rd = load_sites(POLYADB, "db")
da = signed_nearest(CAN.chrom.to_numpy(), CAN.center.to_numpy(), CAN.strand.to_numpy(), Ra)
db = signed_nearest(CAN.chrom.to_numpy(), CAN.center.to_numpy(), CAN.strand.to_numpy(), Rd)
best = np.nanmin(np.abs(np.vstack([da, db])), axis=0)
print(f"\n  --- either database, which is the flag's rule ---")
for w in WINS:
    k = int((best <= w).sum())
    print(f"    within +/-{w:<4} nt: {k:>6,} of {len(CAN):,}   {k/len(CAN):.4f}")
    rows.append(dict(db="either", window_nt=w, n_canonical=len(CAN), n_matched=k,
                     frac=k / len(CAN)))
agree = int(((best <= 500) == CAN.flag_apa_annotated.to_numpy()).sum())
print(f"\n  recomputed 500 nt rule agrees with flag_apa_annotated on "
      f"{agree:,} of {len(CAN):,} canonical peaks  "
      f"{'OK' if agree == len(CAN) else 'MISMATCH'}")
pd.DataFrame(rows).to_csv("apa_flag_check.tsv", sep="\t", index=False)
print("\nwrote apa_flag_check.tsv")
log("DONE_APA_FLAG_CHECK")
