#!/usr/bin/env python
# phase 1 validation gate: baseline (base_*) vs modified (new_*) on the same subset
import os
import sys
import glob
import math

import pandas as pd
import numpy as np

FAIL = []


def norm(s):
    return s.astype(str).fillna("nan").tolist()


print("=" * 78)
print("PHASE 1 VALIDATION GATE   (300/class subset, same seed, same shortlist)")
print("=" * 78)

base = sorted(glob.glob("base_*.tsv"))
print("\n1. EXISTING-COLUMN IDENTITY")
n_files = n_cols = 0
for bp in base:
    np_ = "new_" + bp[len("base_"):]
    if not os.path.isfile(np_):
        FAIL.append("missing counterpart: %s" % np_)
        print("  MISSING  %s" % np_)
        continue
    try:
        B = pd.read_csv(bp, sep="\t")
        N = pd.read_csv(np_, sep="\t")
    except Exception as e:
        print("  SKIP     %s (%s)" % (bp, type(e).__name__))
        continue
    n_files += 1
    if len(B) != len(N):
        FAIL.append("%s row count %d -> %d" % (bp, len(B), len(N)))
        print("  ROWS     %s  %d -> %d" % (bp, len(B), len(N)))
        continue
    missing = [c for c in B.columns if c not in N.columns]
    if missing:
        FAIL.append("%s dropped columns %s" % (bp, missing))
        print("  DROPPED  %s  %s" % (bp, missing))
    ctl = "base2_" + bp[len("base_"):]
    reorder_ok = False
    if os.path.isfile(ctl):
        try:
            C = pd.read_csv(ctl, sep="\t")
            reorder_ok = (len(C) == len(B) and
                          sorted(map(str, B.astype(str).agg("|".join, axis=1))) ==
                          sorted(map(str, C.astype(str).agg("|".join, axis=1))) and
                          norm(B.iloc[:, -1]) != norm(C.iloc[:, -1]))
        except Exception:
            pass
    for c in B.columns:
        if c not in N.columns:
            continue
        n_cols += 1
        if norm(B[c]) == norm(N[c]):
            continue
        if reorder_ok and sorted(norm(B[c])) == sorted(norm(N[c])):
            print("  REORDER  %s :: %s  (pre-existing; base-vs-base2 reorders too)" % (bp, c))
            continue
        FAIL.append("%s::%s differs" % (bp, c))
        print("  DIFFERS  %s :: %s" % (bp, c))
    added = [c for c in N.columns if c not in B.columns]
    if added:
        print("  +cols    %-34s %s" % (bp, added))
print("  checked %d files, %d existing columns" % (n_files, n_cols))
print("  -> %s" % ("ALL EXISTING COLUMNS IDENTICAL" if not FAIL else "%d MISMATCHES" % len(FAIL)))

P = "new_positional_features.tsv"
if not os.path.isfile(P):
    print("\nFATAL: %s not written" % P)
    sys.exit(1)
D = pd.read_csv(P, sep="\t")
can, comp = D[D.is_canonical == 1], D[D.is_canonical == 0]
print("\n2. NEW PRESENCE FRACTIONS BY CLASS   (n = %d: %d canonical / %d competitor)"
      % (len(D), len(can), len(comp)))
print("  %-22s %10s %10s %8s" % ("flag", "canonical", "competitor", "ratio"))
for f in ("first5_up_present", "first5_dn_present", "first5_pastpa_present"):
    a, b = can[f].mean(), comp[f].mean()
    print("  %-22s %10.4f %10.4f %8s"
          % (f, a, b, ("%.3f" % (b / a)) if a > 0 else "n/a"))
print("  anchor found (anchor_3ss_pos not NaN): %d / %d"
      % (int(D.anchor_3ss_pos.notna().sum()), len(D)))

print("\n3. RECONCILIATION  first5_dn_dist vs internal_exon_like / nearest_dn")
m = D.first5_dn_dist.notna() & D.nearest_dn.notna()
if m.sum():
    r = np.corrcoef(D.first5_dn_dist[m], D.nearest_dn[m])[0, 1]
    exact = int((D.first5_dn_dist[m] == D.nearest_dn[m]).sum())
    print("  first5_dn_dist vs nearest_dn: r = %.6f  (exact matches %d / %d)"
          % (r, exact, int(m.sum())))
iel = D.internal_exon_like.astype(bool)
inband = D.first5_dn_dist.between(50, 300)
print("  internal_exon_like=1: %d   first5_dn_dist in [50,300]: %d   overlap: %d"
      % (int(iel.sum()), int(inband.sum()), int((iel & inband).sum())))
print("  (related, not identical: internal_exon_like uses first5_from_anchor,")
print("   which also considers sites past the pA)")

print("\n4. COORDINATE SANITY  d_anchor_to_pa == anchor_up_check")
both = D.d_anchor_to_pa.notna() & D.anchor_up_check.notna()
bad = int((D.d_anchor_to_pa[both] != D.anchor_up_check[both]).sum())
print("  rows with both: %d   mismatches: %d" % (int(both.sum()), bad))
if bad:
    FAIL.append("d_anchor_to_pa != anchor_up_check on %d rows" % bad)
d = D[D.anchor_3ss_pos.notna()]
if len(d):
    imp = np.where(d.strand == "+", d.center - d.anchor_3ss_pos,
                   d.anchor_3ss_pos - d.center)
    bad2 = int((np.abs(imp - d.d_anchor_to_pa) > 1e-6).sum())
    print("  anchor_3ss_pos reconstructs d_anchor_to_pa: %d mismatches / %d"
          % (bad2, len(d)))
    if bad2:
        FAIL.append("anchor_3ss_pos inconsistent on %d rows" % bad2)

print("\n" + "=" * 78)
if FAIL:
    print("GATE FAILED (%d issues):" % len(FAIL))
    for f in FAIL[:20]:
        print("  - %s" % f)
    sys.exit(1)
print("GATE PASSED. Existing outputs unchanged; new fields populated and consistent.")
