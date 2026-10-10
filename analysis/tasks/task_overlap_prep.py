#!/usr/bin/env python3
# prepare the inputs for the external cpa site overlap, and verify the apa filter
import csv, sys, bisect
import numpy as np
import pandas as pd
from collections import defaultdict

def log(*a): print(*a, file=sys.stderr, flush=True)

ATLAS = "../train/data/atlas.clusters.2.0.GRCh38.96.tsv"
POLYADB = "../train/data/polyadb_v4/HumanPas/hg38.PAS.main.tsv"
WIN = 500
DEDUP = 25

# peak tables
T = pd.read_parquet("cpa_peaks_final.parquet", engine="fastparquet")
comp = ~T.is_canonical
keep = comp & ~T.flag_apa_annotated & ~T.flag_reference_terminal_exon
print(f"peaks {len(T):,} in {T.gene.nunique():,} genes")
print(f"  canonical                                  {int(T.is_canonical.sum()):>7,}")
print(f"  competitors, unfiltered                    {int(comp.sum()):>7,}")
print(f"  competitors, APA-annotated removed         {int((comp & T.flag_apa_annotated).sum()):>7,}")
print(f"  competitors, reference terminal exon rm    "
      f"{int((comp & T.flag_reference_terminal_exon & ~T.flag_apa_annotated).sum()):>7,}")
print(f"  competitors, filtered                      {int(keep.sum()):>7,}   expected 44,945"
      f"   {'OK' if int(keep.sum()) == 44945 else 'MISMATCH'}")
assert int(keep.sum()) == 44945

COLS = ["gene", "chrom", "center", "strand", "is_canonical"]
T.loc[T.is_canonical | comp, COLS].to_csv("ovl_peaks_unfiltered.tsv", sep="\t", index=False)
T.loc[T.is_canonical | keep, COLS].to_csv("ovl_peaks_filtered.tsv", sep="\t", index=False)
print(f"\nwrote ovl_peaks_unfiltered.tsv ({int((T.is_canonical | comp).sum()):,} rows) "
      f"and ovl_peaks_filtered.tsv ({int((T.is_canonical | keep).sum()):,} rows)")

# external sets
def read_lifted(path):
    rows = []
    with open(path, newline="\n") as fh:
        hdr = fh.readline().replace("\r", "").rstrip("\n").split("\t")
        ic, ip, ist = hdr.index("hg38_chrom"), hdr.index("hg38_pos"), hdr.index("strand")
        for line in fh:
            p = line.replace("\r", "").rstrip("\n").split("\t")
            if len(p) <= max(ic, ip, ist): continue
            c, s = p[ic], p[ist]
            if s not in "+-" or not p[ip]: continue
            rows.append((c if c.startswith("chr") else "chr" + c, int(p[ip]), s))
    d = pd.DataFrame(rows, columns=["chrom", "pos", "strand"])
    print(f"  {path}: {len(d):,} sites")
    return d

im = read_lifted("external/lifted/immune_hg38.tsv")
cl = read_lifted("external/lifted/cll_hg38.tsv")
im.to_csv("ovl_immune.tsv", sep="\t", index=False)
cl.to_csv("ovl_cll.tsv", sep="\t", index=False)

u = pd.concat([im, cl], ignore_index=True).sort_values(["chrom", "strand", "pos"])
out = []
for (c, s), sub in u.groupby(["chrom", "strand"], sort=False):
    last = None
    for p in sub["pos"].to_numpy():
        if last is None or p - last > DEDUP:
            out.append((c, p, s)); last = p
mg = pd.DataFrame(out, columns=["chrom", "pos", "strand"])
mg.to_csv("ovl_atlas3seq.tsv", sep="\t", index=False)
print(f"\n3'-seq atlas set: immune {len(im):,} + cll {len(cl):,} = {len(im)+len(cl):,} raw, "
      f"{len(mg):,} after merging sites within {DEDUP} nt on the same strand "
      f"({len(im)+len(cl)-len(mg):,} collapsed)")

# the check
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
            rd = csv.DictReader(fh, delimiter="\t")
            for p in rd:
                q = p["PAS_ID"].split(":")
                if len(q) != 3: continue
                c = q[0] if q[0].startswith("chr") else "chr" + q[0]
                recs[(c, q[1])].append(int(q[2]))
    for k in recs: recs[k] = np.sort(np.asarray(recs[k]))
    print(f"  {path}: {sum(len(v) for v in recs.values()):,} sites")
    return recs

def nearest(chrom, pos, strand, recs):
    d = np.full(len(pos), np.inf)
    key = pd.DataFrame({"c": chrom, "p": pos, "s": strand})
    for (c, s), sub in key.groupby(["c", "s"], sort=False):
        arr = recs.get((c, s))
        if arr is None or arr.size == 0: continue
        p = sub["p"].to_numpy()
        i = np.searchsorted(arr, p)
        lo = np.where(i > 0, np.abs(p - arr[np.clip(i - 1, 0, arr.size - 1)]), np.inf)
        hi = np.where(i < arr.size, np.abs(arr[np.clip(i, 0, arr.size - 1)] - p), np.inf)
        d[sub.index.to_numpy()] = np.minimum(lo, hi)
    return d

print("\n===== FILTER CHECK: is the filtered set really absent from both databases =====")
F = T.loc[keep, ["chrom", "center", "strand"]].reset_index(drop=True)
rows = []
bad_total = 0
for lab, path, kind in [("PolyASite 2.0", ATLAS, "atlas"), ("PolyA_DB v4", POLYADB, "db")]:
    R = load_sites(path, kind)
    d = nearest(F.chrom.to_numpy(), F.center.to_numpy(), F.strand.to_numpy(), R)
    bad = int((d <= WIN).sum()); bad_total += bad
    fin = d[np.isfinite(d)]
    print(f"  {lab:<14} within {WIN} nt: {bad:,} of {len(F):,}   "
          f"min distance {fin.min():,.0f} nt   median {np.median(fin):,.0f} nt   "
          f"{'OK' if bad == 0 else 'FILTER AND OVERLAP DISAGREE'}")
    rows.append(dict(db=lab, window_nt=WIN, n_filtered=len(F), n_within_window=bad,
                     min_distance=float(fin.min()), median_distance=float(np.median(fin))))
pd.DataFrame(rows).to_csv("ovl_filter_check.tsv", sep="\t", index=False)
print(f"\nCHECK {'PASS' if bad_total == 0 else 'FAIL'}: recomputed from the source annotation "
      f"files, not from the cached distances the filter used.")
assert bad_total == 0, "filtered peaks survive inside the filter's own window"
log("DONE_OVERLAP_PREP")
