#!/usr/bin/env python3
# corroborate canonical vs competitor peaks against polyasite 2.0 and polya_db v4
import csv, numpy as np
R = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
isc = np.array([int(float(r["is_canonical"])) for r in R])
NP = len(R)
PA = np.load("atlas_peak_cache.npz")
DB = np.load("polyadb_peak_cache.npz")

CLASSES = [("canonical", isc == 1), ("competitor", isc == 0)]
WIN_HALF = 250
SWEEP = [25, 50, 100, 250, 500]

# overlap sweep
with open("polya_corroboration_overlap.tsv", "w") as fh:
    fh.write("db\tclass\thalf_window_nt\tfull_window_nt\tn_total\tn_matched\tfrac_matched\n")
    for dbnm, D in [("PolyASite", PA["dist"]), ("PolyA_DB", DB["dist"])]:
        for cnm, m in CLASSES:
            nt = int(m.sum())
            for hw in SWEEP:
                nm = int((D[m] <= hw).sum())
                fh.write(f"{dbnm}\t{cnm}\t{hw}\t{2*hw}\t{nt}\t{nm}\t{nm/nt:.4f}\n")
print("overlap (fraction matched within +/-250 nt = 500-nt window):")
for dbnm, D in [("PolyASite", PA["dist"]), ("PolyA_DB", DB["dist"])]:
    for cnm, m in CLASSES:
        fr = float((D[m] <= WIN_HALF).mean())
        print(f"  {dbnm:10s} {cnm:11s} n={int(m.sum()):6d}  matched@500nt {fr:.4f}")

# strength distributions (matched inside the 500-nt window)
METRICS = [
    ("PolyASite", "log10_TPM",          np.log10(np.clip(PA["tpm"], 1e-3, None)), PA["dist"]),
    ("PolyASite", "frac_samples",       PA["frac"],                               PA["dist"]),
    ("PolyASite", "n_protocols",        PA["nprot"].astype(float),                PA["dist"]),
    ("PolyA_DB",  "PolyaStrength",      DB["strength"],                           DB["dist"]),
    ("PolyA_DB",  "strength_pctile",    DB["pctile"],                             DB["dist"]),
    ("PolyA_DB",  "log10_AvgRPM",       np.log10(np.clip(DB["rpm"], 1e-3, None)), DB["dist"]),
    ("PolyA_DB",  "PSE_3reads",         DB["pse"],                                DB["dist"]),
]
with open("polya_corroboration_strength.tsv", "w") as fh:
    fh.write("db\tmetric\tclass\tvalue\n")
    for dbnm, lab, vals, D in METRICS:
        for cnm, m in CLASSES:
            sel = m & (D <= WIN_HALF) & np.isfinite(vals)
            for v in vals[sel]:
                fh.write(f"{dbnm}\t{lab}\t{cnm}\t{v:.5f}\n")

# console summary of strength (median / iqr)
print("\nstrength of matched site (within 500-nt window)  median [Q1,Q3]  (n):")
for dbnm, lab, vals, D in METRICS:
    for cnm, m in CLASSES:
        sel = m & (D <= WIN_HALF) & np.isfinite(vals)
        v = vals[sel]
        if len(v):
            q1, md, q3 = np.percentile(v, [25, 50, 75])
            print(f"  {dbnm:10s} {lab:16s} {cnm:11s} {md:8.3f} [{q1:7.3f},{q3:7.3f}]  (n={len(v)})")
print("\nwrote polya_corroboration_overlap.tsv, polya_corroboration_strength.tsv")
