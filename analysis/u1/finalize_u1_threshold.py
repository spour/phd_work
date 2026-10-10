#!/usr/bin/env python3
# post-process the u1 calibration (no rescan): pick the fdr-minimizing threshold
import numpy as np, csv

TOTAL_OBS = 140_880_198
NP = 70_723
OLD = 4.0

rows = []
with open("u1_threshold.tsv") as fh:
    for line in fh:
        if line.startswith("#") or not line.strip() or line.startswith("threshold"): continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 7: continue
        rows.append([float(x) for x in p])
A = np.array(rows)
thr, z, unif, obs, nul, fdr_a, fdr_b = A.T

mask = (thr >= 3.0) & (thr <= 9.0)
jmin = np.where(mask)[0][np.argmin(fdr_a[mask])]
NEW = float(thr[jmin])

ctr, obs_dens = [], []
with open("u1_score_hist.tsv") as fh:
    r = csv.DictReader(fh, delimiter="\t")
    for row in r:
        ctr.append(float(row["score"])); obs_dens.append(float(row["observed"]))
ctr = np.array(ctr); obs_dens = np.array(obs_dens)
def hits(t): return int(round(obs_dens[ctr >= t].sum() * TOTAL_OBS))
h_old, h_new = hits(OLD), hits(NEW)

print("=== U1 threshold: honest summary ===")
print(f"no threshold in [3,9] reaches FDR<=0.10 under the dinucleotide null")
print(f"minimum dinuc-FDR = {fdr_a[jmin]:.3f} at threshold {NEW:.2f} (z={z[jmin]:.2f}, "
      f"top {100*unif[jmin]:.2f}% of 9-mers)")
print(f"OLD 4.0 : dinuc-FDR {fdr_a[np.argmin(np.abs(thr-OLD))]:.3f} ; {h_old:,} hits ({h_old/NP:.1f}/peak)")
print(f"NEW {NEW:.1f} : dinuc-FDR {fdr_a[jmin]:.3f} ; {h_new:,} hits ({h_new/NP:.1f}/peak)")
print(f"fold reduction {h_old/max(h_new,1):.1f}x")

keep = []
with open("u1_threshold.tsv") as fh:
    for line in fh:
        if line.startswith("#") or (not line.strip()): break
        keep.append(line)
with open("u1_threshold.tsv", "w") as fh:
    fh.writelines(keep)
    fh.write("\n# ---- calibration verdict (dinucleotide-shuffle null) ----\n")
    fh.write("# no threshold in [3,9] reaches FDR<=0.10; U1 9-mer score is composition-dominated\n")
    for q in (0.10, 0.05, 0.01):
        fh.write(f"# FDR<={q:.2f}\tthr=NONE\n")
    fh.write(f"# FDR_min\tthr={NEW:.2f}\tfdr={fdr_a[jmin]:.3f}\tz={z[jmin]:.2f}\ttail_frac={unif[jmin]:.5f}\n")
    fh.write(f"# OLD_THR={OLD}\tfdr={fdr_a[np.argmin(np.abs(thr-OLD))]:.3f}\thits={h_old}\tper_peak={h_old/NP:.2f}\n")
    fh.write(f"# NEW_THR={NEW:.2f}\thits={h_new}\tper_peak={h_new/NP:.2f}\tfold_reduction={h_old/max(h_new,1):.2f}\n")
print("\n[out] rewrote chosen block in u1_threshold.tsv")
