#!/usr/bin/env python3
# which half of gc_contrast carries the class difference, and where the classes cross
import csv, sys, numpy as np, pandas as pd
def log(*a): print(*a, file=sys.stderr, flush=True)
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
y = D.is_canonical.to_numpy(); ROW = np.load("upstream_rowmask.npz")["ROW"]
Z = np.load("control_contrast_scan.npz", allow_pickle=True); S = Z["X"]; SN = list(Z["names"])
gd = S[:, SN.index("gc_distal")]; gp = S[:, SN.index("gc_proximal")]
I = np.where(ROW)[0]; yv = y[I]
rows = []
for nm, v in [("distal pA-2000..-438", gd), ("proximal pA-437..-126", gp),
              ("contrast, distal minus proximal", gd - gp)]:
    c = float(np.nanmean(v[I][yv == 1])); k = float(np.nanmean(v[I][yv == 0]))
    rows.append((nm, c, k, c - k))
P = pd.read_csv("gc_profile.tsv", sep="\t").pivot(index="position", columns="class", values="value")
d = (P["canonical"] - P["competitor"]).sort_index()
up = d[d.index < 0]
sign = np.sign(up.values); cross = None
for i in range(1, len(sign)):
    if sign[i] != sign[i - 1]:
        x0, x1 = up.index[i - 1], up.index[i]; y0, y1 = up.values[i - 1], up.values[i]
        cross = float(x0 - y0 * (x1 - x0) / (y1 - y0))
print(f"{'window':<34}{'canonical':>12}{'competitor':>12}{'difference':>12}")
with open("gc_windows.tsv", "w") as fh:
    fh.write("quantity\tcanonical\tcompetitor\tdifference\n")
    for nm, c, k, df in rows:
        print(f"{nm:<34}{c:>12.4f}{k:>12.4f}{df:>+12.4f}")
        fh.write(f"{nm}\t{c:.5f}\t{k:.5f}\t{df:.5f}\n")
    frac = (rows[0][3]) / rows[2][3]
    print(f"\ndistal window supplies {100*frac:.0f}% of the class difference in the contrast")
    print(f"metagene crossover upstream of the pA: {cross:.0f} nt")
    fh.write(f"#distal_share\t{frac:.5f}\t\t\n#crossover\t{cross:.1f}\t\t\n")
log("DONE_GC_WINDOWS")
