#!/usr/bin/env python3
# build the encoding-based dot-heatmap cells for figure 3
import argparse, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cube", default="fig3_ucube.npz")
ap.add_argument("--peak_features", default="fig3_peak_features.tsv")
ap.add_argument("--out", default="fig3_encheat.tsv")
ap.add_argument("--xmax", type=int, default=1000, help="rightmost position bin edge to emit")
a = ap.parse_args()

z = np.load(a.cube)
EDGES = z["edges"]
PF = pd.read_csv(a.peak_features, sep="\t")
if len(PF) != z["cube"].shape[0]:
    print(f"[FATAL] peak_features {len(PF)} rows vs cube {z['cube'].shape[0]}", file=sys.stderr)
    sys.exit(1)
y = PF.is_canonical.values.astype(int)
print(f"[data] {len(PF):,} peaks ; {int((y==1).sum()):,} canonical / {int((y==0).sum()):,} competitor",
      file=sys.stderr)

SPECS = [("u1",  z["cube"],    z["thrs"],     4.0, "u1_dens_win"),
         ("me5", z["me5cube"], z["me5_thrs"], 8.0, "me5_dens_strong")]

ctr = (EDGES[:-1] + EDGES[1:]) / 2.0
keep_bin = ctr <= a.xmax
rows = []
for mech, cube, thrs, strong, _tag in SPECS:
    thrs = np.asarray(thrs, float)
    nb, nt = cube.shape[1], len(thrs)
    for ti in range(nt):
        band_lo = thrs[ti]
        band_hi = thrs[ti + 1] if ti + 1 < nt else np.inf
        cnt = cube[:, :, 3 + ti] - (cube[:, :, 3 + ti + 1] if ti + 1 < nt else 0.0)
        cnt = np.maximum(cnt, 0.0)
        for bi in range(nb):
            if not keep_bin[bi]: continue
            v = cnt[:, bi]
            mc = float(v[y == 1].mean()); mk = float(v[y == 0].mean())
            npk = int((v > 0).sum())
            if npk == 0: continue
            eps = 1.0 / max((y == 1).sum(), (y == 0).sum(), 1)
            rows.append(dict(mech=mech, pos=float(ctr[bi]),
                             score=float(band_lo), score_hi=float(band_hi),
                             log2ratio=float(np.log2((mc + eps) / (mk + eps))),
                             n=npk, canon_mean=mc, comp_mean=mk, strong_thr=float(strong)))
R = pd.DataFrame(rows)
R.to_csv(a.out, sep="\t", index=False)
print(f"[out] {a.out}: {len(R):,} cells ({R.mech.value_counts().to_dict()})", file=sys.stderr)
for m, g in R.groupby("mech"):
    s = g[g.score >= g.strong_thr.iloc[0]]
    print(f"    {m}: strong-band cells {len(s)}, median log2ratio {s.log2ratio.median():+.3f} "
          f"(negative = competitor-enriched)", file=sys.stderr)
