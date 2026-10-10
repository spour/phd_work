#!/usr/bin/env python3
# add the u1 licensing columns to an existing *_peak_features.tsv without rescanning
import argparse, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cube", required=True)
ap.add_argument("--cube_meta", required=True)
ap.add_argument("--peak_features", required=True)
ap.add_argument("--out", default=None, help="default: overwrite --peak_features in place")
ap.add_argument("--scripts_dir", default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--force", action="store_true", help="write even if the alignment check fails")
a = ap.parse_args()

sys.path.insert(0, a.scripts_dir)
from figure3_data import u1_winner_features

z = np.load(a.cube)
cube = z["cube"].astype(np.float64)
edges = z["edges"]; thrs = z["thrs"]
CM = pd.read_csv(a.cube_meta, sep="\t")
PF = pd.read_csv(a.peak_features, sep="\t")
print(f"[in] cube {cube.shape} ; cube_meta {CM.shape} ; peak_features {PF.shape}", file=sys.stderr)

if len(PF) != len(CM) or len(PF) != cube.shape[0]:
    print(f"[FATAL] row counts differ: peak_features {len(PF)}, cube_meta {len(CM)}, "
          f"cube {cube.shape[0]}. These are not the same run.", file=sys.stderr)
    sys.exit(1)

# alignment check on every column the two tables share
shared = [c for c in ("gene", "is_canonical", "acc", "edef_best5", "nearest_dn", "pas")
          if c in PF.columns and c in CM.columns]
bad = []
for c in shared:
    x, y = PF[c], CM[c]
    if x.dtype.kind in "fc" or y.dtype.kind in "fc":
        agree = np.isclose(pd.to_numeric(x, errors="coerce").fillna(-9e9),
                           pd.to_numeric(y, errors="coerce").fillna(-9e9), atol=1e-6).mean()
    else:
        agree = (x.fillna("NA").astype(str) == y.fillna("NA").astype(str)).mean()
    print(f"    {c:14s} row-for-row agreement = {agree:.6f}", file=sys.stderr)
    if agree < 0.9999: bad.append(c)
if bad and not a.force:
    print(f"[FATAL] misaligned on {bad}. Refusing to write. Rerun the scan so the cube and "
          f"the feature table come from one run, or pass --force if you know why.", file=sys.stderr)
    sys.exit(1)

feats = u1_winner_features(cube, edges, thrs)
for k, v in feats.items():
    PF[k] = v
out = a.out or a.peak_features
PF.to_csv(out, sep="\t", index=False)
print(f"[out] wrote {out} with {len(PF):,} rows and new columns: {', '.join(feats)}", file=sys.stderr)
for k, v in feats.items():
    fin = np.isfinite(v)
    print(f"    {k:16s} finite {fin.mean():6.1%}  median {np.median(v[fin]):8.3f}", file=sys.stderr)
