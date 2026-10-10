#!/usr/bin/env python3
# optional re-analysis helper
import argparse, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--denswin", required=True, help="{out}_denswin.npz from figure3_data.py")
ap.add_argument("--auroc", default="fig3_auroc.tsv", help="{out}_auroc.tsv, for the acceptor curve")
ap.add_argument("--out", default="fig3")
ap.add_argument("--donor_search", default="3,4,6.5,8")
ap.add_argument("--u1_thr", type=float, default=4.0)
ap.add_argument("--nybin", type=int, default=14)
ap.add_argument("--scripts_dir", default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args()

sys.path.insert(0, a.scripts_dir)
from figure3_data import bcshow_tables

z = np.load(a.denswin)
dw = {m: z[f"dw_{m}"] for m in ("u1", "donor") if f"dw_{m}" in z}
thrs = {m: z[f"thr_{m}"] for m in dw}
acc_rows = pd.read_csv(a.auroc, sep="\t").to_dict("records")
print(f"[data] {len(z['is_canonical']):,} peaks ; signals {list(dw)}", file=sys.stderr)

AU, RT, picks = bcshow_tables(dw, thrs, z["wins"], z["is_canonical"], acc_rows,
                              u1_thr=a.u1_thr,
                              donor_search=tuple(float(x) for x in a.donor_search.split(",")),
                              nybin=a.nybin)
AU.to_csv(f"{a.out}_bcshow_auroc.tsv", sep="\t", index=False)
RT.to_csv(f"{a.out}_bcshow_ratio.tsv", sep="\t", index=False)
print("\n=== PEAK AUROC per signal (oriented above 0.5) ===")
for m in ("acceptor", "u1", "donor"):
    g = AU[AU.mech == m]
    if not len(g): continue
    r = g.loc[g.auroc.idxmax()]
    print(f"  {m:9s} {r.auroc:.4f}  at {r.window:.0f}nt   raw {r.auroc_raw:.4f}  "
          f"predicts {r.predicts:10s}  [{r.encoding}]")
print(f"\n[out] {a.out}_bcshow_auroc.tsv ; {a.out}_bcshow_ratio.tsv", file=sys.stderr)
