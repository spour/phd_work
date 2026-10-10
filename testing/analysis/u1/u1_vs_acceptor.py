#!/usr/bin/env python3
# item 3 (analysis, no figure): among competitors only
import numpy as np, pandas as pd, math

z = np.load("fig3_ucube.npz")
cube, edges, thrs = z["cube"], z["edges"], z["thrs"]
M = pd.read_csv("fig3_ucube_meta.tsv", sep="\t")
assert len(M) == cube.shape[0]

b = lambda x: int(np.searchsorted(edges, x))
lo, hi = b(-500), b(0)
kb = (0 - (-500)) / 1000.0
def dens(t):
    ti = 3 + int(np.argmin(np.abs(thrs - t)))
    return cube[:, lo:hi, ti].sum(1) / kb
mean_u1 = cube[:, lo:hi, 1].sum(1) / np.maximum(cube[:, lo:hi, 0].sum(1), 1)

acc = M["acc"].values.astype(float)
comp = (M["is_canonical"].values == 0)

def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    r = np.corrcoef(rx, ry)[0, 1]
    z_ = r * math.sqrt(n - 1)
    p = math.erfc(abs(z_) / math.sqrt(2))
    return r, p, n

print("=== Item 3: competitor U1 (PAS-proximal, [-500,0]) vs anchor acceptor ===")
for name, v in [("U1 density/kb @4.0 (old)", dens(4.0)),
                ("U1 density/kb @7.0 (FDR-min)", dens(7.0)),
                ("mean U1 score (threshold-free)", mean_u1)]:
    r, p, n = spearman(v[comp], acc[comp])
    print(f"  {name:32s}  Spearman r = {r:+.4f}  p = {p:.2e}  (n = {n:,} competitors)")
print("\ninterpretation: positive r = U1 occupancy is higher at strong acceptors "
      "(the mechanistic expectation).")
