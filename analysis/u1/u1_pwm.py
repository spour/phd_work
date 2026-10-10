#!/usr/bin/env python3
# the 5' splice site as a position weight matrix
import sys, argparse
import numpy as np
sys.path.insert(0, "scripts")
from cnn_interp_lib import NT
import u1_5ss_landscape_data as L

P = argparse.ArgumentParser()
P.add_argument("--thresh", type=float, default=2.0)
A = P.parse_args()
def log(*a): print(*a, file=sys.stderr, flush=True)

U1 = np.asarray(L.U1, float)
POW = np.array([int(L.POW[p]) for p in range(9)])
best = float(np.nanmax(U1))
keep = np.flatnonzero(np.isfinite(U1) & (U1 >= best - A.thresh))
log(f"[u1] optimum -dG {best:.2f}; {len(keep):,} of {np.isfinite(U1).sum():,} 9-mers lie within "
    f"{A.thresh} kcal/mol of it ({100*len(keep)/np.isfinite(U1).sum():.2f}%)")
B = (keep[:, None] // POW[None, :]) % 4
W = U1[keep] - (best - A.thresh)
PWM = np.stack([np.array([W[B[:, p] == i].sum() for p in range(9)]) for i in range(4)])
PWM /= PWM.sum(0, keepdims=True)
IC = 2.0 + np.where(PWM > 0, PWM * np.log2(np.clip(PWM, 1e-12, None)), 0).sum(0)
cons = "".join(NT[i] for i in PWM.argmax(0))
log(f"[u1] consensus {cons}; per-column information content "
    + " ".join(f"{v:.2f}" for v in IC))
log(f"[u1] the invariant GT sits at intron positions 1-2, columns 4 and 5 here, at "
    f"{IC[3]:.2f} and {IC[4]:.2f} bits of 2")
with open("u1_pwm.tsv", "w") as fh:
    fh.write("offset\tA\tC\tG\tT\tic\n")
    for p in range(9):
        fh.write(f"{p-3}\t" + "\t".join(f"{PWM[i,p]:.4f}" for i in range(4)) + f"\t{IC[p]:.4f}\n")
log("[out] u1_pwm.tsv")
