#!/usr/bin/env python3
# the 3' splice site as a position weight matrix
import sys, argparse
import numpy as np
sys.path.insert(0, "scripts")
from cnn_interp_lib import NT
from maxentpy import maxent
from maxentpy.maxent import load_matrix3

P = argparse.ArgumentParser()
P.add_argument("--thresh", type=float, default=2.0)
P.add_argument("--temp", type=float, default=1.0)
P.add_argument("--chains", type=int, default=192)
P.add_argument("--steps", type=int, default=5000)
P.add_argument("--burn", type=int, default=800)
P.add_argument("--pin_ag", action="store_true")
P.add_argument("--no_c", action="store_true")
P.add_argument("--out", default="me3_pwm.tsv")
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
def log(*a): print(*a, file=sys.stderr, flush=True)
M3 = load_matrix3()

def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0

ALPHA = "AGT" if A.no_c else "ACGT"
OPT = "TTTTTTTTTGTGTGTTTTAGGTT" if A.no_c else "TTTCCTCTCTCTCCCCGCAGGTG"
assert abs(s3(OPT) - (14.38 if A.no_c else 17.172)) < 0.05, s3(OPT)
thr = s3(OPT) - A.thresh
log(f"[me3] optimum {OPT} at score3 {s3(OPT):.2f}; keeping samples at or above {thr:.2f}")
log(f"[me3] the donor's matrix enumerates 4^9 = 262,144 9-mers; this register is 4^21 = "
    f"{4**21:,} sequences with AG pinned, so it is SAMPLED, not enumerated")

rng = np.random.default_rng(A.seed)
PIN = {18: "A", 19: "G"} if A.pin_ag else {}
free = [i for i in range(23) if i not in PIN]
keep = {}
acc_tot = 0
for c in range(A.chains):
    s = list("".join(rng.choice(list(ALPHA), 23)))
    for i, b in PIN.items(): s[i] = b
    cur = s3("".join(s))
    for t in range(A.steps):
        i = free[rng.integers(len(free))]; old = s[i]
        s[i] = ALPHA[rng.integers(len(ALPHA))]
        new = s3("".join(s))
        if new >= cur or rng.random() < np.exp((new - cur) / A.temp):
            cur = new; acc_tot += 1
        else:
            s[i] = old
        if t >= A.burn and cur >= thr:
            keep[("".join(s))] = cur
log(f"[me3] {A.chains} chains x {A.steps} steps, acceptance {acc_tot/(A.chains*A.steps):.2f}; "
    f"{len(keep):,} distinct sequences at or above {thr:.2f}")
S = np.array([[NT.index(c) for c in k] for k in keep])
W = np.array([v for v in keep.values()]) - thr
PWM = np.stack([np.array([W[S[:, p] == i].sum() for p in range(23)]) for i in range(4)])
PWM /= PWM.sum(0, keepdims=True)
IC = 2.0 + np.where(PWM > 0, PWM * np.log2(np.clip(PWM, 1e-12, None)), 0).sum(0)
cons = "".join(NT[i] for i in PWM.argmax(0))
log(f"[me3] consensus {cons}")
log(f"[me3] information content per column " + " ".join(f"{v:.2f}" for v in IC))
log(f"[me3] AG {'pinned' if A.pin_ag else 'NOT pinned, so what appears at 19 and 20 was found'}: "
    f"column 19 is {cons[18]} at {IC[18]:.2f} bits, column 20 is {cons[19]} at {IC[19]:.2f}; "
    f"mean over all 23 columns {IC.mean():.2f}")
log(f"[me3] AG is the consensus at 19-20 in {100*np.mean((S[:,18]==0)&(S[:,19]==2)):.0f}% of "
    f"the sampled ensemble")
with open(A.out, "w") as fh:
    fh.write("offset\tA\tC\tG\tT\tic\n")
    for p in range(23):
        fh.write(f"{p-18}\t" + "\t".join(f"{PWM[i,p]:.4f}" for i in range(4)) + f"\t{IC[p]:.4f}\n")
log(f"[out] {A.out}")
