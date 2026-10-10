#!/usr/bin/env python3
# measure the 3' splice site as its three parts, not as one 23-mer
import sys, argparse
import numpy as np
from multiprocessing import Pool

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--procs", type=int, default=6)
A = P.parse_args()
UP, WIN = 2000, 2500
from cnn_interp_lib import NT

BPC = [dict(C=.40, T=.40, A=.10, G=.10),
       dict(A=.25, C=.25, G=.25, T=.25),
       dict(C=.40, T=.40, A=.10, G=.10),
       dict(T=.70, C=.15, A=.075, G=.075),
       dict(A=.40, G=.40, C=.10, T=.10),
       dict(A=.85, C=.05, G=.05, T=.05),
       dict(C=.40, T=.40, A=.10, G=.10)]
BPM = np.array([[np.log2(c[b]/0.25) for b in "ACGT"] for c in BPC])
IDX = {b: i for i, b in enumerate("ACGT")}
def bp_score(s):
    best = (-99.0, -1)
    for i in range(len(s)-6):
        v = 0.0; ok = True
        for j in range(7):
            k = IDX.get(s[i+j])
            if k is None: ok = False; break
            v += BPM[j, k]
        if ok and v > best[0]: best = (v, i+5)
    return best

_M3 = None
def _init():
    global _M3
    from maxentpy.maxent import load_matrix3
    _M3 = load_matrix3()
def features(row):
    from maxentpy import maxent
    w = "".join(NT[b] if b < 4 else "N" for b in row)
    best = (-99.0, -1)
    for p in range(0, UP-22):
        if w[p+18] == "A" and w[p+19] == "G":
            try: v = float(maxent.score3(w[p:p+23], matrix=_M3))
            except Exception: continue
            if v > best[0]: best = (v, p)
    if best[1] < 0: return (np.nan,)*5
    ag = best[1] + 18
    ppt = w[max(0, ag-20):ag]
    py = sum(c in "CT" for c in ppt)/max(len(ppt), 1)
    run = 0; cur = 0
    for c in ppt:
        cur = cur+1 if c in "CT" else 0
        run = max(run, cur)
    bpw = w[max(0, ag-40):max(0, ag-17)]
    b, _ = bp_score(bpw)
    return best[0], b, py, float(run), float(UP-ag)

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; split = D["split"][ok]
log(f"[data] {len(X):,} peaks; scoring the three parts at the best AG in the 2 kb upstream")
with Pool(A.procs, initializer=_init) as pool:
    R = pool.map(features, list(X), chunksize=64)
R = np.array(R, float)
AG, BP, PY, RUN, DIST = (R[:, i] for i in range(5))
log(f"[parts] at the anchoring AG:")
for nm, v in (("AG, score3", AG), ("branch point, YNYURAY log-odds", BP),
              ("polypyrimidine fraction", PY), ("longest pyrimidine run", RUN),
              ("distance upstream (nt)", DIST)):
    log(f"   {nm:<32} median {np.nanmedian(v):>8.3f}   real 3' ends {np.nanmedian(v[y==1]):>8.3f}"
        f"   competitors {np.nanmedian(v[y==0]):>8.3f}")
def auroc(s, yy):
    m = np.isfinite(s); s, yy = s[m], yy[m]
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); r[o] = np.arange(1, len(s)+1)
    ss = s[o]; i = 0
    while i < len(ss):
        j = i
        while j+1 < len(ss) and ss[j+1] == ss[i]: j += 1
        if j > i: r[o[i:j+1]] = (i+1+j+1)/2
        i = j+1
    n1 = int((yy == 1).sum()); n0 = len(yy)-n1
    return (r[yy == 1].sum()-n1*(n1+1)/2)/(n1*n0)
log(f"\n[discrimination] each part alone, real 3' ends against competitors")
for nm, v in (("AG, score3", AG), ("branch point", BP), ("pyrimidine fraction", PY),
              ("pyrimidine run", RUN)):
    log(f"   {nm:<24} AUROC {auroc(v, y):.4f}")
C = np.column_stack([AG, BP, PY, RUN])
m = np.isfinite(C).all(1)
log(f"\n[correlation between the parts, n={m.sum():,}]")
nm = ["AG", "BP", "PYfrac", "PYrun"]
for i in range(4):
    log("   " + f"{nm[i]:<7}" + " ".join(f"{np.corrcoef(C[m,i],C[m,j])[0,1]:+7.3f}" for j in range(4)))
tr = split == "train"
def z(v):
    mu, sd = np.nanmean(v[tr]), np.nanstd(v[tr])
    return np.nan_to_num((v-mu)/sd, nan=0.0).astype(np.float32)
np.savez_compressed("acceptor3_scalar.npz", ag=z(AG), bp=z(BP), ppt=z(PY), run=z(RUN),
                    dist=z(DIST), raw_ag=AG, raw_bp=BP, raw_ppt=PY, raw_run=RUN, raw_dist=DIST)
log("\n[out] acceptor3_scalar.npz")
