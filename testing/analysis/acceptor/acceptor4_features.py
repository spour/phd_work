#!/usr/bin/env python3
# an acceptor target that is a maximum, so it is compatible with a max-pooling network
import sys, argparse
import numpy as np
from multiprocessing import Pool
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--procs", type=int, default=6)
A = P.parse_args()
UP = 2000
from cnn_interp_lib import NT
BPC = [dict(C=.40,T=.40,A=.10,G=.10), dict(A=.25,C=.25,G=.25,T=.25),
       dict(C=.40,T=.40,A=.10,G=.10), dict(T=.70,C=.15,A=.075,G=.075),
       dict(A=.40,G=.40,C=.10,T=.10), dict(A=.85,C=.05,G=.05,T=.05),
       dict(C=.40,T=.40,A=.10,G=.10)]
BPM = np.array([[np.log2(c[b]/0.25) for b in "ACGT"] for c in BPC])
IDX = {b:i for i,b in enumerate("ACGT")}
_M3 = None
def _init():
    global _M3
    from maxentpy.maxent import load_matrix3
    _M3 = load_matrix3()
def scan(row):
    from maxentpy import maxent
    w = "".join(NT[b] if b<4 else "N" for b in row)
    ags, s3s, bps, pys = [], [], [], []
    for p in range(0, UP-22):
        if w[p+18]=="A" and w[p+19]=="G":
            try: v = float(maxent.score3(w[p:p+23], matrix=_M3))
            except Exception: continue
            ag = p+18
            bw = w[max(0,ag-40):max(0,ag-17)]
            b = -99.0
            for i in range(max(0,len(bw)-6)):
                t = 0.0; okk = True
                for j in range(7):
                    k = IDX.get(bw[i+j])
                    if k is None: okk=False; break
                    t += BPM[j,k]
                if okk and t>b: b=t
            tract = w[max(0,ag-20):ag]
            py = sum(c in "CT" for c in tract)/max(len(tract),1)
            ags.append(ag); s3s.append(v); bps.append(b); pys.append(py)
    if not ags: return None
    return (np.array(ags,np.int32), np.array(s3s,np.float32),
            np.array(bps,np.float32), np.array(pys,np.float32))

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; split = D["split"][ok]
log(f"[data] {len(X):,} peaks; scanning every AG in the 2 kb upstream")
with Pool(A.procs, initializer=_init) as pool:
    S = pool.map(scan, list(X), chunksize=64)
n_ag = np.array([0 if s is None else len(s[0]) for s in S])
log(f"[scan] AGs per window: median {np.median(n_ag):.0f}, min {n_ag.min()}, max {n_ag.max()}")
tr = split == "train"
def pool_stat(f):
    return np.array([np.nan if s is None else f(s) for s in S], float)
mu = {k: np.nanmean(pool_stat(lambda s, i=i: s[i].mean())[tr]) for k,i in
      (("s3",1),("bp",2),("py",3))}
sd = {k: np.nanstd(pool_stat(lambda s, i=i: s[i].mean())[tr]) for k,i in
      (("s3",1),("bp",2),("py",3))}
log(f"[weights] standardising each part on the training split before combining: " +
    ", ".join(f"{k} mean {mu[k]:.3f} sd {sd[k]:.3f}" for k in mu))
def combine(s):
    if s is None: return (np.nan,)*5
    z3 = (s[1]-mu["s3"])/sd["s3"]; zb = (s[2]-mu["bp"])/sd["bp"]; zp = (s[3]-mu["py"])/sd["py"]
    tot = z3 + zb + zp
    k = int(np.argmax(tot))
    return float(tot[k]), float(s[1][k]), float(s[2][k]), float(s[3][k]), float(UP-s[0][k])
C = np.array([combine(s) for s in S], float)
TOT, AG, BP, PY, DIST = (C[:,i] for i in range(5))
def auroc(v, yy):
    m=np.isfinite(v); v,yy=v[m],yy[m]
    o=np.argsort(v,kind="mergesort"); r=np.empty(len(v),float); r[o]=np.arange(1,len(v)+1)
    ss=v[o]; i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2
        i=j+1
    n1=int((yy==1).sum()); n0=len(yy)-n1
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
log(f"\n[target] the combined acceptor score, maximised over every AG in the window")
for nm,v in (("combined (the target)",TOT),("its score3 component",AG),
             ("its branch-point component",BP),("its pyrimidine fraction",PY)):
    log(f"   {nm:<28} AUROC {auroc(v,y):.4f}   median real {np.nanmedian(v[y==1]):>7.3f}"
        f"   competitors {np.nanmedian(v[y==0]):>7.3f}")
log(f"   (for reference, best-score3 alone scored AUROC 0.6177)")
def z(v):
    m,s2 = np.nanmean(v[tr]), np.nanstd(v[tr]); return np.nan_to_num((v-m)/s2, nan=0.).astype(np.float32)
np.savez_compressed("acceptor4_scalar.npz", tot=z(TOT), ag=z(AG), bp=z(BP), py=z(PY),
                    raw_tot=TOT, raw_ag=AG, raw_bp=BP, raw_py=PY, raw_dist=DIST)
log("\n[out] acceptor4_scalar.npz")
