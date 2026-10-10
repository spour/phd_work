#!/usr/bin/env python3
# score the acceptor as a t-rich tract with an ag, not as maxentscan score3
import sys, argparse, math
import numpy as np
from multiprocessing import Pool
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--procs", type=int, default=6)
A = P.parse_args()
UP = 2000
from cnn_interp_lib import NT
_M3 = None
def _init():
    global _M3
    from maxentpy.maxent import load_matrix3
    _M3 = load_matrix3()
def scan(row):
    from maxentpy import maxent
    w = "".join(NT[b] if b < 4 else "N" for b in row)
    bt = bf = br = bs = -9e9
    for p in range(0, UP-22):
        if w[p+18] == "A" and w[p+19] == "G":
            ag = p+18
            t = w[max(0, ag-20):ag]
            if not t: continue
            tf = t.count("T")/len(t)
            pf = sum(c in "CT" for c in t)/len(t)
            run = 0; cur = 0
            for c in t:
                cur = cur+1 if c == "T" else 0
                run = max(run, cur)
            try: s3 = float(maxent.score3(w[p:p+23], matrix=_M3))
            except Exception: s3 = -99.0
            bf = max(bf, tf); br = max(br, run); bt = max(bt, pf)
            bs = max(bs, s3 + 4.0*(tf-pf))
    if bf < -1e8: return (np.nan,)*4
    return bf, float(br), bt, bs
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]
log(f"[data] {len(X):,} peaks; scanning every AG for T content of its tract")
with Pool(A.procs, initializer=_init) as pool:
    R = np.array(pool.map(scan, list(X), chunksize=64), float)
TF, TR, PF, TS = (R[:, i] for i in range(4))
S3 = np.load("acceptor3_scalar.npz")["raw_ag"]
g_, c_, x_ = D["gene"][ok], D["chrom"][ok], D["center"][ok]
Z = np.load("cnn_wg_scores.npz", allow_pickle=True)
zk = {(a,b,int(v)): i for i,(a,b,v) in enumerate(zip(Z["gene"],Z["chrom"],Z["center"]))}
L = np.array([Z["logit"][zk[(a,b,int(v))]] if (a,b,int(v)) in zk else np.nan
              for a,b,v in zip(g_,c_,x_)])
te = np.flatnonzero((split == "test") & np.isfinite(L))
va = np.flatnonzero((split == "val") & np.isfinite(L))
def auroc(v, yy):
    m = np.isfinite(v); v, yy = v[m], yy[m]
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), float); r[o] = np.arange(1, len(v)+1)
    s = v[o]; i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[j+1] == s[i]: j += 1
        if j > i: r[o[i:j+1]] = (i+1+j+1)/2
        i = j+1
    n1 = int((yy == 1).sum()); n0 = len(yy)-n1
    return (r[yy == 1].sum()-n1*(n1+1)/2)/(n1*n0)
def wgs(s, ii):
    G = gene[ii]; Y = y[ii]; by = {}
    for k in range(len(ii)):
        r, c = by.setdefault(G[k], ([], [])); (r if Y[k] == 1 else c).append(s[k])
    v = []
    for r, c in by.values():
        if not r or not c: continue
        r = np.array(r); c = np.array(c)
        v.append(((r[:,None]>c[None,:]).sum()+.5*(r[:,None]==c[None,:]).sum())/(len(r)*len(c)))
    return float(np.mean(v))
def pear(a, b):
    m = np.isfinite(a)&np.isfinite(b); a, b = a[m], b[m]; n = len(a)
    r = float(np.corrcoef(a,b)[0,1]); z = .5*math.log((1+r)/(1-r)); se = 1/math.sqrt(n-3)
    return r, math.tanh(z-1.96*se), math.tanh(z+1.96*se)
FE = {"score3 (what was used all along)": S3, "T fraction of the tract": TF,
      "longest T run": TR, "pyrimidine fraction": PF, "score3 re-weighted to T": TS}
log(f"\n1  DISCRIMINATION, real 3' ends against competitors, held-out test")
log(f"   {'feature':<34} {'pooled AUROC':>13} {'within-gene':>12}")
rows = []
for nm, v in FE.items():
    log(f"   {nm:<34} {auroc(v[te], y[te]):>13.4f} {wgs(v[te], te):>12.4f}")
    rows.append(("discrimination", nm, auroc(v[te], y[te]), wgs(v[te], te)))
log(f"\n2  DOES THE CNN'S SCORE TRACK IT?  (it is flat on score3: +0.026 within real 3' ends)")
c = y[te] == 1
log(f"   {'feature':<34} {'all peaks':>22} {'within real 3 ends':>26}")
for nm, v in FE.items():
    ra = pear(v[te], L[te]); rr = pear(v[te][c], L[te][c])
    log(f"   {nm:<34} {f'{ra[0]:+.4f} [{ra[1]:+.4f},{ra[2]:+.4f}]':>22} "
        f"{f'{rr[0]:+.4f} [{rr[1]:+.4f},{rr[2]:+.4f}]':>26}")
    rows.append(("cnn_correlation", nm, ra[0], rr[0]))
log(f"\n3  DOES IT ADD TO THE CNN?  weight chosen on validation, reported on held-out test")
def zin(v, ii):
    out = np.zeros(len(ii)); G = gene[ii]
    for gg in np.unique(G):
        m = G == gg; s = v[ii][m]
        out[m] = (s-s.mean())/(s.std()+1e-9) if len(s) > 1 else 0.
    return out
W = (0, .05, .1, .15, .2, .3, .4, .5, .75, 1.)
base = wgs(L[te], te); zl, zlv = zin(L, te), zin(L, va)
log(f"   CNN alone {base:.4f}")
log(f"   {'feature':<34} {'w on val':>9} {'TEST within-gene':>18} {'gain':>9} {'oracle best':>12}")
for nm, v in FE.items():
    zv, zvv = zin(v, te), zin(v, va)
    wv = max(((wgs(zlv+w*zvv, va), w) for w in W))[1]
    got = wgs(zl+wv*zv, te); orc = max(wgs(zl+w*zv, te) for w in W)
    log(f"   {nm:<34} {wv:>9.2f} {got:>18.4f} {got-base:>+9.4f} {orc:>12.4f}")
    rows.append(("combination", nm, got, got-base))
np.savez_compressed("acceptor_trich.npz", t_frac=TF, t_run=TR, py_frac=PF, t_score=TS, score3=S3)
with open("acceptor_trich_report.tsv","w") as fh:
    fh.write("section\tfeature\tv1\tv2\n")
    for r in rows: fh.write("\t".join(str(x) for x in r)+"\n")
log("\n[out] acceptor_trich.npz acceptor_trich_report.tsv")
