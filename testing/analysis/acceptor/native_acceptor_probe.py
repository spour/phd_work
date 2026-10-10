#!/usr/bin/env python3
# two checks the native-acceptor result needs before it can be believed
import sys, csv
import numpy as np
FA = "/scratch/spour98/jobs/hg38.fa"
GTF = ("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome/"
       "GRCh38.gencode.v43.annotation.gtf")
def log(*a): print(*a, file=sys.stderr, flush=True)
FAI = {}
for line in open(FA + ".fai"):
    n, ln, off, lb, wb = line.split()[:5]; FAI[n] = (int(ln), int(off), int(lb), int(wb))
fh = open(FA, "rb")
def fetch(c, s, e):
    if c not in FAI: return None
    ln, off, lb, wb = FAI[c]
    if s < 0 or e > ln or e <= s: return None
    fh.seek(off + s // lb * wb + s % lb)
    return fh.read((e-s) + (e-s)//lb + 2).decode().replace("\n", "")[:e-s].upper()
def rc(s): return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0
TX = {}
with open(GTF) as g:
    for line in g:
        if line[0] == "#": continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "exon" or 'transcript_type "protein_coding"' not in f[8]: continue
        i = f[8].find('transcript_id "'); tid = f[8][i+15:f[8].find('"', i+15)]
        TX.setdefault(tid, (f[0], f[6], []))[2].append((int(f[3]), int(f[4])))
ACC = []; seen = set()
for tid, (c, st, ex) in TX.items():
    if c not in FAI or len(ex) < 2: continue
    ex.sort()
    for k in range(len(ex)-1):
        istart, iend = ex[k][1]+1, ex[k+1][0]-1
        if iend - istart + 1 < 300: continue
        key = (c, st, ex[k+1][0] if st == "+" else ex[k][1])
        if key in seen: continue
        s = (fetch(c, ex[k+1][0]-21, ex[k+1][0]+2) if st == "+"
             else (lambda z: rc(z) if z else None)(fetch(c, ex[k][1]-3, ex[k][1]+20)))
        if not s or len(s) != 23 or set(s)-set("ACGT"): continue
        seen.add(key)
        if s[18:20] != "AG": continue
        ACC.append((c, st, key[2], istart, iend, s3(s)))
SC = np.array([a[5] for a in ACC]); rng = np.random.default_rng(42)
o = np.argsort(-SC); top = [ACC[i] for i in o[:150]]
lo, hi = np.percentile(SC, 45), np.percentile(SC, 55)
pool = [a for a in ACC if lo <= a[5] <= hi]
mid = [pool[i] for i in rng.choice(len(pool), 150, replace=False)]
def frag(a, L, kind):
    c, st, pos, istart, iend, _ = a
    if kind == "acc":
        return (fetch(c, pos-1-(L-3), pos+2) if st == "+"
                else (lambda z: rc(z) if z else None)(fetch(c, pos-3, pos-3+L)))
    m = (istart+iend)//2
    if m-istart < 200 or iend-m < 200: return None
    z = fetch(c, m-L//2, m-L//2+L)
    return z if st == "+" else (rc(z) if z else None)

FR = {}
for L in (60, 120):
    for nm, pl, kind in (("acc_top", top, "acc"), ("acc_mid", mid, "acc"),
                         ("intron_ctl", top, "ctl")):
        got = 0
        for a in pl:
            s = frag(a, L, kind)
            if not s or len(s) != L or set(s)-set("ACGT"): continue
            FR[(nm, L, got)] = s; got += 1
            if got >= 150: break
D = {}
for r in csv.DictReader(open("gia_native_acceptor.tsv"), delimiter="\t"):
    D[(r["background"], r["class"], int(r["length"]), r["arm"], int(r["idx"]))] = float(r["best_delta"])
log(f"[join] {len(FR):,} fragments rebuilt; "
    f"{sum(1 for k in FR if ('real_canonical',k[0],k[1],'native',k[2]) in D):,} matched to a score")

def best_internal_s3(s):
    return max((s3(s[i:i+23]) for i in range(len(s)-22)), default=-99.0)

log(f"\n1. BEST ACCEPTOR ANYWHERE INSIDE EACH FRAGMENT (score3, in register)")
log(f"   {'class':<11} {'L':>4} {'n':>4} {'median':>8} {'p90':>7} {'max':>7} {'>=10':>7} {'>=14':>7}")
for L in (60, 120):
    for nm in ("acc_top", "acc_mid", "intron_ctl"):
        v = np.array([best_internal_s3(FR[k]) for k in FR if k[0] == nm and k[1] == L])
        log(f"   {nm:<11} {L:>4} {len(v):>4} {np.median(v):>8.2f} {np.percentile(v,90):>7.2f} "
            f"{v.max():>7.2f} {100*np.mean(v>=10):>6.0f}% {100*np.mean(v>=14):>6.0f}%")

log(f"\n2. PER-FRAGMENT EFFECT AGAINST PER-FRAGMENT COMPOSITION")
for bg in ("real_canonical", "real_competitor"):
    for L in (60, 120):
        X, Y, C, T = [], [], [], []
        for k, s in FR.items():
            if k[1] != L: continue
            d = D.get((bg, k[0], L, "native", k[2]))
            if d is None: continue
            X.append((s.count("A")+s.count("T"))/L); Y.append(d)
            C.append(s.count("C")/L); T.append(s.count("T")/L)
        X, Y, C, T = map(np.array, (X, Y, C, T))
        r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        sl = np.polyfit(X, Y, 1)[0]
        log(f"   {bg:<16} L={L:>3}  n={len(X):>3}   effect vs A+T r={r(X,Y):+.3f}  "
            f"vs C r={r(C,Y):+.3f}  vs T r={r(T,Y):+.3f}   slope {sl:+.3f} logit per unit A+T")
        q = np.quantile(X, [0, .25, .5, .75, 1])
        b = [Y[(X >= q[i]) & (X <= q[i+1])].mean() for i in range(4)]
        log(f"   {'':16}        A+T quartiles {q[0]:.2f}-{q[4]:.2f}: mean effect " +
            "  ".join(f"{x:+.3f}" for x in b))
