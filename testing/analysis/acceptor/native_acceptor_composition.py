#!/usr/bin/env python3
# the base composition of the fragments gia_native_acceptor.py implanted
import sys, gzip
import numpy as np
sys.argv = [sys.argv[0]]
GTF = ("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome/"
       "GRCh38.gencode.v43.annotation.gtf")
FA = "/scratch/spour98/jobs/hg38.fa"
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
    return fh.read((e - s) + (e - s)//lb + 2).decode().replace("\n", "")[:e-s].upper()
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
    for k in range(len(ex) - 1):
        istart, iend = ex[k][1] + 1, ex[k+1][0] - 1
        if iend - istart + 1 < 300: continue
        key = (c, st, ex[k+1][0] if st == "+" else ex[k][1])
        if key in seen: continue
        s = (fetch(c, ex[k+1][0]-21, ex[k+1][0]+2) if st == "+"
             else (lambda z: rc(z) if z else None)(fetch(c, ex[k][1]-3, ex[k][1]+20)))
        if not s or len(s) != 23 or set(s) - set("ACGT"): continue
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
    m = (istart + iend)//2
    if m-istart < 200 or iend-m < 200: return None
    z = fetch(c, m-L//2, m-L//2+L)
    return z if st == "+" else (rc(z) if z else None)
log(f"\n{'class':<11} {'L':>4} {'n':>4} {'A':>6} {'C':>6} {'G':>6} {'T':>6} {'A+T':>6} "
    f"{'pyrimidine':>11} {'T/(T+C)':>8}")
for L in (60, 120):
    for nm, pool_, kind in (("acc_top", top, "acc"), ("acc_mid", mid, "acc"),
                            ("intron_ctl", top, "ctl")):
        S = [frag(a, L, kind) for a in pool_]
        S = [s for s in S if s and len(s) == L and not (set(s) - set("ACGT"))][:150]
        f = {b: np.mean([s.count(b)/L for s in S]) for b in "ACGT"}
        log(f"  {nm:<9} {L:>4} {len(S):>4} {f['A']:>6.3f} {f['C']:>6.3f} {f['G']:>6.3f} "
            f"{f['T']:>6.3f} {f['A']+f['T']:>6.3f} {f['C']+f['T']:>11.3f} "
            f"{f['T']/(f['T']+f['C']):>8.3f}")
