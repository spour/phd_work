#!/usr/bin/env python3
# one scan that serves both open questions: the anchor acceptor's distance
import csv, os, sys, numpy as np, pandas as pd
def log(*a): print(*a, file=sys.stderr, flush=True)
FASTA = "../train/data/hg38.fa"
U1_LU = "../train/splice_train/data/u1_lookup.npy"
UPMAX, DN = 5000, 523
BASE = UPMAX - 2000
ACC_END = 1874
WS = [500, 1000, 2000, 5000]
SPLITS = [250, 437, 750, 1000]
OUTNPZ = "anchor_sweep.npz"

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
ROW = np.load("upstream_rowmask.npz")["ROW"]
N = len(D)

class Faidx:
    _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
    def __init__(s, fa):
        s.fh = open(fa, "rb"); s.idx = {}
        for L in open(fa + ".fai"):
            p = L.split(); s.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    def fetch(s, c, a, b, st="+"):
        c = c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln, off, lb, lw = s.idx[c]
        if a < 0 or b > Ln: return None
        bp = lambda p: off + (p // lb) * lw + (p % lb)
        s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
        q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st == "-" else q

_MAP = np.full(256, -1, np.int64)
for i, ch in enumerate("ACGT"): _MAP[ord(ch)] = i
def code(seq): return _MAP[np.frombuffer(seq.encode("ascii"), np.uint8)]
def khash(c, k, n):
    w = np.lib.stride_tricks.sliding_window_view(c, k)[:n]
    bad = (w < 0).any(1); h = w @ (4 ** np.arange(k - 1, -1, -1)); h[bad] = -1
    return h

from maxentpy import maxent
M3 = maxent.load_matrix3()
MT = [np.array([M3[i][j] for j in range(len(M3[i]))], float) for i in range(9)]
CONST3 = np.log2(maxent.cons1_3["A"] * maxent.cons2_3["G"] /
                 (maxent.bgd_3["A"] * maxent.bgd_3["G"]))
LT = [np.log2(np.maximum(m, 1e-300)) for m in MT]
def me3_scores(c, n):
    out = np.full(n, np.nan)
    ok = (c[18:18 + n] == 0) & (c[19:19 + n] == 2)
    h7 = [khash(c[o:], 7, n) for o in (0, 7, 4, 11)]
    h4a = khash(c[14:], 4, n); h3b = khash(c[20:], 3, n)
    h3 = khash(c[4:], 3, n); h4b = khash(c[7:], 4, n); h3c = khash(c[11:], 3, n)
    parts = h7 + [h4a, h3b, h3, h4b, h3c]
    ok &= np.all(np.array([p >= 0 for p in parts]), 0)
    if not ok.any(): return out
    i = np.where(ok)[0]
    h14 = h4a[i] * 64 + h3b[i]
    out[i] = CONST3 + (LT[0][h7[0][i]] + LT[1][h7[1][i]] + LT[2][h14] + LT[3][h7[2][i]]
                       + LT[4][h7[3][i]] - LT[5][h3[i]] - LT[6][h4b[i]] - LT[7][h3c[i]]
                       - LT[8][h4a[i]])
    return out

def lookup_array(path, neg):
    d = np.load(path, allow_pickle=True); d = d.item() if d.dtype == object else dict(d)
    A = np.full(4 ** 9, np.nan); p = 4 ** np.arange(8, -1, -1)
    for k, v in d.items():
        cc = code(k)
        if (cc < 0).any(): continue
        A[int(cc @ p)] = (-float(v)) if neg else float(v)
    return A
U1 = lookup_array(U1_LU, True)

NAMES = (["anchor_me3", "anchor_off", "anchor_dist"]
         + [f"me3_W{w}" for w in WS]
         + [f"u1_dist_{s}" for s in SPLITS] + [f"u1_prox_{s}" for s in SPLITS])
OUT = np.full((N, len(NAMES)), np.nan)
G = Faidx(FASTA)
rows = np.where(ROW)[0]
if os.environ.get("NROWS"): rows = rows[:int(os.environ["NROWS"])]
short = 0
for n, i in enumerate(rows):
    cen = int(D.center.iloc[i]); st = D.strand.iloc[i]
    a, b = (cen - UPMAX, cen + DN) if st == "+" else (cen - DN, cen + UPMAX)
    q = G.fetch(D.chrom.iloc[i], a, b, st)
    if q is None or len(q) != UPMAX + DN:
        short += 1
        a, b = (cen - 2000, cen + DN) if st == "+" else (cen - DN, cen + 2000)
        q = G.fetch(D.chrom.iloc[i], a, b, st)
        if q is None or len(q) != 2000 + DN: continue
        c = code(q); base = 0
    else:
        c = code(q); base = BASE
    nsc = base + ACC_END + 1
    s3 = me3_scores(c, nsc)
    h = khash(c[base:], 9, ACC_END + 1)
    s1 = np.full(ACC_END + 1, np.nan); ok = h >= 0; s1[ok] = U1[h[ok]]
    acc = s3[base:base + ACC_END + 1]
    f = np.isfinite(acc)
    if f.any():
        j = int(np.nanargmax(acc))
        OUT[i, 0] = acc[j]; OUT[i, 1] = j; OUT[i, 2] = 2000 - (j + 20)
    for k, w in enumerate(WS):
        lo = base + 2000 - w
        if lo < 0: continue
        v = s3[lo:base + ACC_END + 1]
        if np.isfinite(v).any(): OUT[i, 3 + k] = np.nanmax(v)
    for k, s in enumerate(SPLITS):
        d_ = s1[:2000 - s]; p_ = s1[2000 - s:]
        if np.isfinite(d_).any(): OUT[i, 3 + len(WS) + k] = np.nanmean(d_)
        if np.isfinite(p_).any(): OUT[i, 3 + len(WS) + len(SPLITS) + k] = np.nanmean(p_)
    if (n + 1) % 5000 == 0: log(f"  {n+1:,}/{len(rows):,}")

np.savez_compressed(OUTNPZ, X=OUT.astype(np.float64), names=np.array(NAMES))
print(f"scanned {len(rows):,} rows; {short:,} could not fetch 5 kb (2 kb fallback)")
ref = D["spl_upstream_max_me3"].to_numpy(float)
m = ROW & np.isfinite(OUT[:, 0]) & np.isfinite(ref)
print(f"[validate] anchor_me3 vs spl_upstream_max_me3: n {int(m.sum()):,} "
      f"max abs diff {np.max(np.abs(OUT[m,0]-ref[m])):.6f} r {np.corrcoef(OUT[m,0],ref[m])[0,1]:.6f}")
m2 = ROW & np.isfinite(OUT[:, 3 + WS.index(2000)])
print(f"[validate] me3_W2000 vs anchor_me3: max abs diff "
      f"{np.nanmax(np.abs(OUT[m2,3+WS.index(2000)]-OUT[m2,0])):.6f}")
uprox = D["spl_pas_prox_u1"].to_numpy(float)
umean = D["spl_upstream_mean_u1"].to_numpy(float)
udist = (6 * umean - uprox) / 5.0
c437p = OUT[:, 3 + len(WS) + len(SPLITS) + SPLITS.index(437)]
c437d = OUT[:, 3 + len(WS) + SPLITS.index(437)]
m3 = ROW & np.isfinite(c437p) & np.isfinite(uprox)
print(f"[validate] u1_prox_437 vs spl_pas_prox_u1: max abs diff "
      f"{np.max(np.abs(c437p[m3]-uprox[m3])):.6f} r {np.corrcoef(c437p[m3],uprox[m3])[0,1]:.6f}")
m4 = ROW & np.isfinite(c437d) & np.isfinite(udist)
print(f"[validate] u1_dist_437 vs (6*mean-prox)/5: max abs diff "
      f"{np.max(np.abs(c437d[m4]-udist[m4])):.6f} r {np.corrcoef(c437d[m4],udist[m4])[0,1]:.6f}")
log("DONE_ANCHOR_SWEEP")
