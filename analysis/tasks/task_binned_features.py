#!/usr/bin/env python3
# binned splicing and composition features, on one grid, for the fair binning test
import csv, os, sys, numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA = "../train/data/hg38.fa"
U1_LU = "../train/splice_train/data/u1_lookup.npy"
ME5_LU = "../train/splice_train/data/me5_lookup.npy"
UP, DN_SCAN = 2000, 523
NBIN, BINW = 25, 100
GRID_END = NBIN * BINW

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

# coded-sequence helpers
_MAP = np.full(256, -1, np.int64)
for i, ch in enumerate("ACGT"): _MAP[ord(ch)] = i
def code(seq): return _MAP[np.frombuffer(seq.encode("ascii"), np.uint8)]
def khash(c, k, n):
    w = np.lib.stride_tricks.sliding_window_view(c, k)[:n]
    bad = (w < 0).any(1)
    h = w @ (4 ** np.arange(k - 1, -1, -1))
    h[bad] = -1
    return h

# maxent score3, vectorised
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
    h4a = khash(c[14:], 4, n)
    h3b = khash(c[20:], 3, n)
    h3 = khash(c[4:], 3, n); h4b = khash(c[7:], 4, n); h3c = khash(c[11:], 3, n)
    parts = h7 + [h4a, h3b, h3, h4b, h3c]
    ok &= np.all(np.array([p >= 0 for p in parts]), 0)
    if not ok.any(): return out
    i = np.where(ok)[0]
    h14 = h4a[i] * 64 + h3b[i]
    v = (LT[0][h7[0][i]] + LT[1][h7[1][i]] + LT[2][h14] + LT[3][h7[2][i]] + LT[4][h7[3][i]]
         - LT[5][h3[i]] - LT[6][h4b[i]] - LT[7][h3c[i]] - LT[8][h4a[i]])
    out[i] = CONST3 + v
    return out

# 9-mer lookups (pipeline tables)
def lookup_array(path, neg):
    d = np.load(path, allow_pickle=True)
    d = d.item() if d.dtype == object else dict(d)
    A = np.full(4 ** 9, np.nan)
    p = 4 ** np.arange(8, -1, -1)
    for k, v in d.items():
        cc = code(k)
        if (cc < 0).any(): continue
        A[int(cc @ p)] = (-float(v)) if neg else float(v)
    return A
U1 = lookup_array(U1_LU, True); ME5 = lookup_array(ME5_LU, False)
log(f"lookups loaded: u1 {np.isfinite(U1).sum():,} entries, me5 {np.isfinite(ME5).sum():,}")

def nine_scores(c, n, tab, gt=False):
    h = khash(c, 9, n)
    out = np.full(n, np.nan)
    ok = h >= 0
    if gt: ok &= (c[3:3 + n] == 2) & (c[4:4 + n] == 3)
    out[ok] = tab[h[ok]]
    return out

# hand-built donor pwm behind spl_u1_suppression_up (splice_extra_features._donor)
_DONOR = {0:{"A":.3,"C":.1,"G":.4,"T":.2},1:{"A":.6,"C":.1,"G":.1,"T":.2},
          2:{"A":.1,"C":.0,"G":.8,"T":.1},3:{"G":1.0},4:{"T":1.0},
          5:{"A":.6,"G":.3,"C":.05,"T":.05},6:{"A":.7,"G":.1,"C":.1,"T":.1},
          7:{"G":.8,"A":.1,"C":.05,"T":.05},8:{"T":.5,"C":.2,"A":.2,"G":.1}}
DW = np.zeros((9, 5))
for j in range(9):
    for b, ch in enumerate("ACGT"): DW[j, b] = _DONOR[j].get(ch, 0.0)
def donor_density(c, lo, hi, thresh=0.6):
    reg = c[lo:hi]; L = len(reg)
    if L < 12: return np.nan
    w = np.lib.stride_tricks.sliding_window_view(reg, 9)
    idx = np.where(w < 0, 4, w)
    sc = DW[np.arange(9)[None, :], idx].mean(1)
    gt = (reg[3:L - 5] == 2) & (reg[4:L - 4] == 3)
    n = min(len(sc), len(gt))
    hits = int(((sc[:n] >= thresh) & gt[:n]).sum())
    return float(100.0 * hits / max(L - 9, 1))

def binstat(v, nb, w):
    b = np.arange(len(v)) // w
    ok = np.isfinite(v); mx = np.full(nb, np.nan); mn = np.full(nb, np.nan)
    if ok.any():
        cnt = np.bincount(b[ok], minlength=nb)
        s = np.bincount(b[ok], weights=v[ok], minlength=nb)
        nz = cnt > 0
        mn[nz] = s[nz] / cnt[nz]
        m = np.full(nb, -np.inf)
        np.maximum.at(m, b[ok], v[ok])
        mx[nz] = m[nz]
    return mx, mn

PEAKS = os.environ.get("PEAKS", "fig3_new_positional_features.tsv")
R = list(csv.DictReader(open(PEAKS), delimiter="\t"))
if os.environ.get("NROWS"): R = R[:int(os.environ["NROWS"])]
NP = len(R)
G = Faidx(FASTA)
K3NAMES = ["spl_upstream_max_me3", "spl_pas_prox_u1", "spl_u1_suppression_up"]
NAMES = ([f"bin{b+1}_{s}_{st}" for s in ("me3", "me5", "u1") for st in ("max", "mean") for b in range(NBIN)]
         + [f"bin{b+1}_gc" for b in range(NBIN)] + K3NAMES)
PB3 = np.minimum(8, 1 + (np.arange(GRID_END) * 8) // 2500)
OUT = np.full((NP, len(NAMES)), np.nan)
VALIDATE = os.environ.get("VALIDATE", "0") == "1"
vrows, v_me3, v_u1 = [], [], []
for i, r in enumerate(R):
    cen = int(r["center"]); st = r["strand"]
    a, b = (cen - UP, cen + DN_SCAN) if st == "+" else (cen - DN_SCAN, cen + UP)
    q = G.fetch(r["chrom"], a, b, st)
    if q is None or len(q) != UP + DN_SCAN: continue
    c = code(q)
    n3 = GRID_END; n9 = GRID_END
    s3 = me3_scores(c, n3); s5 = nine_scores(c, n9, ME5, gt=True); s1 = nine_scores(c, n9, U1)
    j = 0
    for v in (s3, s5, s1):
        mx, mn = binstat(v, NBIN, BINW)
        OUT[i, j:j + NBIN] = mx; j += NBIN
        OUT[i, j:j + NBIN] = mn; j += NBIN
    m6 = np.isfinite(s3) & (PB3[:len(s3)] <= 6)
    OUT[i, len(NAMES) - 3] = s3[m6].max() if m6.any() else np.nan
    m1 = np.isfinite(s1) & (PB3[:len(s1)] == 6)
    OUT[i, len(NAMES) - 2] = s1[m1].mean() if m1.any() else np.nan
    OUT[i, len(NAMES) - 1] = donor_density(c, UP - 500, UP)
    isgc = (c == 1) | (c == 2); isn = c < 0
    for bb in range(NBIN):
        sl = slice(bb * BINW, (bb + 1) * BINW)
        d = (~isn[sl]).sum()
        OUT[i, j + bb] = isgc[sl].sum() / d if d else np.nan
    if VALIDATE and i % 200 == 0 and len(vrows) < 300:
        W, NB = 2500, 8
        p3 = np.arange(W - 23 + 1); pb3 = np.minimum(NB, 1 + (p3 * NB) // W)
        v3 = me3_scores(c, W - 23 + 1)
        m = np.isfinite(v3) & (pb3 <= 6)
        p9 = np.arange(W - 9 + 1); pb9 = np.minimum(NB, 1 + (p9 * NB) // W)
        v1 = nine_scores(c, W - 9 + 1, U1)
        m1 = np.isfinite(v1) & (pb9 == 6)
        vrows.append(i)
        v_me3.append(v3[m].max() if m.any() else np.nan)
        v_u1.append(v1[m1].mean() if m1.any() else np.nan)
    if (i + 1) % 10000 == 0: log(f"  peak {i+1:,}/{NP:,}")

np.savez_compressed(os.environ.get("OUTNPZ", "binned_features.npz"), X=OUT.astype(np.float32), names=np.array(NAMES))
print(f"wrote binned_features.npz: {NP} peaks x {len(NAMES)} features "
      f"({3*2*NBIN} splice, {NBIN} GC); all-NaN rows {int(np.isnan(OUT).all(1).sum())}")
frac = np.isnan(OUT).mean(0)
for tag, sl in [("me3", slice(0, 2 * NBIN)), ("me5", slice(2 * NBIN, 4 * NBIN)),
                ("u1", slice(4 * NBIN, 6 * NBIN)), ("gc", slice(6 * NBIN, 7 * NBIN))]:
    print(f"  {tag:<4} mean NaN fraction {frac[sl].mean():.4f} (max over bins {frac[sl].max():.4f})")
if VALIDATE:
    import pandas as pd
    D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
    km = {(x["chrom"], int(x["center"]), x["strand"], x["gene"]): k for k, x in enumerate(R)}
    idx = {km[(cc, int(ce), ss, gg)]: k for k, (cc, ce, ss, gg) in
           enumerate(zip(D.chrom, D.center, D.strand, D.gene)) if (cc, int(ce), ss, gg) in km}
    ref3 = np.array([D["spl_upstream_max_me3"].to_numpy()[idx[i]] for i in vrows])
    ref1 = np.array([D["spl_pas_prox_u1"].to_numpy()[idx[i]] for i in vrows])
    a3 = np.array(v_me3); a1 = np.array(v_u1)
    a3 = OUT[vrows, NAMES.index("spl_upstream_max_me3")]
    a1 = OUT[vrows, NAMES.index("spl_pas_prox_u1")]
    ref2 = np.array([D["spl_u1_suppression_up"].to_numpy()[idx[i]] for i in vrows])
    a2 = OUT[vrows, NAMES.index("spl_u1_suppression_up")]
    for nm, x, y in [("spl_upstream_max_me3", a3, ref3), ("spl_pas_prox_u1", a1, ref1),
                     ("spl_u1_suppression_up", a2, ref2)]:
        m = np.isfinite(x) & np.isfinite(y)
        print(f"  [validate] {nm}: n {int(m.sum())}, max abs diff {np.max(np.abs(x[m]-y[m])):.6f}, "
              f"r {np.corrcoef(x[m], y[m])[0,1]:.6f}")
log("DONE_BINNED_FEATURES")
