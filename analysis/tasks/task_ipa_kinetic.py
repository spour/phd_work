#!/usr/bin/env python3
# ipa contrast, part 2: kinetic-competition features from the maxent scan alone
import csv, os, sys, numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA = "../train/data/hg38.fa"
U1_LU = "../train/splice_train/data/u1_lookup.npy"
ME5_LU = "../train/splice_train/data/me5_lookup.npy"
SPAN, PAD = 10000, 30
DTHR, A3THR = 8.46, 10.0
POSITIONS = os.environ.get("POSITIONS", "ipa_positions.tsv")
OUTNPZ = os.environ.get("OUTNPZ", "ipa_kin.npz")

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
def gh(c, i, off, k):
    w = c[i[:, None] + (off + np.arange(k))[None, :]]
    h = w @ (4 ** np.arange(k - 1, -1, -1))
    h[(w < 0).any(1)] = -1
    return h
def khash(c, k, n):
    w = np.lib.stride_tricks.sliding_window_view(c, k)[:n]
    h = w @ (4 ** np.arange(k - 1, -1, -1)); h[(w < 0).any(1)] = -1
    return h

from maxentpy import maxent
M3 = maxent.load_matrix3()
LT = [np.log2(np.maximum(np.array([M3[i][j] for j in range(len(M3[i]))], float), 1e-300)) for i in range(9)]
CONST3 = np.log2(maxent.cons1_3["A"] * maxent.cons2_3["G"] / (maxent.bgd_3["A"] * maxent.bgd_3["G"]))
def me3_at(c, i):
    h = [gh(c, i, o, 7) for o in (0, 7, 4, 11)]
    h4a = gh(c, i, 14, 4); h3b = gh(c, i, 20, 3)
    h3 = gh(c, i, 4, 3); h4b = gh(c, i, 7, 4); h3c = gh(c, i, 11, 3)
    ok = (h[0] >= 0) & (h[1] >= 0) & (h[2] >= 0) & (h[3] >= 0) & (h4a >= 0) & (h3b >= 0) & \
         (h3 >= 0) & (h4b >= 0) & (h3c >= 0)
    out = np.full(len(i), np.nan)
    if not ok.any(): return out
    j = np.where(ok)[0]
    h14 = h4a[j] * 64 + h3b[j]
    out[j] = CONST3 + (LT[0][h[0][j]] + LT[1][h[1][j]] + LT[2][h14] + LT[3][h[2][j]] + LT[4][h[3][j]]
                       - LT[5][h3[j]] - LT[6][h4b[j]] - LT[7][h3c[j]] - LT[8][h4a[j]])
    return out
def lookup_array(path, neg):
    d = np.load(path, allow_pickle=True); d = d.item() if d.dtype == object else dict(d)
    A = np.full(4 ** 9, np.nan); p = 4 ** np.arange(8, -1, -1)
    for k, v in d.items():
        cc = code(k)
        if (cc < 0).any(): continue
        A[int(cc @ p)] = (-float(v)) if neg else float(v)
    return A
U1 = lookup_array(U1_LU, True); ME5 = lookup_array(ME5_LU, False)
log("lookups loaded")

R = list(csv.DictReader(open(POSITIONS), delimiter="\t"))
if os.environ.get("NROWS"): R = R[:int(os.environ["NROWS"])]
NAMES = ["kin_d_up_donor", "kin_d_dn_acceptor", "kin_intron_len", "kin_frac_pos",
         "kin_up_donor_score", "kin_dn_acc_score", "kin_u1_up", "kin_u1_pa"]
OUT = np.full((len(R), len(NAMES)), np.nan)
G = Faidx(FASTA)
L = SPAN + PAD
nod = noa = 0
for i, r in enumerate(R):
    cen = int(r["center"]); st = r["strand"]
    a, b = (cen - L, cen + L) if st == "+" else (cen - L + 1, cen + L + 1)
    q = G.fetch(r["chrom"], a, b, st)
    if q is None or len(q) != 2 * L: continue
    c = code(q); S = L
    # upstream donors: 9-mer start s, gt at s+3,s+4, first intronic base s+3
    lim = S - 3
    gt = np.where((c[3:lim] == 2) & (c[4:lim + 1] == 3))[0]
    if gt.size:
        h = khash(c, 9, len(c) - 8)
        sc = np.full(len(gt), np.nan)
        ok = h[gt] >= 0
        sc[ok] = ME5[h[gt][ok]]
        strong = gt[np.isfinite(sc) & (sc >= DTHR)]
        if strong.size:
            s = strong[-1]
            OUT[i, 0] = S - (s + 3); OUT[i, 4] = float(ME5[h[s]])
        else: nod += 1
    else: nod += 1
    # downstream acceptors: 23-mer start s, ag at s+18,s+19, last intronic base s+19
    lo = S - 18
    hi = len(c) - 23
    if hi > lo:
        idx = np.arange(max(lo, 0), hi)
        ag = idx[(c[idx + 18] == 0) & (c[idx + 19] == 2) & ((idx + 19) > S)]
        if ag.size:
            v = me3_at(c, ag)
            strong = ag[np.isfinite(v) & (v >= A3THR)]
            if strong.size:
                s = strong[0]
                OUT[i, 1] = (s + 19) - S
                OUT[i, 5] = float(v[np.searchsorted(ag, s)])
            else: noa += 1
        else: noa += 1
    else: noa += 1
    if np.isfinite(OUT[i, 0]) and np.isfinite(OUT[i, 1]):
        OUT[i, 2] = OUT[i, 0] + OUT[i, 1]
        OUT[i, 3] = OUT[i, 0] / OUT[i, 2]
    for k, (x0, x1) in enumerate([(S - 1000, S - 500), (S - 100, S + 100)]):
        seg = c[x0:x1 + 8]
        if len(seg) >= 9:
            hh = khash(seg, 9, len(seg) - 8)
            vv = np.where(hh >= 0, U1[np.clip(hh, 0, len(U1) - 1)], np.nan)
            if np.isfinite(vv).any(): OUT[i, 6 + k] = np.nanmean(vv)
    if (i + 1) % 10000 == 0: log(f"  site {i+1:,}/{len(R):,}")
np.savez_compressed(OUTNPZ, X=OUT.astype(np.float32), names=np.array(NAMES))
print(f"{POSITIONS}: {len(R)} sites -> {OUTNPZ}")
print(f"  censored: no strong donor upstream within {SPAN} nt {nod} ({nod/len(R):.3f}), "
      f"no strong acceptor downstream {noa} ({noa/len(R):.3f})")
for j, nm in enumerate(NAMES):
    v = OUT[:, j]; f = np.isfinite(v)
    print(f"  {nm:<20} finite {f.mean():.3f}  median {np.median(v[f]):.3f} "
          f"IQR [{np.percentile(v[f],25):.3f},{np.percentile(v[f],75):.3f}]")
log("DONE_IPA_KINETIC")
