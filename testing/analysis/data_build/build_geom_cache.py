#!/usr/bin/env python3
# one-time cache of the me5-scan quantities needed for competitor-split re-runs
import sys, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
FASTA = "../train/data/hg38.fa"; ME5NPY = "../train/splice_train/data/me5_lookup.npy"
POS = "fig3_new_positional_features.tsv"
ND = 8.0; PAD = 12; CAP = 8000

B = np.full(256, -1, np.int64)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): B[ch] = v
POW = (4 ** np.arange(8, -1, -1)).astype(np.int64)
a = np.load(ME5NPY, allow_pickle=True); me5d = a.item() if a.dtype == object else dict(a)
ks = np.frombuffer("".join(me5d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
ME5 = np.full(4 ** 9, -99.0); ME5[B[ks] @ POW] = np.fromiter(me5d.values(), np.float64, len(me5d))

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
        bp = lambda p: off + (p // lb) * lw + (p % lb); s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
        q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st == "-" else q
G = Faidx(FASTA)

def hit_offsets(chrom, center, strand, span):
    a, b = center - span - PAD, center + span + PAD
    seq = G.fetch(chrom, a, b, strand)
    if not seq or len(seq) < 2 * span: return None
    tok = B[np.frombuffer(seq.encode("ascii"), np.uint8)]
    sw = np.lib.stride_tricks.sliding_window_view(tok, 9); good = (sw >= 0).all(1)
    sc = np.full(sw.shape[0], -99.0); sc[good] = ME5[sw[good] @ POW]
    d = -(span + PAD) + np.arange(sc.size) + 3
    return d, sc

R = list(csv.DictReader(open(POS), delimiter="\t"))
chrom = [r["chrom"] for r in R]; center = [int(r["center"]) for r in R]; strand = [r["strand"] for r in R]
def gf(col): return np.array([float(r[col]) if r[col] not in ("", "nan") else np.nan for r in R])
dpa = gf("d_anchor_to_pa"); apos = gf("anchor_3ss_pos")
NP = len(R)

pres_up = np.full(NP, np.nan); pres_dn = np.full(NP, np.nan)
max_up = np.full(NP, np.nan); max_dn = np.full(NP, np.nan)
max5 = np.full(NP, -99.0)
ncap = 0
for i in range(NP):
    d = dpa[i]
    if not np.isfinite(d) or d < 1: continue
    if d > CAP: d = CAP; ncap += 1
    d = int(d)
    ho = hit_offsets(chrom[i], center[i], strand[i], d)
    if ho is None: continue
    doff, sc = ho
    up_m = (doff >= -d) & (doff < 0); dn_m = (doff > 0) & (doff <= d)
    strong = sc >= ND
    pres_up[i] = 1.0 if (up_m & strong).any() else 0.0
    pres_dn[i] = 1.0 if (dn_m & strong).any() else 0.0
    max_up[i] = float(sc[up_m].max()) if up_m.any() else -99.0
    max_dn[i] = float(sc[dn_m].max()) if dn_m.any() else -99.0
    if np.isfinite(apos[i]) and dpa[i] >= 9:
        max5[i] = float(sc[up_m].max()) if up_m.any() else float(sc.max())
    if (i + 1) % 15000 == 0: log(f"  scan {i+1}/{NP}")
log(f"scan done; d capped at {CAP} for {ncap} peaks")
np.savez("geom_cache.npz", pres_up=pres_up, pres_dn=pres_dn, max5=max5, max_up=max_up, max_dn=max_dn)
log("saved geom_cache.npz")
