#!/usr/bin/env python3
# kinetic-competition features from the maxent scan only (no annotation as a feature)
import sys, csv, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "scripts")
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA  = "../train/data/hg38.fa"
ME5NPY = "../train/splice_train/data/me5_lookup.npy"
POS    = "fig3_new_positional_features.tsv"
DTHR   = 8.46
A3THR  = 8.19
SPAN   = 50000
PAD    = 25

B = np.full(256, -1, np.int64)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): B[ch] = v
POW9 = (4 ** np.arange(8, -1, -1)).astype(np.int64)
_a = np.load(ME5NPY, allow_pickle=True); me5d = _a.item() if _a.dtype == object else dict(_a)
_ks = np.frombuffer("".join(me5d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
ME5 = np.full(4 ** 9, -99.0); ME5[B[_ks] @ POW9] = np.fromiter(me5d.values(), np.float64, len(me5d))

_G = None; _m3 = None
def _init():
    global _G, _m3
    from maxentpy.maxent import load_matrix3
    _m3 = load_matrix3()
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
            if a < 0: a = 0
            if b > Ln: b = Ln
            bp = lambda p: off + (p // lb) * lw + (p % lb)
            s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
            q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
            return (q.translate(s._T)[::-1] if st == "-" else q), a, b
    _G = Faidx(FASTA)

def scan(chrom, center, strand):
    from maxentpy import maxent
    a0 = center - (SPAN + PAD); b0 = center + (SPAN + PAD)
    r = _G.fetch(chrom, a0, b0, strand)
    if r is None: return None
    seq, ra, rb = r
    if len(seq) < 200: return None
    n = len(seq)
    if strand == "+": base = ra - center
    else:             base = center - (rb - 1)
    arr = np.frombuffer(seq.encode("ascii"), np.uint8); tok = B[arr]
    sw = np.lib.stride_tricks.sliding_window_view(tok, 9); good = (sw >= 0).all(1)
    sc5 = np.full(sw.shape[0], -99.0); sc5[good] = ME5[sw[good] @ POW9]
    o5 = base + np.arange(sc5.size) + 3
    up_dist = up_score = np.nan
    dm = (o5 < 0) & (sc5 >= DTHR)
    if dm.any():
        oo = o5[dm]; k = int(np.argmax(oo)); up_dist = float(-oo[k]); up_score = float(sc5[dm][k])
    L3 = len(seq) - 22
    if L3 > 0:
        isA = (arr[18:18 + L3] == 65); isG = (arr[19:19 + L3] == 71)
        agidx = np.where(isA & isG)[0]
        o3 = base + agidx + 20
        dn_dist = dn_score = np.nan
        sel = np.where(o3 > 0)[0]
        for j in sel[np.argsort(o3[sel])]:
            ix = int(agidx[j]); km = seq[ix:ix + 23]
            if len(km) < 23 or "N" in km: continue
            s3 = maxent.score3(km, matrix=_m3)
            if s3 >= A3THR:
                dn_dist = float(o3[j]); dn_score = float(s3); break
    else:
        dn_dist = dn_score = np.nan
    ilen = up_dist + dn_dist if (up_dist == up_dist and dn_dist == dn_dist) else np.nan
    frac = up_dist / ilen if (ilen == ilen and ilen > 0) else np.nan
    return (up_dist, up_score, dn_dist, dn_score, ilen, frac)

def work(job):
    i, c, ce, st = job
    r = scan(c, ce, st)
    return i, r

if __name__ == "__main__":
    R = list(csv.DictReader(open(POS), delimiter="\t")); NP = len(R)
    gene = np.array([r["gene"] for r in R]); isc = np.array([int(float(r["is_canonical"])) for r in R])
    jobs = [(i, R[i]["chrom"], int(R[i]["center"]), R[i]["strand"]) for i in range(NP)]
    K = 6; OUT = np.full((NP, K), np.nan)
    log(f"[kinetic] scanning {NP} peaks +/-{SPAN} (donor>= {DTHR}, acceptor>= {A3THR}) ...")
    with Pool(32, initializer=_init) as p:
        for n, (i, r) in enumerate(p.imap_unordered(work, jobs, chunksize=32)):
            if r is not None: OUT[i] = r
            if (n + 1) % 5000 == 0: log(f"  {n+1}/{NP}")
    names = ["up_donor_dist", "up_donor_score", "dn_acc_dist", "dn_acc_score", "intron_len", "frac_pos"]
    d = {nm: OUT[:, k] for k, nm in enumerate(names)}
    d["is_canonical"] = isc; d["gene"] = gene
    np.savez("kinetic_scan_cache.npz", **d)
    for nm in names:
        v = d[nm]; log(f"  {nm:<16} finite {np.isfinite(v).mean():.3f}  median {np.nanmedian(v):.1f}")
    log("saved kinetic_scan_cache.npz  DONE_KINETIC_SCAN")
