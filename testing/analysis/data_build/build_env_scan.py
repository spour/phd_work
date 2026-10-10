#!/usr/bin/env python3
# fresh maxent genome scan for the 8,977 success peaks (canonical
import sys, csv, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "scripts")
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA   = "../train/data/hg38.fa"
ME5NPY  = "../train/splice_train/data/me5_lookup.npy"
U1NPY   = "../train/splice_train/data/u1_lookup.npy"
LAB     = "cpa_peaks_labeled.tsv"
POS     = "fig3_new_positional_features.tsv"
A3THR   = 6.5
DTHR    = 8.46
ACC_WIN = 2000
U1THR   = 5.25
U1WIN   = 500
PAD     = 25
WINS    = [1000, 2000, 5000, 10000, 20000]
SPAN    = 20000

# dense 5'ss (== score5)
B = np.full(256, -1, np.int64)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): B[ch] = v
POW9 = (4 ** np.arange(8, -1, -1)).astype(np.int64)
_a = np.load(ME5NPY, allow_pickle=True); me5d = _a.item() if _a.dtype == object else dict(_a)
_ks = np.frombuffer("".join(me5d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
ME5 = np.full(4 ** 9, -99.0); ME5[B[_ks] @ POW9] = np.fromiter(me5d.values(), np.float64, len(me5d))
# dense u1 (negate at load: higher = stronger, matches figure3_data)
_u = np.load(U1NPY, allow_pickle=True); u1d = _u.item() if _u.dtype == object else dict(_u)
_uk = np.frombuffer("".join(u1d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
U1 = np.full(4 ** 9, -99.0); U1[B[_uk] @ POW9] = -np.fromiter(u1d.values(), np.float64, len(u1d))

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
            if a < 0 or b > Ln: return None
            bp = lambda p: off + (p // lb) * lw + (p % lb)
            s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
            q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
            return q.translate(s._T)[::-1] if st == "-" else q
    _G = Faidx(FASTA)

def scan(chrom, center, strand):
    from maxentpy import maxent
    a0 = center - (SPAN + PAD); b0 = center + (SPAN + PAD)
    seq = _G.fetch(chrom, a0, b0, strand)
    if not seq or len(seq) < 2 * (SPAN + PAD): return None
    arr = np.frombuffer(seq.encode("ascii"), np.uint8)
    tok = B[arr]
    sw = np.lib.stride_tricks.sliding_window_view(tok, 9); good = (sw >= 0).all(1)
    sc5 = np.full(sw.shape[0], -99.0); sc5[good] = ME5[sw[good] @ POW9]
    o5 = (-(SPAN + PAD)) + np.arange(sc5.size) + 3
    o5s = o5[sc5 >= DTHR]
    best5 = float(sc5[(o5 >= -ACC_WIN) & (o5 <= 0)].max()) if ((o5 >= -ACC_WIN) & (o5 <= 0)).any() else np.nan
    u1v = np.full(sw.shape[0], -99.0); u1v[good] = U1[sw[good] @ POW9]
    ou = (-(SPAN + PAD)) + np.arange(u1v.size) + 3
    umask = (ou >= -U1WIN) & (ou < 0)
    u1_525 = float((u1v[umask] >= U1THR).sum())
    L3 = len(seq) - 22
    isA = (arr[18:18 + L3] == 65); isG = (arr[19:19 + L3] == 71)
    agidx = np.where(isA & isG)[0]
    o3all = (-(SPAN + PAD)) + agidx + 20
    keep = []; acc = -99.0
    for j, ix in enumerate(agidx):
        km = seq[ix:ix + 23]
        if "N" in km: continue
        s3 = maxent.score3(km, matrix=_m3)
        if -(ACC_WIN - 20) <= o3all[j] <= 0 and s3 > acc: acc = s3
        if s3 >= A3THR: keep.append(o3all[j])
    o3s = np.array(keep, float) if keep else np.empty(0)
    up3 = [int(((o3s >= -w) & (o3s < 0)).sum()) for w in WINS]
    dn3 = [int(((o3s > 0) & (o3s <= w)).sum()) for w in WINS]
    up5 = [int(((o5s >= -w) & (o5s < 0)).sum()) for w in WINS]
    dn5 = [int(((o5s > 0) & (o5s <= w)).sum()) for w in WINS]
    return dict(acc=(acc if acc > -99 else np.nan), best5=best5, u1_525=u1_525,
                up3=up3, dn3=dn3, up5=up5, dn5=dn5)

def work(job):
    i, chrom, center, strand = job
    r = scan(chrom, center, strand)
    return i, r

def run(jobs, tag):
    NP = len(jobs); NW = len(WINS)
    acc = np.full(NP, np.nan); best5 = np.full(NP, np.nan); u1 = np.full(NP, np.nan)
    U3 = np.full((NP, NW), np.nan); D3 = np.full((NP, NW), np.nan)
    U5 = np.full((NP, NW), np.nan); D5 = np.full((NP, NW), np.nan)
    log(f"[{tag}] scanning {NP} peaks +/-{SPAN} ...")
    with Pool(32, initializer=_init) as p:
        for n, (i, r) in enumerate(p.imap_unordered(work, jobs, chunksize=32)):
            if r is not None:
                acc[i] = r["acc"]; best5[i] = r["best5"]; u1[i] = r["u1_525"]
                U3[i] = r["up3"]; D3[i] = r["dn3"]; U5[i] = r["up5"]; D5[i] = r["dn5"]
            if (n + 1) % 2000 == 0: log(f"  {tag} {n+1}/{NP}")
    out = dict(acc=acc, best5=best5, u1_525=u1)
    for k, w in enumerate(WINS):
        wk = w // 1000
        out[f"n3ss_up_{wk}"] = U3[:, k]; out[f"n3ss_dn_{wk}"] = D3[:, k]
        out[f"n5ss_up_{wk}"] = U5[:, k]; out[f"n5ss_dn_{wk}"] = D5[:, k]
    return out

if __name__ == "__main__":
    R = list(csv.DictReader(open(LAB), delimiter="\t"))
    succ = [(r["gene"], r["chrom"], int(r["center"]), r["strand"])
            for r in R if r["is_canonical"] == "True" and r["contested"] == "False"]
    log(f"SUCCESS peaks: {len(succ)}")
    sjobs = [(i, c, ce, st) for i, (g, c, ce, st) in enumerate(succ)]
    sout = run(sjobs, "success")
    sout["gene"] = np.array([g for g, c, ce, st in succ])
    sout["chrom"] = np.array([c for g, c, ce, st in succ])
    sout["center"] = np.array([ce for g, c, ce, st in succ])
    sout["strand"] = np.array([st for g, c, ce, st in succ])
    np.savez("success_scan_cache.npz", **sout)
    log("saved success_scan_cache.npz")

    # verification arm: 500 contested peaks vs cached acc + asym
    P = list(csv.DictReader(open(POS), delimiter="\t"))
    cached_acc = {(r["gene"], r["chrom"], int(r["center"]), r["strand"]): float(r["acc"])
                  for r in P if r["acc"] not in ("", "nan")}
    asym = np.load("asym_real_cache.npz")
    pkey = [(r["gene"], r["chrom"], int(r["center"]), r["strand"]) for r in P]
    key2row = {k: i for i, k in enumerate(pkey)}
    rng = np.random.default_rng(7)
    sel = rng.choice(len(P), size=500, replace=False)
    vjobs = [(i, P[s]["chrom"], int(P[s]["center"]), P[s]["strand"]) for i, s in enumerate(sel)]
    vout = run(vjobs, "verify")
    my_acc = vout["acc"]; my_u3 = vout["n3ss_up_5"]; my_d3 = vout["n3ss_dn_5"]
    ca = np.array([cached_acc.get((P[s]["gene"], P[s]["chrom"], int(P[s]["center"]), P[s]["strand"]), np.nan) for s in sel])
    cu3 = np.array([asym["n3ss_up_5"][key2row[pkey[s]]] for s in sel])
    cd3 = np.array([asym["n3ss_dn_5"][key2row[pkey[s]]] for s in sel])
    m = np.isfinite(my_acc) & np.isfinite(ca)
    log(f"[verify] acc: corr {np.corrcoef(my_acc[m], ca[m])[0,1]:.5f}  "
        f"max|diff| {np.abs(my_acc[m]-ca[m]).max():.4f}  n={m.sum()}")
    m3 = np.isfinite(my_u3) & np.isfinite(cu3)
    log(f"[verify] n3ss_up5: identical {(my_u3[m3]==cu3[m3]).mean():.4f}  corr {np.corrcoef(my_u3[m3],cu3[m3])[0,1]:.5f}")
    log(f"[verify] n3ss_dn5: identical {(my_d3[m3]==cd3[m3]).mean():.4f}  corr {np.corrcoef(my_d3[m3],cd3[m3])[0,1]:.5f}")
    np.savez("verify_scan_cache.npz", my_acc=my_acc, cached_acc=ca, my_u3=my_u3, cu3=cu3, my_d3=my_d3, cd3=cd3)
    log("DONE_ENV_SCAN")
