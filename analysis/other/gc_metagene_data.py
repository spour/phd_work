#!/usr/bin/env python3
# base composition across the cleavage site: gc fraction per position
import sys, os, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L

log = L.log
UP, DN, BIN = 2000, 500, 50
NBIN = (UP + DN) // BIN
WINDOWS = ((-500, 0, "upstream [-500,0)"),
           (0, 500, "downstream [0,+500)"),
           (200, 500, "far downstream [+200,+500)"))
NW = len(WINDOWS)
SEED, NBOOT, DEC_SUB = 42, 1000, 30000
CODE = np.full(256, -1, np.int8)
for ch, v in zip(b"ACGTacgt", (0, 1, 2, 3, 0, 1, 2, 3)): CODE[ch] = v

# sites: analysis set + the same seed-42 decoys
def build_sites():
    from fastparquet import ParquetFile
    T = ParquetFile(L.PARQ).to_pandas(columns=["gene", "chrom", "center", "strand",
                                               "in_analysis_set"])
    assert len(T) == L.NP
    key = dict(zip(zip(T.gene, T.chrom, T.center.astype(int), T.strand), T.in_analysis_set.values))
    assert len(key) == L.NP, "peak key not unique"
    IN = np.array([key[(L.gene[i], L.chrom[i], int(L.center[i]), L.strand[i])]
                   for i in range(L.NP)], bool)
    pk = np.flatnonzero(IN)
    cls = np.where(L.isc[pk] == 1, "real", "competitor")
    log(f"analysis set: {(cls=='real').sum()} real + {(cls=='competitor').sum()} competitor "
        f"in {len(set(L.gene[pk]))} genes")

    D = np.load("decoy_set.npz"); dpos = D["pos"]; dgene = D["gene"].astype(int)
    gi = np.load("u1_window_cache.npz")["gene_idx"]
    o = np.argsort(gi, kind="stable"); gs = gi[o]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    gcanon = []
    for j in range(len(cut) - 1):
        idx = o[cut[j]:cut[j+1]]; cc = idx[L.isc[idx] == 1]; kk = idx[L.isc[idx] == 0]
        if len(cc) == 1 and len(kk) >= 1: gcanon.append(int(cc[0]))
    gcanon = np.array(gcanon)
    dci = gcanon[dgene]
    pool = np.flatnonzero(IN[dci])
    rng = np.random.default_rng(SEED)
    ds = pool[rng.choice(len(pool), min(DEC_SUB, len(pool)), replace=False)]
    log(f"decoys {len(dpos)} total, {len(pool)} in analysis-set genes, {len(ds)} sampled "
        f"(seed {SEED})")
    return dict(
        chrom=np.r_[np.array(L.chrom)[pk], np.array(L.chrom)[dci[ds]]],
        center=np.r_[np.array(L.center)[pk], dpos[ds]].astype(np.int64),
        strand=np.r_[np.array(L.strand)[pk], np.array(L.strand)[dci[ds]]],
        gene=np.r_[L.gene[pk], L.gene[dci[ds]]],
        cls=np.r_[cls, np.full(len(ds), "decoy")])

SITES = {}
def _init(): L.G_["G"].fh = open(L.FASTA, "rb")

def _one(i):
    ch = SITES["chrom"][i]; c = int(SITES["center"][i]); st = SITES["strand"][i]
    a, b = (c - UP, c + DN) if st == "+" else (c - DN + 1, c + UP + 1)
    s = L.G_["G"].fetch(ch, a, b, st)
    out = np.full(NBIN + NW, np.nan)
    if not s or len(s) != UP + DN: return out
    t = CODE[np.frombuffer(s.encode("ascii"), np.uint8)]
    ok = t >= 0; gc = (t == 1) | (t == 2)
    n = np.add.reduceat(ok.astype(np.float64), np.arange(0, UP + DN, BIN))
    g = np.add.reduceat((gc & ok).astype(np.float64), np.arange(0, UP + DN, BIN))
    out[:NBIN] = np.where(n > 0, g / np.maximum(n, 1), np.nan)
    for k, (lo, hi, _) in enumerate(WINDOWS):
        sl = slice(lo + UP, hi + UP)
        nn = ok[sl].sum()
        out[NBIN + k] = (gc[sl] & ok[sl]).sum() / nn if nn else np.nan
    return out

# stats, inline
def wilcoxon(d):
    d = d[np.isfinite(d)]; d = d[d != 0]; n = len(d)
    if n < 6: return float("nan"), float("nan")
    a = np.abs(d); o = np.argsort(a, kind="stable"); a = a[o]; sg = np.sign(d[o])
    r = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j+1] == a[i]: j += 1
        r[i:j+1] = (i + j + 2) / 2.0; i = j + 1
    W = float(r[sg > 0].sum()); mu = n * (n + 1) / 4.0
    _, cnt = np.unique(a, return_counts=True); tie = float((cnt**3 - cnt).sum())
    sd = np.sqrt(n*(n+1)*(2*n+1)/24.0 - tie/48.0)
    z = (W - mu - 0.5*np.sign(W - mu)) / sd
    from math import erfc, sqrt
    return z, erfc(abs(z)/sqrt(2))

if __name__ == "__main__":
    S = build_sites(); SITES.update(S)
    N = len(S["cls"])
    G = L.G_["G"]; rc = str.maketrans("ACGTacgt", "TGCAtgca"); bad = 0; checked = 0
    for i in np.random.default_rng(0).choice(N, 200, replace=False):
        c = int(S["center"][i]); st = S["strand"][i]
        b0 = G.fetch(S["chrom"][i], c, c + 1, "+")
        a, b = (c - UP, c + DN) if st == "+" else (c - DN + 1, c + UP + 1)
        seq = G.fetch(S["chrom"][i], a, b, st)
        if not b0 or not seq or len(seq) != UP + DN: continue
        checked += 1
        want = b0 if st == "+" else b0.translate(rc)
        if seq[UP].upper() != want.upper(): bad += 1
    assert bad == 0, f"orientation check failed on {bad}/{checked}"
    log(f"[check] transcript offset 0 == base at center on {checked} random sites: OK")

    t0 = time.time()
    with Pool(min(32, os.cpu_count() or 8), initializer=_init) as p:
        M = np.array(p.map(_one, range(N), chunksize=64))
    log(f"scanned {N} sites in {time.time()-t0:.0f}s; "
        f"{int(np.isfinite(M[:,:NBIN]).all(1).sum())} complete profiles")

    cls = S["cls"]; gsel = S["gene"]
    offs = np.arange(NBIN) * BIN - UP + BIN // 2
    with open("gc_metaprofile.tsv", "w") as fh:
        fh.write("offset\tclass\tgc\tse\tn\n")
        for c in ("real", "competitor", "decoy"):
            m = cls == c
            for j in range(NBIN):
                v = M[m, j]; v = v[np.isfinite(v)]
                fh.write(f"{offs[j]}\t{c}\t{v.mean():.6f}\t{v.std(ddof=1)/np.sqrt(len(v)):.6f}\t{len(v)}\n")
    log("")
    def band(lo, hi):
        j = slice((lo + UP)//BIN, (hi + UP)//BIN)
        return {c: float(np.nanmean(M[cls == c, j])) for c in ("real","competitor","decoy")}
    for lab, (lo, hi) in (("[-100,0)", (-100, 0)), ("[+200,+300)", (200, 300))):
        b = band(lo, hi)
        log(f"[GC] {lab:12s} real {b['real']:.4f}  competitor {b['competitor']:.4f}  "
            f"decoy {b['decoy']:.4f}   (real - competitor {b['real']-b['competitor']:+.4f})")

    # within-gene, confound-free
    by = {}
    for i in range(N):
        if cls[i] == "decoy": continue
        r, k = by.setdefault(gsel[i], ([], []))
        (r if cls[i] == "real" else k).append(i)
    genes = [g for g, (r, k) in by.items() if r and k]
    log(f"\nwithin-gene: {len(genes)} genes with both a real end and >=1 competitor")
    rows = []
    with open("gc_within_gene.tsv", "w") as fh:
        fh.write("gene\twindow\tgc_real\tgc_competitor_mean\tdelta\tn_competitors\n")
        for wi, (_, _, wname) in enumerate(WINDOWS):
            col = NBIN + wi
            d = []; wr = []
            for g in genes:
                r, k = by[g]
                vr = np.nanmean(M[r, col]); vk = np.nanmean(M[k, col])
                if not (np.isfinite(vr) and np.isfinite(vk)): continue
                kk = M[k, col]; kk = kk[np.isfinite(kk)]
                wr.append(((kk > vr).sum() + 0.5*(kk == vr).sum()) / len(kk))
                d.append(vr - vk)
                fh.write(f"{g}\t{wname}\t{vr:.6f}\t{vk:.6f}\t{vr-vk:.6f}\t{len(kk)}\n")
            d = np.array(d); wr = np.array(wr)
            z, p = wilcoxon(d)
            rs = np.random.default_rng(SEED)
            B = rs.integers(0, len(wr), size=(NBOOT, len(wr)))
            lo, hi = np.percentile(wr[B].mean(1), [2.5, 97.5])
            rows.append((wname, len(d), wr.mean(), lo, hi, d.mean(), np.median(d), z, p,
                         float((d < 0).mean())))
            log(f"[within-gene] {wname:22s} n={len(d)}  win-rate P(competitor GC > real GC) "
                f"{wr.mean():.4f} [{lo:.4f},{hi:.4f}]  mean delta {d.mean():+.5f}  "
                f"median {np.median(d):+.5f}  frac(real more AT-rich) {float((d<0).mean()):.3f}  "
                f"Wilcoxon z={z:+.2f} p={p:.3g}")
    with open("gc_within_gene_stats.tsv", "w") as fh:
        fh.write("window\tn_genes\twinrate_comp_gt_real\tci_lo\tci_hi\tmean_delta\t"
                 "median_delta\twilcoxon_z\twilcoxon_p\tfrac_real_more_at_rich\n")
        for r in rows:
            fh.write("\t".join(str(x) if isinstance(x, (int, str)) else f"{x:.6g}" for x in r) + "\n")
    for c in ("real", "competitor", "decoy"):
        log(f"[n] {c}: {int((cls==c).sum())}")
    log("[out] gc_metaprofile.tsv gc_within_gene.tsv gc_within_gene_stats.tsv")
