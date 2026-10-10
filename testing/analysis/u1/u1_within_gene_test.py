#!/usr/bin/env python3
# the panel-b data step of figure_splicing_environment
import sys, os, math, time
import numpy as np
from multiprocessing import Pool
import u1_5ss_landscape_data as L

log = L.log
WIN_LO, WIN_HI = -1000, 500
THR = 4.0
SPAN, PAD = 1000, 12
NBOOT, SEED = 1000, 42
CACHE = f"u1_dens{int(THR)}_w{-WIN_LO}_{WIN_HI}.npz"
TAG = f"[{WIN_LO},+{WIN_HI}]"

SITES = {}
def _init(): L.G_["G"].fh = open(L.FASTA, "rb")

def site_dens(ch, ctr, st):
    seq = L.G_["G"].fetch(ch, ctr - SPAN - PAD, ctr + SPAN + PAD, st)
    if not seq or len(seq) < 2 * (SPAN + PAD): return np.nan
    tok = L.B[np.frombuffer(seq.encode("ascii"), np.uint8)]
    sw = np.lib.stride_tricks.sliding_window_view(tok, 9); good = (sw >= 0).all(1)
    u1 = np.where(good, L.U1[np.where(good, sw @ L.POW, 0)], np.nan)
    d = -(SPAN + PAD) + np.arange(u1.shape[0]) + 3
    m = (d >= WIN_LO) & (d <= WIN_HI)
    return float(np.nansum(u1[m] >= THR) / ((WIN_HI - WIN_LO + 1) / 1000.0))

def _job(i): return site_dens(SITES["chrom"][i], int(SITES["center"][i]), SITES["strand"][i])

def wilcoxon(d):
    nz = d[d != 0]; n = len(nz); a = np.abs(nz)
    o = np.argsort(a, kind="stable"); sa = a[o]
    _, inv, cnt = np.unique(sa, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); mid = ((cs - cnt) + cs + 1) / 2.0
    rank = np.empty(n); rank[o] = mid[inv]
    V = float(rank[nz > 0].sum()); mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(cnt ** 3 - cnt)) / 48.0
    z = (V - mu + (0.5 if V < mu else -0.5)) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    logp = (math.log(2) - z * z / 2 - math.log(abs(z) * math.sqrt(2 * math.pi))) / math.log(10)
    return V, n, z, ("< 2.2e-16" if p < 2.2e-16 else f"{p:.3g}"), logp, len(d) - n

if __name__ == "__main__":
    S = L.build_sites(); NS = len(S["cls"])
    if os.path.exists(CACHE) and "--rescan" not in sys.argv:
        v = np.load(CACHE)["v"].astype(np.float64); assert len(v) == NS
        log(f"[scan] cache {CACHE}")
    else:
        SITES.update(chrom=S["chrom"], center=S["center"], strand=S["strand"])
        t = time.time(); v = np.empty(NS)
        with Pool(min(32, os.cpu_count() or 8), initializer=_init) as p:
            for i, x in enumerate(p.imap(_job, range(NS), chunksize=64)): v[i] = x
        log(f"[scan] RESCANNED -dG over {TAG} for {NS} sites in {time.time()-t:.0f}s "
            f"(the npz 100 nt U1 block holds mean -dG per bin, not thresholded counts)")
        np.savez_compressed(CACHE, v=v.astype(np.float32))
        v = v.astype(np.float32).astype(np.float64)

    P = L.gene_pairs(S["gene"], S["cls"]); genes = list(P.keys())
    d = []
    for g in genes:
        r, k = P[g]
        rv = v[r][np.isfinite(v[r])]; kv = v[k][np.isfinite(v[k])]
        d.append(rv.mean() - kv.mean() if rv.size and kv.size else np.nan)
    d = np.array(d); keep = np.isfinite(d)
    gk = [g for g, m in zip(genes, keep) if m]; d = d[keep]

    wr, lo, hi, ng = L.winrate(v, P, boot=NBOOT, seed=SEED)
    V, n, z, pstr, logp, nzero = wilcoxon(d)
    frac = float((d < 0).mean())
    log(f"[{TAG}] win-rate P(competitor > real) = {wr:.4f} [{lo:.4f},{hi:.4f}] over {ng} paired genes")
    log(f"[{TAG}] mean per-gene delta {d.mean():+.4f} /kb  median {np.median(d):+.4f} /kb  "
        f"{100*frac:.1f}% of genes left of 0")
    log(f"[{TAG}] Wilcoxon V = {V:.0f}, n = {n} ({nzero} zero deltas dropped), z = {z:+.1f}, "
        f"p {pstr} (log10 p ~ {logp:.0f})")
    log(f"[compare] previous window [-500,+250] was 0.660 / -19.5 /kb / 71%; recomputed above, not reused")

    with open("u1_gene_delta.tsv", "w") as fh:
        fh.write("gene\tdelta_per_kb\twindow\n")
        for g, x in zip(gk, d): fh.write(f"{g}\t{x:.5f}\t{TAG}\n")
    with open("u1_wilcoxon.tsv", "w") as fh:
        fh.write("key\tvalue\n")
        for k_, x in (("window", TAG), ("thr", THR), ("V", f"{V:.0f}"), ("n_genes", n),
                      ("n_paired_genes", ng), ("z", f"{z:.4f}"), ("p", pstr),
                      ("log10_p", f"{logp:.1f}"), ("winrate", f"{wr:.5f}"),
                      ("winrate_lo", f"{lo:.5f}"), ("winrate_hi", f"{hi:.5f}"),
                      ("mean_delta", f"{d.mean():.5f}"), ("median_delta", f"{np.median(d):.5f}"),
                      ("frac_below_zero", f"{frac:.5f}")):
            fh.write(f"{k_}\t{x}\n")
    log(f"[out] u1_gene_delta.tsv u1_wilcoxon.tsv {CACHE}")
