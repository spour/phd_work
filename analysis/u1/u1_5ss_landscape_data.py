#!/usr/bin/env python3
# data for a figure characterising u1 and the donor 5'ss around competitors and real 3'
import sys, os, math, time
import numpy as np
from multiprocessing import Pool

SRC = "task_v32_analyses.py"; MARK = "# ---- +/-2kb donor scan for all peaks ----"
G_ = {"__name__": "__main__"}
exec(compile(open(SRC).read().split(MARK)[0], SRC + ":header", "exec"), G_)

gene = G_["gene"]; isc = G_["isc"]; chrom = G_["chrom"]; center = G_["center"]; strand = G_["strand"]
NP = G_["NP"]; FASTA = G_["FASTA"]; B = G_["B"]; POW = G_["POW"]; ME5 = G_["ME5"]
PARQ = "cpa_peaks_final.parquet"
U1NPY = "../train/splice_train/data/u1_lookup.npy"

SPAN, PAD = 2000, 40
MBIN, WBIN = 100, 25
WSPAN = 500
SCAL_LO, SCAL_HI = -500, 250
ND5, THR4, THR7 = 8.0, 4.0, 7.0
ME3_LO, ME3_HI = -2000, -126
ND3 = 8.0
DN_FLANK = 300
DEC_SUB, SEED = 30000, 42
NBOOT = 1000
CLASSES = ("real", "competitor", "decoy")

def log(*a): print(*a, file=sys.stderr, flush=True)

# dense u1 (-dg), same 4^9 footing as the me5 array
_a = np.load(U1NPY, allow_pickle=True); _u1d = _a.item() if _a.dtype == object else dict(_a)
_ks = np.frombuffer("".join(_u1d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
U1 = np.full(4 ** 9, np.nan)
U1[B[_ks] @ POW] = -np.fromiter(_u1d.values(), np.float64, len(_u1d))
log(f"u1 lookup {len(_u1d)} 9-mers, -dG in [{np.nanmin(U1):.2f},{np.nanmax(U1):.2f}]")

BE100 = np.arange(-SPAN, SPAN + 1, MBIN);   BC100 = (BE100[:-1] + BE100[1:]) / 2.0
BE25  = np.arange(-WSPAN, WSPAN + 1, WBIN); BC25  = (BE25[:-1] + BE25[1:]) / 2.0
N100, N25 = len(BC100), len(BC25)
O_U1_100, O_U1_25, O_D5_100 = 0, N100, N100 + N25
O_S = O_D5_100 + N100
K = {k: O_S + i for i, k in enumerate(
    ["u1_dens4", "u1_dens7", "me5_dens8", "u1_mean_up", "d5_up", "d5_dn",
     "me3_max_up", "anchor_off", "iv_donor", "dn_donor_300", "ok"])}
NFEAT = O_S + len(K)

def _binmean(d, v, edges, nb):
    bi = np.digitize(d, edges) - 1; m = (bi >= 0) & (bi < nb) & np.isfinite(v)
    s = np.bincount(bi[m], weights=v[m], minlength=nb)
    c = np.bincount(bi[m], minlength=nb).astype(float)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)

def _bindens(d, hit, edges, nb):
    bi = np.digitize(d, edges) - 1; m = (bi >= 0) & (bi < nb)
    s = np.bincount(bi[m], weights=hit[m].astype(float), minlength=nb)
    c = np.bincount(bi[m], minlength=nb).astype(float)
    return np.where(c > 0, s / np.maximum(c, 1) * 1000.0, np.nan)

def site_features(ch, ctr, st):
    out = np.full(NFEAT, np.nan); out[K["ok"]] = 0.0
    seq = G_["G"].fetch(ch, ctr - SPAN - PAD, ctr + SPAN + PAD, st)
    if not seq or len(seq) < 2 * (SPAN + PAD): return out
    tok = B[np.frombuffer(seq.encode("ascii"), np.uint8)]
    sw = np.lib.stride_tricks.sliding_window_view(tok, 9); good = (sw >= 0).all(1)
    code = np.where(good, sw @ POW, 0)
    me5 = np.where(good, ME5[code], -99.0)
    u1 = np.where(good, U1[code], np.nan)
    d = -(SPAN + PAD) + np.arange(me5.shape[0]) + 3
    keep = (d >= -SPAN) & (d <= SPAN)
    d, me5, u1 = d[keep], me5[keep], u1[keep]
    hit5 = me5 >= ND5

    out[O_U1_100:O_U1_100 + N100] = _binmean(d, u1, BE100, N100)
    out[O_U1_25:O_U1_25 + N25]    = _binmean(d, u1, BE25, N25)
    out[O_D5_100:O_D5_100 + N100] = _bindens(d, hit5, BE100, N100)

    w = (d >= SCAL_LO) & (d <= SCAL_HI); nkb = (SCAL_HI - SCAL_LO + 1) / 1000.0
    out[K["u1_dens4"]]  = np.nansum(u1[w] >= THR4) / nkb
    out[K["u1_dens7"]]  = np.nansum(u1[w] >= THR7) / nkb
    out[K["me5_dens8"]] = hit5[w].sum() / nkb
    up = (d >= -WSPAN) & (d < 0)
    out[K["u1_mean_up"]] = np.nanmean(u1[up]) if up.any() else np.nan

    hu = d[hit5 & (d < 0)]; hd = d[hit5 & (d > 0)]
    out[K["d5_up"]] = -hu.max() if hu.size else np.nan
    out[K["d5_dn"]] =  hd.min() if hd.size else np.nan
    out[K["dn_donor_300"]] = float(((hd > 0) & (hd <= DN_FLANK)).any()) if hd.size else 0.0

    from maxentpy import maxent
    lo = ME3_LO - 19 + (SPAN + PAD); hi = ME3_HI - 19 + (SPAN + PAD)
    best = -99.0; bestpos = np.nan
    for i in range(max(0, lo), min(len(seq) - 23 + 1, hi)):
        if seq[i + 18] != "A" or seq[i + 19] != "G": continue
        k = seq[i:i + 23]
        if "N" in k: continue
        try: v = maxent.score3(k, matrix=M3)
        except Exception: continue
        if v > best: best = v; bestpos = i + 19 - (SPAN + PAD)
    out[K["me3_max_up"]] = best
    out[K["anchor_off"]] = bestpos
    out[K["iv_donor"]] = (float(((hu > bestpos) & (hu < 0)).any()) if hu.size and np.isfinite(bestpos)
                          else 0.0)
    out[K["ok"]] = 1.0
    return out

M3 = None
def _init():
    global M3
    from maxentpy.maxent import load_matrix3
    M3 = load_matrix3()
    G_["G"].fh = open(FASTA, "rb")

SITES = {}
def _job(i):
    return site_features(SITES["chrom"][i], int(SITES["center"][i]), SITES["strand"][i])

# site list
def build_sites():
    import pyarrow.parquet as pq
    T = pq.read_table(PARQ).to_pandas()
    assert len(T) == NP
    key = dict(zip(zip(T.gene, T.chrom, T.center.astype(int), T.strand), T.in_analysis_set.values))
    assert len(key) == NP, "peak key not unique"
    IN = np.array([key[(gene[i], chrom[i], int(center[i]), strand[i])] for i in range(NP)], bool)
    pk = np.flatnonzero(IN)
    cls = np.where(isc[pk] == 1, "real", "competitor")
    log(f"analysis set: {(cls=='real').sum()} real + {(cls=='competitor').sum()} competitor "
        f"in {len(set(gene[pk]))} genes")

    D = np.load("decoy_set.npz"); dpos = D["pos"]; dgene = D["gene"].astype(int)
    gi = np.load("u1_window_cache.npz")["gene_idx"]
    o = np.argsort(gi, kind="stable"); gs = gi[o]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)]
    gcanon = []
    for j in range(len(cut) - 1):
        idx = o[cut[j]:cut[j + 1]]; cc = idx[isc[idx] == 1]; kk = idx[isc[idx] == 0]
        if len(cc) == 1 and len(kk) >= 1: gcanon.append(int(cc[0]))
    gcanon = np.array(gcanon)
    dci = gcanon[dgene]
    pool = np.flatnonzero(IN[dci])
    rng = np.random.default_rng(SEED)
    ds = pool[rng.choice(len(pool), min(DEC_SUB, len(pool)), replace=False)]
    log(f"decoys {len(dpos)} total, {len(pool)} in analysis-set genes, {len(ds)} sampled (seed {SEED})")

    return dict(
        chrom=np.r_[np.array(chrom)[pk], np.array(chrom)[dci[ds]]],
        center=np.r_[np.array(center)[pk], dpos[ds]].astype(np.int64),
        strand=np.r_[np.array(strand)[pk], np.array(strand)[dci[ds]]],
        gene=np.r_[gene[pk], gene[dci[ds]]],
        cls=np.r_[cls, np.full(len(ds), "decoy")],
        peak_row=np.r_[pk, np.full(len(ds), -1)])

# stats
def gene_pairs(gsel, csel):
    by = {}
    for i, (g, c) in enumerate(zip(gsel, csel)):
        if c == "decoy": continue
        r, k = by.setdefault(g, ([], []))
        (r if c == "real" else k).append(i)
    return {g: (np.array(r), np.array(k)) for g, (r, k) in by.items() if r and k}

def winrate(v, P, boot=NBOOT, seed=SEED):
    per = []
    for r, k in P.values():
        rv = v[r]; kv = v[k]
        rv = rv[np.isfinite(rv)]; kv = kv[np.isfinite(kv)]
        if not rv.size or not kv.size: continue
        w = (kv[None, :] > rv[:, None]).sum() + 0.5 * (kv[None, :] == rv[:, None]).sum()
        per.append(w / (rv.size * kv.size))
    per = np.array(per)
    rs = np.random.default_rng(seed)
    bs = per[rs.integers(0, len(per), size=(boot, len(per)))].mean(1)
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(per)

def paired_delta(v, P, boot=NBOOT, seed=SEED):
    d = []
    for r, k in P.values():
        rv = v[r]; kv = v[k]
        rv = rv[np.isfinite(rv)]; kv = kv[np.isfinite(kv)]
        if not rv.size or not kv.size: continue
        d.append(rv.mean() - kv.mean())
    d = np.array(d); rs = np.random.default_rng(seed)
    bs = d[rs.integers(0, len(d), size=(boot, len(d)))].mean(1)
    return d, float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan,) * 3
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h

def ecdf_on(v, grid):
    v = np.sort(v[np.isfinite(v)])
    return np.searchsorted(v, grid, side="right") / max(len(v), 1)

# panels
def panel_A(F, S, out="u1_metagene.tsv"):
    rows = []
    for c in CLASSES:
        m = (S["cls"] == c) & (F[:, K["ok"]] == 1)
        A = F[np.ix_(m, np.arange(O_U1_100, O_U1_100 + N100))]
        mu = np.nanmean(A, 0); se = np.nanstd(A, 0) / math.sqrt(m.sum())
        for j, off in enumerate(BC100): rows.append((off, c, mu[j], se[j], int(m.sum())))
    with open(out, "w") as fh:
        fh.write("offset\tclass\tu1_dg\tse\tn\n")
        for r in rows: fh.write(f"{r[0]:.0f}\t{r[1]}\t{r[2]:.5f}\t{r[3]:.5f}\t{r[4]}\n")
    pa = {c: [r[2] for r in rows if r[1] == c and r[0] == -50][0] for c in CLASSES}
    log("[A] pA bin (-50 nt) mean U1 -dG: " + "  ".join(f"{c} {pa[c]:.3f}" for c in CLASSES)
        + f"   ordering {' < '.join(sorted(pa, key=pa.get))}")
    return pa

def panel_B(F, S, P, out="u1_withingene.tsv"):
    cols = np.arange(O_U1_25, O_U1_25 + N25)
    R = np.full((len(P), N25), np.nan); C = np.full((len(P), N25), np.nan)
    for i, (r, k) in enumerate(P.values()):
        R[i] = np.nanmean(F[np.ix_(r, cols)], 0); C[i] = np.nanmean(F[np.ix_(k, cols)], 0)
    ctr = (R + C) / 2.0
    Rc, Cc = R - ctr, C - ctr
    n = len(P)
    rows = []
    for j, off in enumerate(BC25):
        for nm, A in (("real", Rc), ("competitor", Cc)):
            rows.append((off, nm, np.nanmean(A[:, j]), np.nanstd(A[:, j]) / math.sqrt(n), n))
        dj = C[:, j] - R[:, j]
        rows.append((off, "difference", np.nanmean(dj), np.nanstd(dj) / math.sqrt(n), n))
    with open(out, "w") as fh:
        fh.write("offset\tseries\tvalue\tse\tn_genes\n")
        for r in rows: fh.write(f"{r[0]:.0f}\t{r[1]}\t{r[2]:.5f}\t{r[3]:.5f}\t{r[4]}\n")
    up = BC25 < 0
    dif = np.nanmean(C[:, up], 1) - np.nanmean(R[:, up], 1)
    rs = np.random.default_rng(SEED)
    bs = dif[rs.integers(0, len(dif), size=(NBOOT, len(dif)))].mean(1)
    wr = winrate(F[:, K["u1_mean_up"]], P)
    log(f"[B] integrated upstream (-500..0) paired difference competitor - real = "
        f"{dif.mean():+.4f} kcal/mol [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] "
        f"over {len(dif)} genes;  within-gene win-rate P(comp>real) = {wr[0]:.4f} "
        f"[{wr[1]:.4f},{wr[2]:.4f}]")
    return dict(diff=dif.mean(), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                wr=wr[0], wr_lo=wr[1], wr_hi=wr[2], n=len(dif))

def panel_C(F, S, P, out="u1_scalar.tsv"):
    grid = np.arange(0, 401, 4.0)
    rows = []
    st = {}
    for thr, col in (("4", "u1_dens4"), ("7", "u1_dens7")):
        v = F[:, K[col]]
        for c in CLASSES:
            m = (S["cls"] == c) & (F[:, K["ok"]] == 1)
            e = ecdf_on(v[m], grid)
            for x, f in zip(grid, e): rows.append(("ecdf", thr, c, x, f, int(m.sum())))
        st[thr] = dict(zip(("wr", "wr_lo", "wr_hi", "ng"), winrate(v, P)))
        d, dm, dlo, dhi = paired_delta(v, P)
        st[thr].update(delta=dm, dlo=dlo, dhi=dhi)
        if thr == "4":
            lo, hi = np.percentile(d, [0.5, 99.5])
            edges = np.linspace(lo, hi, 41); cnt, _ = np.histogram(d, bins=edges)
            for j in range(len(cnt)):
                rows.append(("delta", thr, "real_minus_comp", (edges[j] + edges[j + 1]) / 2,
                             float(cnt[j]), len(d)))
    with open(out, "w") as fh:
        fh.write("kind\tthr\tclass\tx\tvalue\tn\n")
        for r in rows: fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.5f}\t{r[5]}\n")
    for thr, lab in (("4", "U1-favorable density"), ("7", "positioned U1 sites")):
        s = st[thr]
        log(f"[C] thr-{thr} ({lab}): win-rate P(comp>real) = {s['wr']:.4f} [{s['wr_lo']:.4f},{s['wr_hi']:.4f}], "
            f"paired delta real-comp = {s['delta']:+.3f} /kb [{s['dlo']:+.3f},{s['dhi']:+.3f}], {s['ng']} genes")
    return st

def panel_D(F, S, out="nearest5ss.tsv"):
    grid = np.arange(0, 2001, 20.0)
    rows = []; med = {}
    for side, col in (("upstream", "d5_up"), ("downstream", "d5_dn")):
        for c in CLASSES:
            m = (S["cls"] == c) & (F[:, K["ok"]] == 1)
            v = F[m, K[col]]
            n = int(m.sum()); nf = int(np.isfinite(v).sum())
            e = np.searchsorted(np.sort(v[np.isfinite(v)]), grid, side="right") / max(n, 1)
            for x, f in zip(grid, e): rows.append((side, c, x, f, n, nf))
            fin = np.sort(v[np.isfinite(v)])
            med[(side, c)] = fin[int(0.5 * n)] if nf > n / 2 else np.nan
    with open(out, "w") as fh:
        fh.write("side\tclass\tdistance\tecdf\tn\tn_with_site\n")
        for r in rows: fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.0f}\t{r[3]:.5f}\t{r[4]}\t{r[5]}\n")
    log("[D] median distance to nearest strong 5'ss (censored at 2 kb):")
    for side in ("upstream", "downstream"):
        log("    " + side.ljust(11) + "  ".join(
            f"{c} {med[(side,c)]:.0f} nt" if np.isfinite(med[(side, c)]) else f"{c} >2000 nt"
            for c in CLASSES))
    return med

def panel_E(F, S, P, out="donor_flanked.tsv"):
    rows = []; res = {}
    for c in CLASSES:
        m = (S["cls"] == c) & (F[:, K["ok"]] == 1)
        acc = F[m, K["me3_max_up"]] >= ND3
        iv = F[m, K["iv_donor"]] == 1
        dn = F[m, K["dn_donor_300"]] == 1
        for nm, v in (("acceptor", acc), ("intervening_donor", iv), ("donor_dn300", dn),
                      ("donor_flanked_intervening", acc & iv), ("donor_flanked_dn300", acc & dn)):
            p, lo, hi = wilson(int(v.sum()), len(v))
            rows.append((c, nm, p, lo, hi, int(v.sum()), len(v)))
            res[(c, nm)] = (p, lo, hi)
    with open(out, "w") as fh:
        fh.write("class\tproperty\tfrac\tci_lo\tci_hi\tk\tn\n")
        for r in rows: fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.5f}\t{r[3]:.5f}\t{r[4]:.5f}\t{r[5]}\t{r[6]}\n")
    acc_all = (F[:, K["me3_max_up"]] >= ND3).astype(float)
    wr = winrate(acc_all * (F[:, K["iv_donor"]] == 1), P)
    wrd = winrate(acc_all * (F[:, K["dn_donor_300"]] == 1), P)
    log("[E] donor-flanked fractions (Wilson 95%):")
    for c in CLASSES:
        p, lo, hi = res[(c, "donor_flanked_intervening")]
        q = res[(c, "donor_flanked_dn300")]
        a = res[(c, "acceptor")]; iv = res[(c, "intervening_donor")]
        log(f"    {c:11s} acceptor {a[0]:.4f}  intervening donor {iv[0]:.4f}  "
            f"acceptor+intervening {p:.4f} [{lo:.4f},{hi:.4f}]  acceptor+dn300 {q[0]:.4f} [{q[1]:.4f},{q[2]:.4f}]")
    log(f"    within-gene win-rate P(comp>real): acceptor+intervening {wr[0]:.4f} [{wr[1]:.4f},{wr[2]:.4f}]   "
        f"acceptor+dn300 {wrd[0]:.4f} [{wrd[1]:.4f},{wrd[2]:.4f}]")
    return res, wr, wrd

def panel_F(F, S, out="u1_vs_5ss.tsv"):
    m = F[:, K["ok"]] == 1
    x = F[m, K["u1_dens4"]]; y = F[m, K["me5_dens8"]]
    x7 = F[m, K["u1_dens7"]]
    g = np.isfinite(x) & np.isfinite(y); x, y = x[g], y[g]
    r = float(np.corrcoef(x, y)[0, 1])
    nkb = (SCAL_HI - SCAL_LO + 1) / 1000.0
    yc = np.rint(y * nkb).astype(int)
    xe = np.linspace(np.percentile(x, 0.2), np.percentile(x, 99.8), 49)
    ymax = int(np.percentile(yc, 99.9))
    H, _, _ = np.histogram2d(x, np.minimum(yc, ymax), bins=[xe, np.arange(-0.5, ymax + 1.5)])
    with open(out, "w") as fh:
        fh.write("x\ty\tcount\tr\tn\n")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i, j] > 0:
                    fh.write(f"{(xe[i]+xe[i+1])/2:.3f}\t{j}\t{int(H[i,j])}\t{r:.5f}\t{len(x)}\n")
    g7 = np.isfinite(x7) & np.isfinite(F[m, K["me5_dens8"]])
    r7 = float(np.corrcoef(x7[g7], F[m, K["me5_dens8"]][g7])[0, 1])
    rr = {}
    for c in CLASSES:
        mm = (S["cls"][m][g] == c)
        rr[c] = float(np.corrcoef(x[mm], y[mm])[0, 1])
    log(f"[F] Pearson r over [{SCAL_LO},{SCAL_HI}]: U1 -dG>=4 density vs strong-5'ss density = {r:.4f} "
        f"(n={len(x)});  U1 -dG>=7 (positioned) vs strong-5'ss = {r7:.4f};  per class(thr4) "
        + "  ".join(f"{c} {rr[c]:.3f}" for c in CLASSES))
    return r, r7, rr

if __name__ == "__main__":
    CACHE = "u1_5ss_sitescan.npz"
    S = build_sites()
    NS = len(S["cls"])
    if os.path.exists(CACHE) and "--rescan" not in sys.argv:
        Z = np.load(CACHE, allow_pickle=True); F = Z["F"]
        assert len(F) == NS and Z["nfeat"] == NFEAT, "cache stale, rerun with --rescan"
        log(f"scan cache {CACHE} ({F.shape})")
    else:
        SITES.update(chrom=S["chrom"], center=S["center"], strand=S["strand"])
        t = time.time(); F = np.empty((NS, NFEAT), np.float64)
        with Pool(min(32, os.cpu_count() or 8), initializer=_init) as p:
            for i, v in enumerate(p.imap(_job, range(NS), chunksize=32)):
                F[i] = v
                if (i + 1) % 20000 == 0: log(f"  scan {i+1}/{NS}  {time.time()-t:.0f}s")
        log(f"scan done {NS} sites  {time.time()-t:.0f}s  ok {int((F[:,K['ok']]==1).sum())}")
        np.savez_compressed(CACHE, F=F.astype(np.float32), nfeat=NFEAT,
                            cls=S["cls"], gene=S["gene"], center=S["center"])
        F = F.astype(np.float32).astype(np.float64)

    accp = np.array([float(r) if r not in ("", "nan") else np.nan
                     for r in (x["acc"] for x in G_["R"])])
    pr = S["peak_row"]; mm = pr >= 0
    a = accp[pr[mm]]; b = F[mm, K["me3_max_up"]]
    ok = np.isfinite(a) & np.isfinite(b) & (b > -50)
    log(f"[check] maxent3 upstream max vs pipeline `acc`: r = {np.corrcoef(a[ok],b[ok])[0,1]:.4f}, "
        f"median |diff| = {np.median(np.abs(a[ok]-b[ok])):.3f} bits, n = {int(ok.sum())}")

    P = gene_pairs(S["gene"], S["cls"])
    log(f"paired genes (>=1 real and >=1 competitor): {len(P)}")

    pa = panel_A(F, S)
    b = panel_B(F, S, P)
    c = panel_C(F, S, P)
    d = panel_D(F, S)
    e, ewr, ewrd = panel_E(F, S, P)
    r, r7, rr = panel_F(F, S)

    n = {cl: int(((S["cls"] == cl) & (F[:, K["ok"]] == 1)).sum()) for cl in CLASSES}
    with open("u1_5ss_stats.tsv", "w") as fh:
        fh.write("key\tvalue\tlo\thi\tn\n")
        for cl in CLASSES:
            fh.write(f"n_{cl}\t{n[cl]}\t\t\t{n[cl]}\n")
            fh.write(f"pA_u1_{cl}\t{pa[cl]:.5f}\t\t\t{n[cl]}\n")
        fh.write(f"n_paired_genes\t{len(P)}\t\t\t{len(P)}\n")
        fh.write(f"B_upstream_diff\t{b['diff']:.5f}\t{b['lo']:.5f}\t{b['hi']:.5f}\t{b['n']}\n")
        fh.write(f"B_winrate\t{b['wr']:.5f}\t{b['wr_lo']:.5f}\t{b['wr_hi']:.5f}\t{b['n']}\n")
        for thr in ("4", "7"):
            s = c[thr]
            fh.write(f"C_winrate_thr{thr}\t{s['wr']:.5f}\t{s['wr_lo']:.5f}\t{s['wr_hi']:.5f}\t{s['ng']}\n")
            fh.write(f"C_delta_thr{thr}\t{s['delta']:.5f}\t{s['dlo']:.5f}\t{s['dhi']:.5f}\t{s['ng']}\n")
        for side in ("upstream", "downstream"):
            for cl in CLASSES:
                v = d[(side, cl)]
                fh.write(f"D_median_{side}_{cl}\t{'' if not np.isfinite(v) else f'{v:.1f}'}\t\t\t{n[cl]}\n")
        for cl in CLASSES:
            for nm in ("acceptor", "intervening_donor", "donor_flanked_intervening", "donor_flanked_dn300"):
                p, lo, hi = e[(cl, nm)]
                fh.write(f"E_{nm}_{cl}\t{p:.5f}\t{lo:.5f}\t{hi:.5f}\t{n[cl]}\n")
        fh.write(f"E_winrate_intervening\t{ewr[0]:.5f}\t{ewr[1]:.5f}\t{ewr[2]:.5f}\t{ewr[3]}\n")
        fh.write(f"E_winrate_dn300\t{ewrd[0]:.5f}\t{ewrd[1]:.5f}\t{ewrd[2]:.5f}\t{ewrd[3]}\n")
        fh.write(f"F_pearson_r\t{r:.5f}\t\t\t{sum(n.values())}\n")
        fh.write(f"F_pearson_r_thr7\t{r7:.5f}\t\t\t{sum(n.values())}\n")
        for cl in CLASSES: fh.write(f"F_r_{cl}\t{rr[cl]:.5f}\t\t\t{n[cl]}\n")
    log("\n[n] " + "  ".join(f"{cl} {n[cl]}" for cl in CLASSES) + f"   paired genes {len(P)}")
    log("[out] u1_metagene.tsv u1_withingene.tsv u1_scalar.tsv nearest5ss.tsv "
        "donor_flanked.tsv u1_vs_5ss.tsv u1_5ss_stats.tsv " + CACHE)
