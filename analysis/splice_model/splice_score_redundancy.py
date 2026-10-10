#!/usr/bin/env python3
# every measurement in figure_splicing_environment, from a single scan of one window
import sys, math, time
import numpy as np
from multiprocessing import Pool

def log(*a): print(*a, file=sys.stderr, flush=True)

import u1_5ss_landscape_data as L
import gc_metagene_data as GM
G_, U1, B, POW = L.G_, L.U1, L.B, L.POW
ME5 = G_["ME5"]; FASTA = L.FASTA
UP, DN, PAD = 2000, 500, 40
NOFF = UP + DN + 1
CLASSES = L.CLASSES
MBIN = 100
BEDGE = np.arange(-UP, DN + 1, MBIN); BCEN = (BEDGE[:-1] + BEDGE[1:]) / 2.0
NBIN = len(BCEN)
ND5 = 8.0
THR4 = 4.0
KB = NOFF / 1000.0

# score grids
# the coarse joint grid is the plotting resolution
U_LO, U_HI, U_W = -4.0, 12.0, 0.05
M_LO, M_HI, M_W = -50.0, 12.0, 0.25
JU_W, JM_W = 0.50, 1.00
NU = int(round((U_HI - U_LO) / U_W)); NM = int(round((M_HI - M_LO) / M_W))
JU = int(round((U_HI - U_LO) / JU_W)); JM = int(round((M_HI - M_LO) / JM_W))
UE = U_LO + U_W * np.arange(NU + 1); ME = M_LO + M_W * np.arange(NM + 1)
THR_U1 = (4.0, 7.0); THR_ME5 = (8.0,)
for t in THR_U1:
    assert np.isclose(UE, t).any() and np.isclose(U_LO + JU_W * np.arange(JU + 1), t).any(), t
for t in THR_ME5:
    assert np.isclose(ME, t).any() and np.isclose(M_LO + JM_W * np.arange(JM + 1), t).any(), t
GT = B[np.frombuffer(b"GT", np.uint8)]
BATCH = 250

SITES = {}

def _init():
    G_["G"].fh = open(FASTA, "rb")

def _blank():
    return [[np.zeros(JU * JM, np.int64), np.zeros(NU, np.int64),
             np.zeros(NM, np.int64), np.zeros(6)] for _ in (0, 1)]

def _add(A, r):
    for k in (0, 1):
        for j in range(4): A[k][j] += r[k][j]

def _one(i, A, out):
    ch = SITES["chrom"][i]; ctr = int(SITES["center"][i]); st = SITES["strand"][i]
    a, b = (ctr - UP - PAD, ctr + DN + PAD) if st == "+" else (ctr - DN - PAD + 1, ctr + UP + PAD + 1)
    seq = G_["G"].fetch(ch, a, b, st)
    if not seq or len(seq) != UP + DN + 2 * PAD: return 0
    tok = B[np.frombuffer(seq.encode("ascii"), np.uint8)]
    sw = np.lib.stride_tricks.sliding_window_view(tok, 9)
    d = -(UP + PAD) + np.arange(sw.shape[0]) + 3
    keep = (sw >= 0).all(1) & (d >= -UP) & (d <= DN)
    if not keep.any(): return 0
    w = sw[keep]; dk = d[keep]; code = w @ POW
    u = U1[code]; m = ME5[code]
    isgt = (w[:, 3] == GT[0]) & (w[:, 4] == GT[1])
    fin = np.isfinite(u) & np.isfinite(m)
    u, m, isgt, dk = u[fin], m[fin], isgt[fin], dk[fin]
    if not u.size: return 0
    assert u.min() >= U_LO and u.max() < U_HI and m.min() >= M_LO and m.max() < M_HI

    bi = np.digitize(dk, BEDGE) - 1
    ok = (bi >= 0) & (bi < NBIN)
    cnt = np.bincount(bi[ok], minlength=NBIN).astype(float)
    hit = np.bincount(bi[ok], weights=(m[ok] >= ND5).astype(float), minlength=NBIN)
    out[:NBIN] = np.where(cnt > 0, hit / np.maximum(cnt, 1) * 1000.0, np.nan)
    out[NBIN] = float((u >= THR4).sum()) / (len(u) / 1000.0)

    for k, (uu, mm) in enumerate(((u, m), (u[isgt], m[isgt]))):
        if not uu.size: continue
        A[k][0] += np.bincount(((uu - U_LO) / JU_W).astype(np.int64) * JM
                               + ((mm - M_LO) / JM_W).astype(np.int64), minlength=JU * JM)
        A[k][1] += np.bincount(((uu - U_LO) / U_W).astype(np.int64), minlength=NU)
        A[k][2] += np.bincount(((mm - M_LO) / M_W).astype(np.int64), minlength=NM)
        A[k][3] += (len(uu), uu.sum(), mm.sum(), (uu * uu).sum(), (mm * mm).sum(), (uu * mm).sum())
    return 1

def _job(b):
    cls, idx = b
    A = _blank(); R = np.full((len(idx), NBIN + 1), np.nan); ok = 0
    for j, i in enumerate(idx): ok += _one(i, A, R[j])
    return cls, A, R, np.asarray(idx), ok

def wilcoxon(d):
    nz = d[d != 0]; n = len(nz); aa = np.abs(nz)
    o = np.argsort(aa, kind="stable"); sa = aa[o]
    _, inv, cnt = np.unique(sa, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); mid = ((cs - cnt) + cs + 1) / 2.0
    rank = np.empty(n); rank[o] = mid[inv]
    V = float(rank[nz > 0].sum()); mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(cnt ** 3 - cnt)) / 48.0
    z = (V - mu + (0.5 if V < mu else -0.5)) / math.sqrt(var)
    pv = math.erfc(abs(z) / math.sqrt(2))
    logp = (math.log(2) - z * z / 2 - math.log(abs(z) * math.sqrt(2 * math.pi))) / math.log(10)
    return V, n, z, ("< 2.2e-16" if pv < 2.2e-16 else f"{pv:.3g}"), logp, len(d) - n

def pearson(mo):
    n, sx, sy, sxx, syy, sxy = mo
    if n < 2: return float("nan")
    cov = sxy / n - (sx / n) * (sy / n)
    return float(cov / math.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2)))

def exceed(h, edges):
    return np.r_[h[::-1].cumsum()[::-1], 0] / max(h.sum(), 1)

def at(E, edges, t):
    j = int(np.argmin(np.abs(edges - t))); return float(E[j])

if __name__ == "__main__":
    t0 = time.time()
    S = GM.build_sites(); n = len(S["cls"])
    EXP = {"real": 4847, "competitor": 43547, "decoy": 30000}
    for c in CLASSES:
        got = int((S["cls"] == c).sum())
        assert got == EXP[c], f"{c}: {got} sites, panels A/B were drawn on {EXP[c]}"
    log(f"sites {n}: " + "  ".join(f"{c} {int((S['cls']==c).sum())}" for c in CLASSES))
    SITES.update(S)
    BATCHES = [(c, list(ix[j:j + BATCH]))
               for c in CLASSES
               for ix in [np.flatnonzero(S["cls"] == c)]
               for j in range(0, len(ix), BATCH)]
    ACC = {c: _blank() for c in CLASSES}; nok = {c: 0 for c in CLASSES}
    SITE = np.full((n, NBIN + 1), np.nan)
    done = 0
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16, initializer=_init) as P:
        for cls, A, R, idx, ok in P.imap_unordered(_job, BATCHES):
            _add(ACC[cls], A); SITE[idx] = R; nok[cls] += ok; done += 1
            if done % 50 == 0: log(f"  batch {done}/{len(BATCHES)}  {time.time()-t0:.0f}s")
    log(f"scanned {sum(nok.values())} sites in {time.time()-t0:.0f}s  " +
        "  ".join(f"{c} {nok[c]}" for c in CLASSES))

    # panel a: strong-5'ss metagene over [-up, +dn]
    with open("splice_env_metagene.tsv", "w") as fh:
        fh.write("offset\tclass\tdonor_per_kb\tse\tn\n")
        for c in CLASSES:
            Msk = (S["cls"] == c) & np.isfinite(SITE[:, 0])
            Ab = SITE[np.ix_(Msk, np.arange(NBIN))]
            mu = np.nanmean(Ab, 0); se = np.nanstd(Ab, 0) / math.sqrt(int(Msk.sum()))
            for j, off in enumerate(BCEN):
                fh.write(f"{off:.0f}\t{c}\t{mu[j]:.5f}\t{se[j]:.5f}\t{int(Msk.sum())}\n")
    MG = {c: np.nanmean(SITE[np.ix_((S["cls"] == c) & np.isfinite(SITE[:, 0]),
                                    np.arange(NBIN))], 0) for c in CLASSES}
    jm50 = int(np.argmin(np.abs(BCEN + 50)))
    up = BCEN < 0; dn = BCEN > 0; up500 = (BCEN >= -500) & (BCEN < 0)
    log("")
    log(f"[A] window [-{UP},+{DN}]  {NBIN} bins of {MBIN} nt, hit = MaxEnt 5'ss >= {ND5:g}")
    for c in CLASSES:
        log(f"[A]   {c:<11} -50 bin {MG[c][jm50]:.4f}   upstream[-500,0) {MG[c][up500].mean():.4f}"
            f"   all upstream {MG[c][up].mean():.4f}   downstream(0,+{DN}] {MG[c][dn].mean():.4f} /kb")

    # panel b: within-gene u1 density difference
    v = SITE[:, NBIN]
    PR = L.gene_pairs(S["gene"], S["cls"]); genes = list(PR.keys())
    dd = []
    for g in genes:
        r, k = PR[g]
        rv = v[r][np.isfinite(v[r])]; kv = v[k][np.isfinite(v[k])]
        dd.append(rv.mean() - kv.mean() if rv.size and kv.size else np.nan)
    dd = np.array(dd); keep = np.isfinite(dd)
    gk = [g for g, m in zip(genes, keep) if m]; dd = dd[keep]
    V, nw, z, pstr, logp, nzero = wilcoxon(dd)
    frac = float((dd < 0).mean())
    with open("splice_env_gene_delta.tsv", "w") as fh:
        fh.write("gene\tdelta_per_kb\twindow\n")
        for g, x in zip(gk, dd): fh.write(f"{g}\t{x:.5f}\t[-{UP},+{DN}]\n")
    with open("splice_env_wilcoxon.tsv", "w") as fh:
        fh.write("key\tvalue\n")
        for k_, x in (("window", f"[-{UP},+{DN}]"), ("thr", f"{THR4:g}"), ("V", f"{V:.0f}"),
                      ("n_genes", nw), ("n_paired_genes", len(dd)), ("n_zero_delta", nzero),
                      ("z", f"{z:.4f}"), ("p", pstr), ("log10_p", f"{logp:.1f}"),
                      ("mean_delta", f"{dd.mean():.5f}"), ("median_delta", f"{np.median(dd):.5f}"),
                      ("frac_below_zero", f"{frac:.5f}")):
            fh.write(f"{k_}\t{x}\n")
    log(f"[B] window [-{UP},+{DN}]  U1 sites (-dG >= {THR4:g}) per kb, real minus mean competitor")
    log(f"[B]   {len(dd)} paired genes   mean {dd.mean():+.3f}   median {np.median(dd):+.3f} /kb"
        f"   {100*frac:.1f}% below zero")
    log(f"[B]   Wilcoxon V = {V:.0f}, n = {nw} ({nzero} zero deltas dropped), z = {z:+.4f}, "
        f"p {pstr} (log10 p ~ {logp:.1f})")

    POOL = _blank()
    for c in CLASSES: _add(POOL, ACC[c])
    EU, EM = exceed(POOL[0][1], UE), exceed(POOL[0][2], ME)
    EUg, EMg = exceed(POOL[1][1], UE), exceed(POOL[1][2], ME)

    rows = []
    def put(k, v, nn=""): rows.append((k, v, nn))
    for lab, k in (("all", 0), ("gt", 1)):
        put(f"r_{lab}_pooled", f"{pearson(POOL[k][3]):.5f}", int(POOL[k][3][0]))
        for c in CLASSES: put(f"r_{lab}_{c}", f"{pearson(ACC[c][k][3]):.5f}", int(ACC[c][k][3][0]))
        put(f"n_pos_{lab}", int(POOL[k][3][0]), int(POOL[k][3][0]))
    for c in CLASSES: put(f"n_sites_{c}", nok[c], nok[c])
    put("window", f"[-{UP},+{DN}]")
    for t in THR_U1:
        put(f"exceed_u1_{t:g}", f"{at(EU,UE,t):.6f}", int(POOL[0][3][0]))
        put(f"exceed_gt_u1_{t:g}", f"{at(EUg,UE,t):.6f}", int(POOL[1][3][0]))
    for t in THR_ME5:
        put(f"exceed_me5_{t:g}", f"{at(EM,ME,t):.6f}", int(POOL[0][3][0]))
        put(f"exceed_gt_me5_{t:g}", f"{at(EMg,ME,t):.6f}", int(POOL[1][3][0]))
    put("frac_positions_gt", f"{POOL[1][3][0]/POOL[0][3][0]:.6f}", int(POOL[0][3][0]))
    for c in CLASSES:
        put(f"exceed_u1_4_{c}", f"{at(exceed(ACC[c][0][1],UE),UE,4.0):.6f}", nok[c])
        put(f"exceed_me5_8_{c}", f"{at(exceed(ACC[c][0][2],ME),ME,8.0):.6f}", nok[c])
    with open("splice_score_stats.tsv", "w") as fh:
        fh.write("key\tvalue\tn\n")
        for k, v, nn in rows: fh.write(f"{k}\t{v}\t{nn}\n")

    with open("splice_score_exceedance.tsv", "w") as fh:
        fh.write("score\tclass\tthreshold\tfrac\n")
        for score, ei, edges in (("u1_dg", 1, UE), ("me5", 2, ME)):
            for c in CLASSES + ("pooled",):
                h = POOL[0][ei] if c == "pooled" else ACC[c][0][ei]
                for t, f in zip(edges, exceed(h, edges)):
                    fh.write(f"{score}\t{c}\t{t:.4f}\t{f:.8f}\n")

    uc = U_LO + JU_W * (np.arange(JU) + .5); mc = M_LO + JM_W * (np.arange(JM) + .5)
    with open("splice_score_joint.tsv", "w") as fh:
        fh.write("subset\tu1_dg\tme5\tcount\n")
        for lab, k in (("all", 0), ("gt", 1)):
            H = POOL[k][0].reshape(JU, JM)
            for i in range(JU):
                for j in range(JM):
                    if H[i, j]: fh.write(f"{lab}\t{uc[i]:.3f}\t{mc[j]:.3f}\t{H[i,j]}\n")

    log(f"[r]  all positions  pooled {pearson(POOL[0][3]):.4f}   " +
        "  ".join(f"{c} {pearson(ACC[c][0][3]):.4f}" for c in CLASSES))
    log(f"[r]  GT-anchored    pooled {pearson(POOL[1][3]):.4f}   " +
        "  ".join(f"{c} {pearson(ACC[c][1][3]):.4f}" for c in CLASSES))
    log(f"[n]  {int(POOL[0][3][0]):,} scored 9-mer positions, "
        f"{int(POOL[1][3][0]):,} GT-anchored ({100*POOL[1][3][0]/POOL[0][3][0]:.2f}%)")
    for t in THR_U1:
        log(f"[thr] U1 -dG >= {t:g}: {100*at(EU,UE,t):.3f}% of positions, "
            f"{100*at(EUg,UE,t):.2f}% of GT-anchored positions")
    for t in THR_ME5:
        log(f"[thr] MaxEnt 5'ss >= {t:g}: {100*at(EM,ME,t):.4f}% of positions, "
            f"{100*at(EMg,ME,t):.3f}% of GT-anchored positions")
    log("[out] splice_env_metagene.tsv splice_env_gene_delta.tsv splice_env_wilcoxon.tsv "
        "splice_score_joint.tsv splice_score_exceedance.tsv splice_score_stats.tsv")
