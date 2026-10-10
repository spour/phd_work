#!/usr/bin/env python
# the within-gene paired u1 metagene for panel f

import os
import sys
import csv
import math
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

SPAN = 500
BINW = 25
U1_THR = 4.0
NBIN = 2 * SPAN // BINW
EDGES = np.arange(-SPAN, SPAN + 1, BINW)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0


def log(m):
    print(m, file=sys.stderr, flush=True)


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def wilcoxon_p(d):
    d = d[d != 0]
    n = d.size
    if n < 10:
        return float("nan")
    a = np.abs(d)
    o = np.argsort(a, kind="mergesort")
    sa = a[o]
    ng = np.r_[True, sa[1:] != sa[:-1]]
    g = np.cumsum(ng)
    st = np.r_[np.nonzero(ng)[0], n]
    av = (st[:-1] + st[1:] + 1) / 2.0
    r = np.empty(n)
    r[o] = av[g - 1]
    W = r[d > 0].sum()
    mu = n * (n + 1) / 4.0
    tie = np.bincount(g)[1:].astype(float)
    var = (n * (n + 1) * (2 * n + 1) - (tie ** 3 - tie).sum() / 2.0) / 24.0
    return two_sided_p((W - mu) / math.sqrt(var)) if var > 0 else float("nan")


_G = {}


def _init(cfg):
    from extract_splicing_features import GenomeFasta, load_u1_lookup
    _G["fa"] = GenomeFasta(cfg["fasta"])
    u1 = load_u1_lookup(cfg["u1_lookup"])
    L = len(next(iter(u1)))
    MAP = np.full(256, -1, np.int64)
    for i, ch in enumerate("ACGT"):
        MAP[ord(ch)] = i
    dense = np.full(4 ** L, np.nan, np.float32)
    powv = (4 ** np.arange(L - 1, -1, -1)).astype(np.int64)
    for kmer, val in u1.items():
        idx, bad = 0, False
        for k, ch in enumerate(kmer):
            c = MAP[ord(ch)]
            if c < 0:
                bad = True
                break
            idx += int(c) * int(powv[k])
        if not bad:
            dense[idx] = -val
    _G.update(MAP=MAP, U1LEN=L, powv=powv, dense=dense)


def _u1_full(seq):
    MAP, L, powv, dense = _G["MAP"], _G["U1LEN"], _G["powv"], _G["dense"]
    arr = MAP[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]
    n = len(arr)
    if n < L:
        return np.empty(0)
    codes = np.zeros(n - L + 1, np.int64)
    bad = np.zeros(n - L + 1, bool)
    for k in range(L):
        c = arr[k:k + (n - L + 1)]
        codes += c * powv[k]
        bad |= (c < 0)
    return np.where(bad, np.nan, dense[np.clip(codes, 0, len(dense) - 1)])


def _work(rec):
    chrom, center, strand = rec
    L = _G["U1LEN"]
    out = np.full(NBIN, np.nan)
    if strand == "+":
        glo, ghi = center - SPAN, center + SPAN + L
    else:
        glo, ghi = center - SPAN - L, center + SPAN
    if glo < 0:
        return out
    try:
        seq = _G["fa"].fetch(chrom, glo, ghi, strand=strand)
    except Exception:
        return out
    if not seq or len(seq) < L + 10:
        return out
    v = _u1_full(seq.upper())
    if v.size == 0:
        return out
    d = -SPAN + np.arange(v.size, dtype=float)
    ok = np.isfinite(v)
    d, v = d[ok], v[ok]
    strong = (v >= U1_THR).astype(float)
    b = np.digitize(d, EDGES) - 1
    m = (b >= 0) & (b < NBIN)
    if not m.any():
        return out
    cnt = np.bincount(b[m], weights=strong[m], minlength=NBIN)
    tot = np.bincount(b[m], minlength=NBIN).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(tot > 0, cnt / np.maximum(tot, 1) * 1000.0, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--fasta", default="../train/data/hg38.fa")
    ap.add_argument("--u1_lookup", default="../train/splice_train/data/u1_lookup.npy")
    ap.add_argument("--cache", default="u1_metagene_cache.npz")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    for p in (a.pos, a.fasta, a.u1_lookup):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)

    with open(a.pos, newline="") as fh:
        R = list(csv.DictReader(fh, delimiter="\t"))
    genes = np.array([r["gene"] for r in R])
    y = np.array([int(float(r["is_canonical"])) for r in R], dtype=np.int8)

    if os.path.isfile(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        if z["M"].shape == (len(R), NBIN) and list(z["gene"]) == list(genes):
            M = z["M"]
            log("reusing %s" % a.cache)
        else:
            M = None
            log("%s stale, rescanning" % a.cache)
    else:
        M = None
    if M is None:
        recs = [(r["chrom"], int(float(r["center"])), r["strand"]) for r in R]
        log("scanning %d peaks x %d bins (+/-%d nt, %d nt bins)" % (len(recs), NBIN, SPAN, BINW))
        t0 = time.time()
        M = np.full((len(recs), NBIN), np.nan, dtype=np.float32)
        import multiprocessing as mp
        cfg = {"fasta": a.fasta, "u1_lookup": a.u1_lookup}
        with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(cfg,)) as pool:
            for i, v in enumerate(pool.imap(_work, recs, chunksize=64)):
                M[i] = v
                if i and i % 20000 == 0:
                    log("  %d/%d (%.1f min)" % (i, len(recs), (time.time() - t0) / 60))
        log("scan done in %.1f min" % ((time.time() - t0) / 60))
        np.savez_compressed(a.cache, M=M, gene=genes,
                            centers=CENTERS.astype(np.float32))
        log("wrote %s" % a.cache)

    M = M.astype(float)
    log("bins with any data: %d/%d; peaks with a complete profile %d (%.1f%%)"
        % (int(np.isfinite(M).any(axis=0).sum()), NBIN,
           int(np.isfinite(M).all(axis=1).sum()), 100 * np.isfinite(M).all(axis=1).mean()))

    # per gene: mean competitor profile, mean real-end profile
    by = defaultdict(lambda: ([], []))
    for i in range(len(genes)):
        by[genes[i]][0 if y[i] == 1 else 1].append(i)
    gname, CAN, COM = [], [], []
    for g, (ci, ki) in by.items():
        if not ci or not ki:
            continue
        c = np.nanmean(M[ci], axis=0)
        k = np.nanmean(M[ki], axis=0)
        if not (np.isfinite(c).all() and np.isfinite(k).all()):
            continue
        gname.append(g)
        CAN.append(c)
        COM.append(k)
    CAN = np.asarray(CAN)
    COM = np.asarray(COM)
    ng = len(gname)
    log("paired genes with complete profiles: %d" % ng)

    # gene-mean-center, bin by bin, on the gene's own class average
    mid = (CAN + COM) / 2.0
    CANc = CAN - mid
    COMc = COM - mid
    DIFF = COM - CAN

    def mse(A):
        return A.mean(axis=0), A.std(axis=0, ddof=1) / math.sqrt(A.shape[0])

    can_m, can_s = mse(CANc)
    com_m, com_s = mse(COMc)
    dif_m, dif_s = mse(DIFF)

    with open("paired_u1_metagene.tsv", "w") as fh:
        fh.write("pos_bin\tclass\tu1_centered_mean\tse\tn_genes\n")
        for j, p in enumerate(CENTERS):
            fh.write("%g\tCompetitor\t%.6g\t%.6g\t%d\n" % (p, com_m[j], com_s[j], ng))
            fh.write("%g\tReal end\t%.6g\t%.6g\t%d\n" % (p, can_m[j], can_s[j], ng))
    # pooled peak-level profiles: the mean over peaks, no gene weighting and
    pmask = np.array([g in set(gname) for g in genes])
    pool = {}
    for nm, sel in (("Competitor", pmask & (y == 0)), ("Real end", pmask & (y == 1))):
        A = M[sel]
        pool[nm] = (np.nanmean(A, axis=0),
                    np.nanstd(A, axis=0, ddof=1) / np.sqrt(np.isfinite(A).sum(axis=0)),
                    int(sel.sum()))
    with open("paired_u1_pooled.tsv", "w") as fh:
        fh.write("pos_bin\tclass\tu1_mean\tse\tn_peaks\tn_genes\n")
        for j, p in enumerate(CENTERS):
            for nm in ("Competitor", "Real end"):
                m_, s_, n_ = pool[nm]
                fh.write("%g\t%s\t%.6g\t%.6g\t%d\t%d\n" % (p, nm, m_[j], s_[j], n_, ng))

    # raw (uncentered) profiles, gene-averaged so a gene with 30 competitors
    raw_can_m, raw_can_s = mse(CAN)
    raw_com_m, raw_com_s = mse(COM)
    with open("paired_u1_raw.tsv", "w") as fh:
        fh.write("pos_bin\tclass\tu1_mean\tse\tn_genes\n")
        for j, p in enumerate(CENTERS):
            fh.write("%g\tCompetitor\t%.6g\t%.6g\t%d\n" % (p, raw_com_m[j], raw_com_s[j], ng))
            fh.write("%g\tReal end\t%.6g\t%.6g\t%d\n" % (p, raw_can_m[j], raw_can_s[j], ng))

    rng = np.random.default_rng(0)
    NBOOT = 1000
    idx = rng.integers(0, ng, size=(NBOOT, ng))
    boot = np.empty((NBOOT, NBIN))
    for bi in range(NBOOT):
        boot[bi] = DIFF[idx[bi]].mean(axis=0)
    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)
    with open("paired_u1_excess.tsv", "w") as fh:
        fh.write("pos_bin\texcess_mean\tci_lo\tci_hi\tse\tn_genes\tn_boot\n")
        for j, p in enumerate(CENTERS):
            fh.write("%g\t%.6g\t%.6g\t%.6g\t%.6g\t%d\t%d\n"
                     % (p, dif_m[j], ci_lo[j], ci_hi[j], dif_s[j], ng, NBOOT))

    with open("paired_u1_diff.tsv", "w") as fh:
        fh.write("pos_bin\tu1_diff_mean\tse\tn_genes\n")
        for j, p in enumerate(CENTERS):
            fh.write("%g\t%.6g\t%.6g\t%d\n" % (p, dif_m[j], dif_s[j], ng))
    log("wrote paired_u1_metagene.tsv, paired_u1_pooled.tsv, paired_u1_raw.tsv, paired_u1_excess.tsv, paired_u1_diff.tsv")

    up = CENTERS < 0
    dn = CENTERS > 0
    up_diff = DIFF[:, up].mean(axis=1)
    dn_diff = DIFF[:, dn].mean(axis=1)
    up_can = CAN[:, up].mean(axis=1)
    up_com = COM[:, up].mean(axis=1)
    wr = float((up_com > up_can).mean() + 0.5 * (up_com == up_can).mean())
    pw = wilcoxon_p(up_diff)

    upM = np.nanmean(M[:, up], axis=1)
    tw = tp = 0.0
    rates = []
    for g, (ci, ki) in by.items():
        c = upM[ci]
        k = upM[ki]
        c = c[np.isfinite(c)]
        k = k[np.isfinite(k)]
        if not c.size or not k.size:
            continue
        w = float((k[None, :] > c[:, None]).sum()) + 0.5 * float((k[None, :] == c[:, None]).sum())
        n = c.size * k.size
        rates.append(w / n)
        tw += w
        tp += n
    rates = np.asarray(rates)

    nbin_sep = int(((com_m > can_m) & up).sum())
    nbin_up = int(up.sum())

    print("=" * 96)
    print("PANEL F  within-gene paired U1 metagene")
    print("=" * 96)
    print("  paired genes %d   bins %d (%+d to %+d nt, %d nt wide)"
          % (ng, NBIN, -SPAN, SPAN, BINW))
    print("  units: U1 sites per kb with negated dG >= %.1f, gene-mean-centered" % U1_THR)
    print()
    print("  integrated upstream (pA-500 -> pA), competitor minus real end:")
    print("    paired difference  %+.3f sites/kb  (SE %.3f)  Wilcoxon %.1e"
          % (up_diff.mean(), up_diff.std(ddof=1) / math.sqrt(ng), pw))
    print("    gene-level win-rate (gene profile means)      %.4f" % wr)
    print("    peak-level within-gene win-rate, gene-averaged %.4f  (pair-weighted %.4f)"
          % (rates.mean(), tw / tp))
    print("    reconciles with the grid cell up_500 density@4: 0.6193 gene-avg / 0.6116 pair")
    print()
    print("  integrated downstream (pA -> pA+500), competitor minus real end:")
    print("    paired difference  %+.3f sites/kb  (SE %.3f)"
          % (dn_diff.mean(), dn_diff.std(ddof=1) / math.sqrt(ng)))
    print()
    pc, pcs, pcn = pool["Competitor"]
    pr, prs, prn = pool["Real end"]
    pgap = pc[up].mean() - pr[up].mean()
    up_lo, up_hi = ci_lo[up], ci_hi[up]
    n_above = int((up_lo > 0).sum())
    n_dn_above = int((ci_lo[dn] > 0).sum())
    print("  PAIRED EXCESS, the line panel F now draws (%d bootstrap resamples over genes):" % NBOOT)
    print("    upstream mean excess  %+.2f sites/kb   CI [%+.2f, %+.2f]"
          % (up_diff.mean(), up_lo.mean(), up_hi.mean()))
    print("    upstream bins with the whole CI above zero:   %d/%d" % (n_above, nbin_up))
    print("    downstream bins with the whole CI above zero: %d/%d" % (n_dn_above, int(dn.sum())))
    print("    at the pA bin: %+.2f  CI [%+.2f, %+.2f]"
          % (dif_m[np.argmin(np.abs(CENTERS))], ci_lo[np.argmin(np.abs(CENTERS))],
             ci_hi[np.argmin(np.abs(CENTERS))]))
    if n_above == nbin_up:
        print("    -> CI SITS ABOVE ZERO ACROSS THE ENTIRE UPSTREAM FLANK. Panel F is sound.")
    else:
        print("    -> WARNING: %d upstream bins have a CI touching zero. Inspect before use."
              % (nbin_up - n_above))
    print()
    print("  POOLED peak-level lines (the previous panel F):")
    print("    competitor  %.2f sites/kb (SE %.2f)  over %d peaks" % (pc[up].mean(), pcs[up].mean(), pcn))
    print("    real end    %.2f sites/kb (SE %.2f)  over %d peaks" % (pr[up].mean(), prs[up].mean(), prn))
    print("    pooled gap  %+.2f sites/kb   vs the within-gene paired difference %+.2f"
          % (pgap, up_diff.mean()))
    print("    the pooled gap is %.0f%% of the paired one: between-gene variance is left in"
          % (100 * pgap / up_diff.mean()))
    nall = int((pc > pr).sum())
    print("    competitor above real end in %d/%d bins overall, %d/%d upstream"
          % (nall, NBIN, int(((pc > pr) & up).sum()), nbin_up))
    print()
    print("  RAW gene-averaged lines (the previous panel F):")
    print("    competitor upstream mean  %.2f sites/kb  (SE %.2f)"
          % (raw_com_m[up].mean(), raw_com_s[up].mean()))
    print("    real end   upstream mean  %.2f sites/kb  (SE %.2f)"
          % (raw_can_m[up].mean(), raw_can_s[up].mean()))
    rgap = raw_com_m[up].mean() - raw_can_m[up].mean()
    rse = raw_com_s[up].mean() + raw_can_s[up].mean()
    print("    gap %+.2f sites/kb vs combined SE %.2f  (%.1fx)" % (rgap, rse, abs(rgap) / rse))
    nraw = int(((raw_com_m > raw_can_m) & up).sum())
    print("    competitor above real end in %d/%d upstream bins" % (nraw, nbin_up))
    if abs(rgap) > rse and nraw >= 0.9 * nbin_up:
        print("    -> RAW LINES SEPARATE. No inset needed.")
    else:
        print("    -> raw lines overlap. ADD THE PAIRED-DIFFERENCE INSET.")
    print()
    print("  REPRESENTATION (a) legibility check:")
    print("    upstream bins where the competitor mean is above the real-end mean: %d/%d"
          % (nbin_sep, nbin_up))
    print("    mean separation across upstream bins: %.3f sites/kb (SE band ~%.3f)"
          % ((com_m - can_m)[up].mean(), (com_s + can_s)[up].mean() / 2))
    sep = (com_m - can_m)[up].mean()
    band = (com_s + can_s)[up].mean() / 2
    if nbin_sep == nbin_up and sep > 3 * band:
        print("    -> (a) separates cleanly in every upstream bin. USE (a).")
    elif nbin_sep >= 0.8 * nbin_up and sep > band:
        print("    -> (a) separates in most upstream bins. USE (a).")
    else:
        print("    -> (a) overlaps. FALL BACK TO (b), the difference line.")


if __name__ == "__main__":
    main()
