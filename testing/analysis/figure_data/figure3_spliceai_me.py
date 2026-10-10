#!/usr/bin/env python
# compute side for figure3_spliceai_me

import os
import sys
import csv
import time
import hashlib
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import pyBigWig
except ImportError:
    sys.exit("pyBigWig not installed. Run: pip install pyBigWig")

BASE_URL = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/"
FILES = {
    ("acceptor", "+"): "spliceAiAcceptorPlus.bw",
    ("acceptor", "-"): "spliceAiAcceptorMinus.bw",
    ("donor", "+"): "spliceAiDonorPlus.bw",
    ("donor", "-"): "spliceAiDonorMinus.bw",
}
BIGWIGS = {}
SIGNALS = ("acceptor", "donor")

UP = 2000
DN = 500
FULL = UP + DN
WINDOWS = (250, 500, 1000, 1500, 2000)

MG_EDGES = np.arange(-500, 501, 40)
MG_CTRS = (MG_EDGES[:-1] + MG_EDGES[1:]) / 2.0

MAXENT_ACC_BASELINE = 0.606
EXPECT_SA_ACC = 0.745
EXPECT_SA_DON = 0.675
TOL = 0.02

N_THREADS = 12
MIN_CELL = 200
N_DECILES = 10
HOLDOUT_SEED = 0


def log(m):
    print(m, file=sys.stderr, flush=True)


# auroc


def auroc(scores, labels):
    ok = ~np.isnan(scores)
    s, y = scores[ok], labels[ok]
    n = s.size
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    newgrp = np.r_[True, ss[1:] != ss[:-1]]
    grp = np.cumsum(newgrp)
    starts = np.r_[np.nonzero(newgrp)[0], n]
    avg = (starts[:-1] + starts[1:] + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[grp - 1]
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def read_peaks(path):
    rows = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        need = {"gene", "chrom", "center", "strand", "is_canonical"}
        missing = need - set(rdr.fieldnames or [])
        if missing:
            sys.exit("missing required columns: %s" % sorted(missing))
        for r in rdr:
            try:
                rows.append((r["gene"], r["chrom"], int(float(r["center"])),
                             r["strand"], int(float(r["is_canonical"]))))
            except (ValueError, TypeError):
                continue
    return rows


def read_tsv(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def open_bw(url, tries=4):
    for t in range(tries):
        try:
            return pyBigWig.open(url)
        except (RuntimeError, OSError):
            if t == tries - 1:
                return None
            time.sleep(2 ** t)
    return None


def oriented(bw, chrom, center, strand, clen):
    full = np.full(FULL, np.nan, dtype=np.float32)
    if strand == "+":
        lo, hi = center - UP, center + DN
    else:
        lo, hi = center - DN, center + UP
    lo_c, hi_c = max(0, lo), min(hi, clen)
    if hi_c <= lo_c:
        return full
    try:
        raw = np.asarray(bw.values(chrom, lo_c, hi_c), dtype=np.float32)
    except (RuntimeError, OverflowError):
        return full
    if strand == "+":
        m_lo = center + DN - hi_c
        full[m_lo:m_lo + raw.size] = raw[::-1]
    else:
        m_lo = lo_c - center + DN
        full[m_lo:m_lo + raw.size] = raw
    return full


def make_worker(peaks, feats, mg_sum, mg_cnt, mg_lock_free_slot):

    def run(job):
        slot, idxs = job
        handles = {}
        try:
            for k, i in enumerate(idxs):
                _g, chrom, center, strand, cls = peaks[i]
                for s_i, sig in enumerate(SIGNALS):
                    ent = handles.get((sig, strand))
                    if ent is None:
                        bw = open_bw(BIGWIGS[(sig, strand)])
                        if bw is None:
                            continue
                        ent = handles[(sig, strand)] = (bw, bw.chroms())
                    bw, sizes = ent
                    clen = sizes.get(chrom)
                    if clen is None:
                        continue
                    full = oriented(bw, chrom, center, strand, clen)
                    up = full[DN:]
                    for w_i, W in enumerate(WINDOWS):
                        seg = up[:W]
                        cov = ~np.isnan(seg)
                        if not cov.any():
                            continue
                        v = seg[cov]
                        col = s_i * len(WINDOWS) + w_i
                        feats[i, col] = v.mean() if sig == "acceptor" else v.sum()
                    good = ~np.isnan(full)
                    mg_sum[slot, s_i, cls][good] += full[good]
                    mg_cnt[slot, s_i, cls][good] += 1
                if k and k % 5000 == 0:
                    log("    slot %d: %d/%d" % (slot, k, len(idxs)))
        finally:
            for bw, _ in handles.values():
                bw.close()
        return len(idxs)

    return run


def scan(peaks):
    n = len(peaks)
    feats = np.full((n, len(SIGNALS) * len(WINDOWS)), np.nan, dtype=np.float64)
    order = sorted(range(n), key=lambda i: (peaks[i][1], peaks[i][2]))
    jobs = [(k, order[k::N_THREADS]) for k in range(N_THREADS)]
    jobs = [(k, c) for k, c in jobs if c]
    mg_sum = np.zeros((len(jobs), len(SIGNALS), 2, FULL), dtype=np.float64)
    mg_cnt = np.zeros((len(jobs), len(SIGNALS), 2, FULL), dtype=np.int64)
    done = 0
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for got in ex.map(make_worker(peaks, feats, mg_sum, mg_cnt, None), jobs):
            done += got
            log("  scanned %d/%d" % (done, n))
    return feats, mg_sum.sum(axis=0), mg_cnt.sum(axis=0)


def gene_split(genes):
    cache, out = {}, np.empty(len(genes), dtype=np.int8)
    for i, g in enumerate(genes):
        h = cache.get(g)
        if h is None:
            d = hashlib.md5(("%d:%s" % (HOLDOUT_SEED, g)).encode()).hexdigest()
            h = cache[g] = int(d[:8], 16) & 1
        out[i] = h
    return out


# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_ucube_meta.tsv")
    ap.add_argument("--indir", default=".",
                    help="dir holding the existing MaxEnt <prefix>_*.tsv files")
    ap.add_argument("--outdir", default=None,
                    help="where to write the _spliceai_me tsvs (default: --indir)")
    ap.add_argument("--prefix", default="fig3")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--threads", type=int, default=N_THREADS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    globals()["N_THREADS"] = max(1, args.threads)
    outdir = args.outdir or args.indir

    def pf(name):
        return os.path.join(args.indir, "%s_%s" % (args.prefix, name))

    n_local = 0
    for key, name in FILES.items():
        local = os.path.join(args.bw_dir, name)
        if not args.remote and os.path.isfile(local):
            BIGWIGS[key] = local
            n_local += 1
        else:
            BIGWIGS[key] = BASE_URL + name
    log("bigWigs: %d/%d local" % (n_local, len(FILES)))

    # maxent side: read, never recompute
    bca = read_tsv(pf("bcshow_auroc.tsv"))
    if bca is None:
        sys.exit("missing %s (MaxEnt panel D source)" % pf("bcshow_auroc.tsv"))
    cc = read_tsv(pf("class_counts.tsv"))
    n_can = n_comp = None
    if cc:
        for r in cc:
            if r.get("cls") == "Canonical":
                n_can = int(float(r["n"]))
            elif r.get("cls") == "Competitor":
                n_comp = int(float(r["n"]))

    def maxent_peak(mech):
        rows = [r for r in bca if r["mech"] == mech]
        if not rows:
            return float("nan"), float("nan")
        best = max(rows, key=lambda r: float(r["auroc"]))
        return float(best["auroc"]), float(best["auroc_raw"])
    me_don_or, me_don_raw = maxent_peak("donor")
    me_acc_or, _ = maxent_peak("acceptor")
    log("MaxEnt read: acceptor peak %.4f, donor peak %.4f oriented (%.4f raw)"
        % (me_acc_or, me_don_or, me_don_raw))

    # spliceai side: scan once
    peaks = read_peaks(args.table)
    if args.limit and args.limit < len(peaks):
        step = len(peaks) / float(args.limit)
        peaks = [peaks[int(i * step)] for i in range(args.limit)]
    y = np.array([p[4] for p in peaks], dtype=np.int8)
    split = gene_split([p[0] for p in peaks])
    log("%d peaks, scanning SpliceAI once ..." % len(peaks))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats, mg_sum, mg_cnt = scan(peaks)
    log("scan done in %.1f min" % ((time.time() - t0) / 60.0))

    i_acc2k = 0 * len(WINDOWS) + (len(WINDOWS) - 1)
    i_don2k = 1 * len(WINDOWS) + (len(WINDOWS) - 1)
    sa_acc = feats[:, i_acc2k]
    sa_don = feats[:, i_don2k]
    te = split == 1
    tr = split == 0

    a_acc_te = auroc(sa_acc[te], y[te])
    a_don_te = auroc(sa_don[te], y[te])
    n_drop_acc = int(np.isnan(sa_acc).sum())
    n_drop_don = int(np.isnan(sa_don).sum())

    log("SANITY")
    log("  peaks %d  canonical %d  competitor %d"
        % (len(peaks), int((y == 1).sum()), int((y == 0).sum())))
    log("  train %d (%d canon) / test %d (%d canon)"
        % (int(tr.sum()), int((y[tr] == 1).sum()),
           int(te.sum()), int((y[te] == 1).sum())))
    log("  dropped all-NaN: acceptor %d, donor %d" % (n_drop_acc, n_drop_don))
    log("  sa_acc mean-2kb AUROC (test) = %.4f  (expect ~%.3f)"
        % (a_acc_te, EXPECT_SA_ACC))
    log("  sa_don sum-2kb  AUROC (test) = %.4f  (expect ~%.3f)"
        % (a_don_te, EXPECT_SA_DON))
    if abs(a_acc_te - EXPECT_SA_ACC) > TOL:
        msg = ("acceptor test AUROC %.4f is off %.3f by more than %.2f"
               % (a_acc_te, EXPECT_SA_ACC, TOL))
        if args.limit:
            log("  WARNING (subsampled run, --limit %d): %s" % (args.limit, msg))
        else:
            sys.exit("STOP: %s. Array orientation or strand map is wrong." % msg)
    else:
        log("  -> orientation reproduces the locked grid winners.")
    if abs(a_don_te - EXPECT_SA_DON) > TOL:
        log("  WARNING: donor test AUROC %.4f drifts from %.3f."
            % (a_don_te, EXPECT_SA_DON))

    p1 = os.path.join(outdir, "auroc_window_spliceai_me.tsv")
    with open(p1, "w") as fh:
        fh.write("signal\treduction\twin_far\tauroc_test\n")
        for s_i, sig in enumerate(SIGNALS):
            red = "mean" if sig == "acceptor" else "sum"
            for w_i, W in enumerate(WINDOWS):
                a = auroc(feats[te, s_i * len(WINDOWS) + w_i], y[te])
                fh.write("%s\t%s\t%d\t%.4f\n" % (sig, red, W, a))

    if n_can is None or n_comp is None:
        n_can, n_comp = int((y == 1).sum()), int((y == 0).sum())
    b = n_can / float(n_can + n_comp)
    ok = ~np.isnan(sa_acc) & ~np.isnan(sa_don)

    def decile(v):
        edges = np.unique(np.quantile(v, np.linspace(0, 1, N_DECILES + 1)))
        idx = np.clip(np.searchsorted(edges, v, side="right") - 1,
                      0, max(0, len(edges) - 2))
        return idx, edges

    ab, aed = decile(sa_acc[ok])
    db, ded = decile(sa_don[ok])
    yo = y[ok]
    p2 = os.path.join(outdir, "acc_x_don_spliceai_me.tsv")
    n_cells = 0
    with open(p2, "w") as fh:
        fh.write("acc_bin_ctr\tacc_lo\tacc_hi\tdon_bin_ctr\tdon_lo\tdon_hi\t"
                 "n\tn_canonical\tp_canonical\tlog2_or\n")
        for i in range(max(1, len(aed) - 1)):
            for j in range(max(1, len(ded) - 1)):
                m = (ab == i) & (db == j)
                n = int(m.sum())
                if n < MIN_CELL:
                    continue
                nc = int((yo[m] == 1).sum())
                p = min(max(nc / float(n), 1e-4), 1 - 1e-4)
                lor = np.log2((p / (1 - p)) / (b / (1 - b)))
                fh.write("%d\t%.6g\t%.6g\t%d\t%.6g\t%.6g\t%d\t%d\t%.4f\t%.4f\n"
                         % (i + 1, aed[i], aed[i + 1], j + 1, ded[j], ded[j + 1],
                            n, nc, nc / float(n), lor))
                n_cells += 1

    p3 = os.path.join(outdir, "marginal_auroc_spliceai_me.tsv")
    with open(p3, "w") as fh:
        fh.write("metric\tauroc\tauroc_raw\tdirection\tsource\n")
        fh.write("maxent_acceptor\t%.4f\t%.4f\ttrue end\tpublished baseline\n"
                 % (MAXENT_ACC_BASELINE, MAXENT_ACC_BASELINE))
        fh.write("maxent_donor\t%.4f\t%.4f\tcompetitor\t%s_bcshow_auroc.tsv peak\n"
                 % (me_don_or, me_don_raw, args.prefix))
        fh.write("spliceai_acceptor\t%.4f\t%.4f\ttrue end\theld-out test\n"
                 % (a_acc_te, a_acc_te))
        fh.write("spliceai_donor\t%.4f\t%.4f\ttrue end (marginal)\theld-out test\n"
                 % (a_don_te, a_don_te))

    pos_of_m = DN - np.arange(FULL)
    keep = (pos_of_m >= MG_EDGES[0]) & (pos_of_m < MG_EDGES[-1])
    binidx = np.full(FULL, -1, dtype=np.int64)
    binidx[keep] = np.clip(
        ((pos_of_m[keep] - MG_EDGES[0]) // 40).astype(np.int64), 0, len(MG_CTRS) - 1)
    p4 = os.path.join(outdir, "metagene_spliceai_me.tsv")
    with open(p4, "w") as fh:
        fh.write("cls\tpos\tsa_acc_mean\tsa_don_mean\n")
        for cls, nm in ((1, "Canonical"), (0, "Competitor")):
            vals = {}
            for s_i, sig in enumerate(SIGNALS):
                s = np.bincount(binidx[keep], weights=mg_sum[s_i, cls][keep],
                                minlength=len(MG_CTRS))
                c = np.bincount(binidx[keep], weights=mg_cnt[s_i, cls][keep],
                                minlength=len(MG_CTRS))
                vals[sig] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            for k, ctr in enumerate(MG_CTRS):
                fh.write("%s\t%.1f\t%.6g\t%.6g\n"
                         % (nm, ctr, vals["acceptor"][k], vals["donor"][k]))

    log("WROTE")
    for p in (p1, p2, p3, p4):
        log("  %s" % p)
    log("panel E cells kept (>=%d peaks): %d" % (MIN_CELL, n_cells))
    log("base rate b = %.5f (n = %d canonical / %d competitor)" % (b, n_can, n_comp))


if __name__ == "__main__":
    main()
