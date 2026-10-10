#!/usr/bin/env python
# grid-search the discriminative capacity of spliceai donor (5'ss) and acceptor

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

BASE = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/"
FILES = {
    ("acceptor", "+"): "spliceAiAcceptorPlus.bw",
    ("acceptor", "-"): "spliceAiAcceptorMinus.bw",
    ("donor", "+"): "spliceAiDonorPlus.bw",
    ("donor", "-"): "spliceAiDonorMinus.bw",
}
BIGWIGS = {}
SIGNALS = ("acceptor", "donor")

SPAN = 2000
THRESHOLDS = (0.2, 0.5, 0.8)
WINDOWS = [
    (0, 250), (0, 500), (0, 1000), (0, 1500), (0, 2000),
    (0, 100), (100, 300), (300, 600), (600, 1200),
]
EXPANDING = sorted((w for w in WINDOWS if w[0] == 0), key=lambda w: w[1])

BASE_MAXENT_ACC = 0.606
BASE_SPLICEAI_ACC_MAX_2KB = 0.737
BASE_SPLICEAI_DON_MAX_2KB = 0.648

N_THREADS = 12
MIN_CELL = 200
HOLDOUT_SEED = 0


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# encoding grid


def build_columns():
    cols = []
    for sig in SIGNALS:
        for near, far in WINDOWS:
            for red in ("max", "mean", "sum"):
                cols.append((sig, near, far, red, None))
            for red in ("density_per_kb", "frac_above"):
                for t in THRESHOLDS:
                    cols.append((sig, near, far, red, t))
    return cols


COLUMNS = build_columns()
COL_INDEX = {c: i for i, c in enumerate(COLUMNS)}


def reduce_window(arr, near, far, out, base):
    w = arr[near:far]
    cov = ~np.isnan(w)
    n_cov = int(cov.sum())
    if n_cov == 0:
        out[base:base + 9] = np.nan
        return
    v = w[cov]
    kb = (far - near) / 1000.0
    out[base + 0] = v.max()
    out[base + 1] = v.mean()
    out[base + 2] = v.sum()
    for j, t in enumerate(THRESHOLDS):
        c = float((v >= t).sum())
        out[base + 3 + j] = c / kb
        out[base + 6 + j] = c / n_cov
    return


# auroc


def auroc(scores, labels):
    ok = ~np.isnan(scores)
    s = scores[ok]
    y = labels[ok]
    n = s.size
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    newgrp = np.r_[True, ss[1:] != ss[:-1]]
    grp = np.cumsum(newgrp)
    starts = np.r_[np.nonzero(newgrp)[0], n]
    avg = (starts[:-1] + starts[1:] + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[grp - 1]
    rsum = ranks[y == 1].sum()
    return (rsum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), n


def read_peaks(path):
    rows = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        need = {"gene", "chrom", "center", "strand", "is_canonical", "acc"}
        missing = need - set(rdr.fieldnames or [])
        if missing:
            sys.exit("missing required columns: %s" % sorted(missing))
        for r in rdr:
            try:
                rows.append((r["gene"], r["chrom"], int(float(r["center"])),
                             r["strand"], int(float(r["is_canonical"])),
                             float(r["acc"])))
            except (ValueError, TypeError):
                continue
    return rows


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
    if strand == "+":
        start, end = max(0, center - SPAN), center
    else:
        start, end = center, min(center + SPAN, clen)
    if end <= start:
        return np.full(SPAN, np.nan, dtype=np.float32)
    try:
        v = bw.values(chrom, start, end)
    except (RuntimeError, OverflowError):
        return np.full(SPAN, np.nan, dtype=np.float32)
    a = np.asarray(v, dtype=np.float32)
    if strand == "+":
        a = a[::-1]
    if a.size < SPAN:
        a = np.concatenate([a, np.full(SPAN - a.size, np.nan, dtype=np.float32)])
    return a


def make_worker(peaks, feats):
    n_per_sig = len(COLUMNS) // len(SIGNALS)

    def run(idxs):
        handles = {}
        try:
            for n, i in enumerate(idxs):
                _gene, chrom, center, strand, _y, _acc = peaks[i]
                row = feats[i]
                for s_i, sig in enumerate(SIGNALS):
                    ent = handles.get((sig, strand))
                    if ent is None:
                        bw = open_bw(BIGWIGS[(sig, strand)])
                        if bw is None:
                            row[:] = np.nan
                            continue
                        ent = handles[(sig, strand)] = (bw, bw.chroms())
                    bw, sizes = ent
                    clen = sizes.get(chrom)
                    if clen is None:
                        row[s_i * n_per_sig:(s_i + 1) * n_per_sig] = np.nan
                        continue
                    arr = oriented(bw, chrom, center, strand, clen)
                    base = s_i * n_per_sig
                    for near, far in WINDOWS:
                        reduce_window(arr, near, far, row, base)
                        base += 9
                if n and n % 5000 == 0:
                    log("    thread progress %d/%d" % (n, len(idxs)))
        finally:
            for bw, _ in handles.values():
                bw.close()
        return len(idxs)

    return run


def scan(peaks):
    feats = np.full((len(peaks), len(COLUMNS)), np.nan, dtype=np.float32)
    order = sorted(range(len(peaks)), key=lambda i: (peaks[i][1], peaks[i][2]))
    chunks = [order[k::N_THREADS] for k in range(N_THREADS)]
    chunks = [c for c in chunks if c]
    done = 0
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        for got in ex.map(make_worker(peaks, feats), chunks):
            done += got
            log("  scanned %d/%d peaks" % (done, len(peaks)))
    return feats


# holdout


def gene_split(genes):
    cache = {}
    out = np.empty(len(genes), dtype=np.int8)
    for i, g in enumerate(genes):
        h = cache.get(g)
        if h is None:
            d = hashlib.md5(("%d:%s" % (HOLDOUT_SEED, g)).encode()).hexdigest()
            h = cache[g] = int(d[:8], 16) & 1
        out[i] = h
    return out


def label(red, thr):
    if thr is None or thr == "":
        return red
    return "%s@%s" % (red, thr if isinstance(thr, str) else "%.1f" % thr)


# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_ucube_meta.tsv")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--remote", action="store_true",
                    help="force remote streaming even if local copies exist")
    ap.add_argument("--threads", type=int, default=N_THREADS)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: evenly sample N peaks across the table")
    ap.add_argument("--out-prefix", default="spliceai")
    args = ap.parse_args()
    globals()["N_THREADS"] = max(1, args.threads)

    n_local = 0
    for key, name in FILES.items():
        local = os.path.join(args.bw_dir, name)
        if not args.remote and os.path.isfile(local):
            BIGWIGS[key] = local
            n_local += 1
        else:
            BIGWIGS[key] = BASE + name
    log("bigWigs: %d/%d local (%s)" % (n_local, len(FILES), args.bw_dir))
    if n_local < len(FILES):
        log("  NOTE: streaming %d remotely; UCSC may drop connections on a full run"
            % (len(FILES) - n_local))

    peaks = read_peaks(args.table)
    if args.limit and args.limit < len(peaks):
        step = len(peaks) / float(args.limit)
        peaks = [peaks[int(i * step)] for i in range(args.limit)]
    log("%d peaks, %d encoding columns (%d per signal)"
        % (len(peaks), len(COLUMNS), len(COLUMNS) // len(SIGNALS)))

    y = np.array([p[4] for p in peaks], dtype=np.int8)
    split = gene_split([p[0] for p in peaks])
    n_genes = len(set(p[0] for p in peaks))

    log("scanning bigWigs once ...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = scan(peaks)
    log("scan done in %.1f min" % ((time.time() - t0) / 60.0))

    tr = split == 0
    te = split == 1
    n_per_sig = len(COLUMNS) // len(SIGNALS)

    log("computing %d x 2 AUROCs ..." % len(COLUMNS))
    recs = []
    for ci, (sig, near, far, red, thr) in enumerate(COLUMNS):
        col = feats[:, ci].astype(np.float64)
        a_tr, n_tr = auroc(col[tr], y[tr])
        a_te, n_te = auroc(col[te], y[te])
        recs.append({
            "signal": sig, "win_near": near, "win_far": far,
            "reduction": red, "thr": "" if thr is None else "%.1f" % thr,
            "auroc_train": a_tr, "auroc_test": a_te,
            "n_used": n_tr + n_te, "ci": ci,
        })

    grid_path = "%s_grid.tsv" % args.out_prefix
    with open(grid_path, "w") as fh:
        fh.write("signal\twin_near\twin_far\treduction\tthr\t"
                 "auroc_train\tauroc_test\tn_used\n")
        for r in recs:
            fh.write("%s\t%d\t%d\t%s\t%s\t%.4f\t%.4f\t%d\n"
                     % (r["signal"], r["win_near"], r["win_far"], r["reduction"],
                        r["thr"], r["auroc_train"], r["auroc_test"], r["n_used"]))

    n_drop = {}
    for s_i, sig in enumerate(SIGNALS):
        blk = feats[:, s_i * n_per_sig:(s_i + 1) * n_per_sig]
        n_drop[sig] = int(np.isnan(blk).all(axis=1).sum())

    ci_dm = COL_INDEX[("donor", 0, 2000, "max", None)]
    ci_am = COL_INDEX[("acceptor", 0, 2000, "max", None)]
    don_max_all, _ = auroc(feats[:, ci_dm].astype(np.float64), y)
    acc_max_all, _ = auroc(feats[:, ci_am].astype(np.float64), y)

    print("=" * 72)
    print("SpliceAI encoding grid: canonical vs competitor 3' ends")
    print("=" * 72)
    print()
    print("SANITY")
    print("  peaks                      : %d" % len(peaks))
    print("  genes                      : %d" % n_genes)
    print("  canonical / competitor     : %d / %d" % (int((y == 1).sum()),
                                                      int((y == 0).sum())))
    for sig in SIGNALS:
        print("  dropped all-NaN (%-8s) : %d" % (sig, n_drop[sig]))
    print("  donor  max [0,2000] AUROC  : %.4f  (expect ~%.3f)"
          % (don_max_all, BASE_SPLICEAI_DON_MAX_2KB))
    print("  acceptor max [0,2000] AUROC: %.4f  (expect ~%.3f)"
          % (acc_max_all, BASE_SPLICEAI_ACC_MAX_2KB))
    off = abs(don_max_all - BASE_SPLICEAI_DON_MAX_2KB)
    if off > 0.02:
        print()
        print("  STOP: donor-max[0,2000] is %.4f, off the expected %.3f by %.3f."
              % (don_max_all, BASE_SPLICEAI_DON_MAX_2KB, off))
        print("  Array orientation or strand map is wrong. Not reporting the grid.")
        sys.exit(1)
    print("  -> orientation/strand map reproduce the prior run.")
    print()

    print("GENE HOLDOUT (50/50 by md5 of gene, seed %d)" % HOLDOUT_SEED)
    for nm, m in (("train", tr), ("test", te)):
        print("  %-5s: %6d peaks  %5d canonical  %6d competitor  %5d genes"
              % (nm, int(m.sum()), int((y[m] == 1).sum()), int((y[m] == 0).sum()),
                 len(set(p[0] for p, k in zip(peaks, m) if k))))
    print()

    print("WINNER PER SIGNAL (selected on TRAIN; auroc_test is the honest number)")
    print("  %-9s %-12s %-16s %8s %8s %7s"
          % ("signal", "window", "reduction", "train", "TEST", "gap"))
    best_by_sig = {}
    for sig in SIGNALS:
        cand = [r for r in recs if r["signal"] == sig
                and not np.isnan(r["auroc_train"])]
        b = max(cand, key=lambda r: r["auroc_train"])
        best_by_sig[sig] = b
        print("  %-9s [%4d,%4d] %-16s %8.4f %8.4f %+7.4f"
              % (sig, b["win_near"], b["win_far"], label(b["reduction"], b["thr"]),
                 b["auroc_train"], b["auroc_test"],
                 b["auroc_test"] - b["auroc_train"]))
    print()

    print("NAIVE BEST auroc_test in the whole grid  [OPTIMISTIC - selected on the")
    print("thing it is scored on; the gap vs the train-selected row is the mirage]")
    for sig in SIGNALS:
        cand = [r for r in recs if r["signal"] == sig
                and not np.isnan(r["auroc_test"])]
        b = max(cand, key=lambda r: r["auroc_test"])
        hon = best_by_sig[sig]["auroc_test"]
        print("  %-9s [%4d,%4d] %-16s test=%.4f  OPTIMISTIC (honest %.4f, +%.4f)"
              % (sig, b["win_near"], b["win_far"], label(b["reduction"], b["thr"]),
                 b["auroc_test"], hon, b["auroc_test"] - hon))
    print()

    for sig in SIGNALS:
        a = best_by_sig[sig]["auroc_test"]
        if a < 0.5:
            print("  NOTE: %s auroc_test = %.4f is BELOW 0.5 -- raw direction is"
                  % (sig, a))
            print("        competitor-high, not canonical-high. Reported raw, not flipped.")
    print()

    print("BASELINES")
    print("  MaxEnt acceptor marginal        : %.3f" % BASE_MAXENT_ACC)
    print("  SpliceAI acceptor max 2kb (prior): %.3f" % BASE_SPLICEAI_ACC_MAX_2KB)
    print("  SpliceAI donor    max 2kb (prior): %.3f" % BASE_SPLICEAI_DON_MAX_2KB)
    print()

    dpath = "%s_auroc_vs_window.tsv" % args.out_prefix
    print("PANEL D: auroc_test vs win_far at each signal's winning reduction")
    with open(dpath, "w") as fh:
        fh.write("signal\treduction\tthr\twin_near\twin_far\tauroc_train\tauroc_test\n")
        for sig in SIGNALS:
            b = best_by_sig[sig]
            print("  %s / %s" % (sig, label(b["reduction"], b["thr"])))
            for near, far in EXPANDING:
                key = (sig, near, far, b["reduction"],
                       None if b["thr"] == "" else float(b["thr"]))
                r = recs[COL_INDEX[key]]
                print("    [%4d,%4d]  train %.4f   test %.4f"
                      % (near, far, r["auroc_train"], r["auroc_test"]))
                fh.write("%s\t%s\t%s\t%d\t%d\t%.4f\t%.4f\n"
                         % (sig, b["reduction"], b["thr"], near, far,
                            r["auroc_train"], r["auroc_test"]))
    print()

    epath = "%s_acc_x_donor.tsv" % args.out_prefix
    ai = best_by_sig["acceptor"]["ci"]
    di = best_by_sig["donor"]["ci"]
    av = feats[:, ai].astype(np.float64)
    dv = feats[:, di].astype(np.float64)
    ok = ~np.isnan(av) & ~np.isnan(dv)

    def qbin(v, nb=10):
        edges = np.unique(np.nanquantile(v, np.linspace(0, 1, nb + 1)))
        return np.clip(np.searchsorted(edges, v, side="right") - 1,
                       0, max(0, len(edges) - 2)), edges

    ab, aedges = qbin(av[ok])
    db, dedges = qbin(dv[ok])
    yo = y[ok]
    n_written = 0
    with open(epath, "w") as fh:
        fh.write("acc_bin\tacc_lo\tacc_hi\tdon_bin\tdon_lo\tdon_hi\tn\tn_canonical\tp_canonical\n")
        for i in range(max(1, len(aedges) - 1)):
            for j in range(max(1, len(dedges) - 1)):
                m = (ab == i) & (db == j)
                n = int(m.sum())
                if n < MIN_CELL:
                    continue
                nc = int((yo[m] == 1).sum())
                fh.write("%d\t%.4g\t%.4g\t%d\t%.4g\t%.4g\t%d\t%d\t%.4f\n"
                         % (i, aedges[i], aedges[i + 1], j, dedges[j], dedges[j + 1],
                            n, nc, nc / float(n)))
                n_written += 1
    print("PANEL E: acceptor x donor joint P(canonical), >=%d peaks/cell" % MIN_CELL)
    print("  acceptor encoding: [%d,%d] %s"
          % (best_by_sig["acceptor"]["win_near"], best_by_sig["acceptor"]["win_far"],
             label(best_by_sig["acceptor"]["reduction"], best_by_sig["acceptor"]["thr"])))
    print("  donor    encoding: [%d,%d] %s"
          % (best_by_sig["donor"]["win_near"], best_by_sig["donor"]["win_far"],
             label(best_by_sig["donor"]["reduction"], best_by_sig["donor"]["thr"])))
    print("  cells written: %d" % n_written)
    print()

    print("WROTE")
    for p in (grid_path, dpath, epath):
        print("  %s" % p)


if __name__ == "__main__":
    main()
