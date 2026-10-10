#!/usr/bin/env python
# spliceai-scored intervening donor, matching panel f's maxent definition

import os
import sys
import csv
import math
import time
import hashlib
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CACHE = "sa_interv_donor_cache.tsv"
BW = {"+": "spliceAiDonorPlus.bw", "-": "spliceAiDonorMinus.bw"}
MIN_CELL = 200
N_DECILES = 10
HOLDOUT_SEED = 0
Z95 = 1.959963984540054


def log(m):
    print(m, file=sys.stderr, flush=True)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_tsv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def auroc(scores, labels):
    ok = ~np.isnan(scores)
    s, y = scores[ok], labels[ok]
    n = s.size
    npos = int((y == 1).sum())
    if npos == 0 or npos == n:
        return float("nan"), n
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    ng = np.r_[True, ss[1:] != ss[:-1]]
    g = np.cumsum(ng)
    st = np.r_[np.nonzero(ng)[0], n]
    av = (st[:-1] + st[1:] + 1) / 2.0
    r = np.empty(n)
    r[o] = av[g - 1]
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * (n - npos)), n


def scan(rows, bw_dir, threads):
    try:
        import pyBigWig
    except ImportError:
        sys.exit("pyBigWig not installed")
    paths = {}
    for s, nm in BW.items():
        p = os.path.join(bw_dir, nm)
        if not os.path.isfile(p):
            sys.exit("missing %s" % p)
        paths[s] = p
    n = len(rows)
    out = np.full(n, np.nan)
    span = np.full(n, np.nan)

    def run(idxs):
        h = {}
        try:
            for i in idxs:
                r = rows[i]
                anc = fnum(r["anchor_3ss_pos"])
                if not math.isfinite(anc):
                    continue
                c = int(float(r["center"]))
                strand = r["strand"]
                lo, hi = (int(min(anc, c)), int(max(anc, c)))
                ent = h.get(strand)
                if ent is None:
                    bw = pyBigWig.open(paths[strand])
                    ent = h[strand] = (bw, bw.chroms())
                bw, sizes = ent
                clen = sizes.get(r["chrom"])
                if clen is None:
                    continue
                lo, hi = max(0, lo), min(hi, clen)
                if hi <= lo:
                    span[i] = 0.0
                    continue
                span[i] = hi - lo
                try:
                    v = np.asarray(bw.values(r["chrom"], lo, hi), dtype=np.float32)
                except (RuntimeError, OverflowError):
                    continue
                v = v[~np.isnan(v)]
                if v.size:
                    out[i] = float(v.max())
        finally:
            for bw, _ in h.values():
                bw.close()
        return len(idxs)

    order = sorted(range(n), key=lambda i: (rows[i]["chrom"], int(float(rows[i]["center"]))))
    jobs = [order[k::threads] for k in range(threads)]
    jobs = [j for j in jobs if j]
    log("scanning donor bigWigs over anchor->pA windows (%d peaks) ..." % n)
    t0 = time.time()
    done = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            for got in ex.map(run, jobs):
                done += got
                log("  scanned %d/%d" % (done, n))
    log("scan done in %.1f min" % ((time.time() - t0) / 60.0))
    return out, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--scalars", default="sa_peak_scalars_final.tsv")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--rescan", action="store_true")
    args = ap.parse_args()

    for p in (args.pos, args.scalars):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)
    P = read_tsv(args.pos)
    S = read_tsv(args.scalars)
    if len(P) != len(S):
        sys.exit("row count mismatch: %s %d vs %s %d" % (args.pos, len(P), args.scalars, len(S)))

    genes = [r["gene"] for r in P]
    y = np.array([int(float(r["is_canonical"])) for r in P], dtype=np.int8)
    sa_acc = np.array([fnum(r["sa_acc_max"]) for r in S])
    n_can, n_comp = int((y == 1).sum()), int((y == 0).sum())
    b = n_can / float(n_can + n_comp)

    if os.path.isfile(CACHE) and not args.rescan:
        C = read_tsv(CACHE)
        if len(C) == len(P):
            iv = np.array([fnum(r["sa_interv_donor"]) for r in C])
            span = np.array([fnum(r["window_nt"]) for r in C])
            log("reused %s" % CACHE)
        else:
            iv = None
    else:
        iv = None
    if iv is None:
        iv, span = scan(P, args.bw_dir, max(1, args.threads))
        with open(CACHE, "w") as fh:
            fh.write("gene\tchrom\tcenter\tstrand\tis_canonical\tanchor_3ss_pos\t"
                     "window_nt\tsa_interv_donor\n")
            for i, r in enumerate(P):
                fh.write("%s\t%s\t%s\t%s\t%d\t%s\t%.6g\t%.6g\n"
                         % (r["gene"], r["chrom"], r["center"], r["strand"], y[i],
                            r["anchor_3ss_pos"], span[i], iv[i]))
        log("cached -> %s" % CACHE)

    log("")
    log("COVERAGE")
    log("  peaks %d (%d real / %d competitor), base rate %.5f" % (len(P), n_can, n_comp, b))
    log("  sa_interv_donor defined: %d (%.2f%%)"
        % (int(np.isfinite(iv).sum()), 100.0 * np.isfinite(iv).sum() / len(P)))
    ok = np.isfinite(span)
    log("  anchor->pA window nt: median %.0f, p10 %.0f, p90 %.0f"
        % (np.median(span[ok]), np.quantile(span[ok], .1), np.quantile(span[ok], .9)))
    log("  marginal AUROC of sa_interv_donor (all peaks): %.4f" % auroc(iv, y)[0])
    log("  marginal AUROC of sa_acc_max      (all peaks): %.4f" % auroc(sa_acc, y)[0])

    # binned table on the matched feature
    m = np.isfinite(iv) & np.isfinite(sa_acc)

    def decile(v):
        e = np.unique(np.quantile(v, np.linspace(0, 1, N_DECILES + 1)))
        return np.clip(np.searchsorted(e, v, side="right") - 1, 0, max(0, len(e) - 2)), e

    ab, aed = decile(sa_acc[m])
    db, ded = decile(iv[m])
    yo = y[m]
    rows = []
    for i in range(max(1, len(aed) - 1)):
        for j in range(max(1, len(ded) - 1)):
            sel = (ab == i) & (db == j)
            n = int(sel.sum())
            if n < MIN_CELL:
                continue
            nc = int((yo[sel] == 1).sum())
            p = min(max(nc / float(n), 1e-4), 1 - 1e-4)
            rows.append(dict(acc_bin=i + 1, don_bin=j + 1, n=n, n_canonical=nc,
                             p_canonical=nc / float(n),
                             log2_or=math.log2((p / (1 - p)) / (b / (1 - b))),
                             base_rate=b))
    out = "heat_spliceai_interv.tsv"
    with open(out, "w") as fh:
        fh.write("acc_bin\tdon_bin\tn\tn_canonical\tp_canonical\tlog2_or\tbase_rate\n")
        for r in rows:
            fh.write("%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\n"
                     % (r["acc_bin"], r["don_bin"], r["n"], r["n_canonical"],
                        r["p_canonical"], r["log2_or"], r["base_rate"]))
    log("  wrote %s (%d cells >= %d)" % (out, len(rows), MIN_CELL))

    # per-acceptor-row slopes, the f/g comparison
    log("")
    log("PER-ACCEPTOR-ROW SLOPE of log2 OR vs intervening-donor bin")
    log("  (F MaxEnt is DOWN at every acceptor level; G must match to be the same object)")
    log("  %-12s %8s %8s %9s %s" % ("acc bin", "first", "last", "slope", "dir"))
    n_down = n_up = 0
    for a in sorted({r["acc_bin"] for r in rows}, reverse=True):
        rr = sorted([r for r in rows if r["acc_bin"] == a], key=lambda r: r["don_bin"])
        if len(rr) < 3:
            continue
        x = np.array([r["don_bin"] for r in rr], float)
        v = np.array([r["log2_or"] for r in rr], float)
        sl = float(np.polyfit(x, v, 1)[0])
        n_down += sl < 0
        n_up += sl >= 0
        log("  %-12d %8.2f %8.2f %9.4f %s" % (a, v[0], v[-1], sl, "DOWN" if sl < 0 else "UP"))
    log("  rows declining: %d / %d" % (n_down, n_down + n_up))

    # tertile auroc on the matched feature, train-cut applied to test
    split = np.array([int(hashlib.md5(("%d:%s" % (HOLDOUT_SEED, g)).encode()).hexdigest()[:8], 16) & 1
                      for g in genes], dtype=np.int8)
    tr, te = split == 0, split == 1
    q = np.nanquantile(sa_acc[tr], [1 / 3.0, 2 / 3.0])
    tert = np.digitize(sa_acc, q)
    log("")
    log("TERTILE AUROC of sa_interv_donor within acceptor tertiles (TEST, raw)")
    log("  %-8s %9s %-12s %8s" % ("tertile", "auroc", "direction", "n"))
    strong = None
    for t_i, lab in enumerate(("weak", "medium", "strong")):
        sel = te & (tert == t_i) & np.isfinite(sa_acc)
        a, n = auroc(iv[sel], y[sel])
        d = "canonical" if a > 0.5 else "COMPETITOR"
        if lab == "strong":
            strong = a
        log("  %-8s %9.4f %-12s %8d" % (lab, a, d, n))
    log("")
    if strong is not None and strong < 0.5:
        log("  -> strong-acceptor tertile %.4f < 0.5: competitor direction, as with sum-over-2kb." % strong)
    else:
        log("  -> WARNING: strong-acceptor tertile %.4f is NOT competitor direction." % strong)


if __name__ == "__main__":
    main()
