#!/usr/bin/env python
# all derived data and statistics for the merged figure 3

import os
import sys
import csv
import math
import time
import argparse
import warnings
from collections import defaultdict

import numpy as np

MIN_CELL = 200
Z95 = 1.959963984540054
UP = 2000
CACHE = "sa_peak_scalars_final.tsv"
BASE_URL = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/"
BW_NAMES = {
    ("acceptor", "+"): "spliceAiAcceptorPlus.bw",
    ("acceptor", "-"): "spliceAiAcceptorMinus.bw",
    ("donor", "+"): "spliceAiDonorPlus.bw",
    ("donor", "-"): "spliceAiDonorMinus.bw",
}

REQ_MAXENT = ("ratio.tsv", "bcshow_ratio.tsv", "bcshow_auroc.tsv",
              "acc_x_interv.tsv", "class_counts.tsv",
              "first5_from_anchor.tsv", "internal_exon_like.tsv", "metagene.tsv")
REQ_SA = ("sa_acc_x_don_final.tsv", "sa_auroc_window_final.tsv",
          "sa_tertile_final.tsv")
OPTIONAL = ("acc_paired_summary.tsv",)


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


# statistics


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_label(p):
    if not (p > 1e-300):
        return "p < 1e-300"
    if p < 0.001:
        return "p = %.0e" % p
    return "p = %.3f" % p


def logistic_irls(X, y, max_iter=60, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-10)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(XtW @ X, XtW @ z, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(mu * (1 - mu), 1e-10)
    cov = np.linalg.inv((X.T * w) @ X)
    return beta, np.sqrt(np.diag(cov))


def zscore(v):
    ok = ~np.isnan(v)
    mu, sd = v[ok].mean(), v[ok].std()
    return (v - mu) / (sd if sd > 0 else 1.0)


def fit_or(y, preds, want, label, scorer):
    names = [n for n, _ in preds]
    M = np.column_stack([v for _, v in preds])
    ok = ~np.isnan(M).any(axis=1)
    X = np.column_stack([np.ones(int(ok.sum())), M[ok]])
    beta, se = logistic_irls(X, y[ok].astype(float))
    j = names.index(want) + 1
    b, s = beta[j], se[j]
    p = two_sided_p(b / s)
    return {"block": "effect", "effect_label": label, "scorer": scorer,
            "estimate": math.exp(b), "ci_lo": math.exp(b - Z95 * s),
            "ci_hi": math.exp(b + Z95 * s), "p": p, "p_label": p_label(p),
            "null_value": 1.0, "n": int(ok.sum()), "source": "computed"}


def within_gene_rates(score, y, genes):
    by = defaultdict(lambda: ([], []))
    for i in range(len(score)):
        if np.isnan(score[i]):
            continue
        by[genes[i]][0 if y[i] == 1 else 1].append(score[i])
    out = []
    for _g, (can, comp) in by.items():
        if not can or not comp:
            continue
        c = np.asarray(can)[:, None]
        k = np.asarray(comp)[None, :]
        out.append(float((c > k).sum() + 0.5 * (c == k).sum()) / (c.size * k.size))
    return np.asarray(out)


def wilcoxon_signed_rank_p(d):
    d = d[d != 0]
    n = d.size
    if n < 10:
        return float("nan")
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    newgrp = np.r_[True, sa[1:] != sa[:-1]]
    grp = np.cumsum(newgrp)
    starts = np.r_[np.nonzero(newgrp)[0], n]
    avg = (starts[:-1] + starts[1:] + 1) / 2.0
    ranks = np.empty(n)
    ranks[order] = avg[grp - 1]
    W = ranks[d > 0].sum()
    mu = n * (n + 1) / 4.0
    tie = np.bincount(grp)[1:].astype(float)
    var = (n * (n + 1) * (2 * n + 1) - (tie ** 3 - tie).sum() / 2.0) / 24.0
    if var <= 0:
        return float("nan")
    return two_sided_p((W - mu) / math.sqrt(var))


# spliceai rescan fallback


def rescan(rows, bw_dir, threads):
    try:
        import pyBigWig
    except ImportError:
        sys.exit("cache absent and pyBigWig not installed; cannot rescan")
    from concurrent.futures import ThreadPoolExecutor
    bw_paths = {}
    missing = []
    for key, nm in BW_NAMES.items():
        p = os.path.join(bw_dir, nm)
        if os.path.isfile(p):
            bw_paths[key] = p
        else:
            missing.append(p)
    if missing:
        print("MISSING bigWigs (needed because %s is absent):" % CACHE, file=sys.stderr)
        for m in missing:
            print("  %s" % m, file=sys.stderr)
        sys.exit(1)
    need = {"chrom", "center", "strand"}
    if not need <= set(rows[0].keys()):
        sys.exit("cache absent and peak table lacks chrom/center/strand")
    peaks = [(r["chrom"], int(float(r["center"])), r["strand"]) for r in rows]
    n = len(peaks)
    out = np.full((n, 2), np.nan)

    def oriented(bw, chrom, center, strand, clen):
        arr = np.full(UP, np.nan, dtype=np.float32)
        lo, hi = (center - UP, center) if strand == "+" else (center, center + UP)
        lo_c, hi_c = max(0, lo), min(hi, clen)
        if hi_c <= lo_c:
            return arr
        try:
            raw = np.asarray(bw.values(chrom, lo_c, hi_c), dtype=np.float32)
        except (RuntimeError, OverflowError):
            return arr
        if strand == "+":
            arr[center - hi_c:center - lo_c] = raw[::-1]
        else:
            arr[lo_c - center:hi_c - center] = raw
        return arr

    def run(idxs):
        handles = {}
        try:
            for i in idxs:
                chrom, center, strand = peaks[i]
                for s_i, sig in enumerate(("acceptor", "donor")):
                    ent = handles.get((sig, strand))
                    if ent is None:
                        bw = pyBigWig.open(bw_paths[(sig, strand)])
                        ent = handles[(sig, strand)] = (bw, bw.chroms())
                    bw, sizes = ent
                    clen = sizes.get(chrom)
                    if clen is None:
                        continue
                    a = oriented(bw, chrom, center, strand, clen)
                    cov = ~np.isnan(a)
                    if not cov.any():
                        continue
                    v = a[cov]
                    out[i, s_i] = v.max() if sig == "acceptor" else v.sum()
        finally:
            for bw, _ in handles.values():
                bw.close()
        return len(idxs)

    order = sorted(range(n), key=lambda i: (peaks[i][0], peaks[i][1]))
    jobs = [order[k::threads] for k in range(threads)]
    jobs = [j for j in jobs if j]
    log("cache absent; scanning bigWigs (%d peaks) ..." % n)
    t0 = time.time()
    done = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            for got in ex.map(run, jobs):
                done += got
                log("  scanned %d/%d" % (done, n))
    log("scan done in %.1f min" % ((time.time() - t0) / 60.0))
    return out[:, 0], out[:, 1]


# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_ucube_meta.tsv")
    ap.add_argument("--indir", default=".")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--prefix", default="fig3")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    outdir = args.outdir or args.indir

    def pf(n):
        return os.path.join(args.indir, "%s_%s" % (args.prefix, n))

    # dependency contract
    peak_tbl = args.table if os.path.isfile(args.table) \
        else os.path.join(args.indir, args.table)
    req = [pf(n) for n in REQ_MAXENT] + \
          [os.path.join(args.indir, n) for n in REQ_SA] + [peak_tbl]
    missing = [p for p in req if not os.path.isfile(p)]
    if missing:
        print("MISSING REQUIRED INPUTS (%d):" % len(missing), file=sys.stderr)
        for m in missing:
            print("  %s" % m, file=sys.stderr)
        print("Run figure3_data.py and the SpliceAI steps first.", file=sys.stderr)
        sys.exit(1)
    log("dependency contract satisfied (%d required files)" % len(req))
    for n in OPTIONAL:
        log("  optional %s: %s" % (pf(n), "present" if os.path.isfile(pf(n)) else "absent"))

    P = read_tsv(peak_tbl)
    cols = set(P[0].keys())
    don_col = "interv_donor" if "interv_donor" in cols else None
    if don_col is None:
        for c in ("interv_sum5", "interv_density5"):
            if c in cols:
                don_col = c
                break
    if don_col is None:
        sys.exit("no interv_donor or fallback donor column in %s" % peak_tbl)
    if don_col != "interv_donor":
        log("NOTE: interv_donor absent; using %s" % don_col)
    genes = [r["gene"] for r in P]
    y = np.array([int(float(r["is_canonical"])) for r in P], dtype=np.int8)
    acc = np.array([fnum(r["acc"]) for r in P])
    ivd = np.array([fnum(r[don_col]) for r in P])

    cache_p = os.path.join(args.indir, CACHE)
    if os.path.isfile(cache_p):
        S = read_tsv(cache_p)
        if len(S) != len(P):
            sys.exit("%s has %d rows, %s has %d" % (CACHE, len(S), peak_tbl, len(P)))
        if [r["gene"] for r in S] != genes:
            sys.exit("%s gene order does not match %s" % (CACHE, peak_tbl))
        sacc = np.array([fnum(r["sa_acc_max"]) for r in S])
        sdon = np.array([fnum(r["sa_don_sum"]) for r in S])
        log("reused cached SpliceAI scalars: %s (no bigWig access)" % CACHE)
    else:
        sacc, sdon = rescan(P, args.bw_dir, max(1, args.threads))
        with open(cache_p, "w") as fh:
            fh.write("gene\tchrom\tcenter\tstrand\tis_canonical\tsa_acc_max\tsa_don_sum\n")
            for i, r in enumerate(P):
                fh.write("%s\t%s\t%s\t%s\t%d\t%.6g\t%.6g\n"
                         % (r["gene"], r.get("chrom", ""), r.get("center", ""),
                            r.get("strand", ""), y[i], sacc[i], sdon[i]))
        log("cached -> %s" % cache_p)

    n_can, n_comp = int((y == 1).sum()), int((y == 0).sum())
    b = n_can / float(n_can + n_comp)
    log("peaks %d (%d real / %d competitor), base rate %.5f" % (len(P), n_can, n_comp, b))

    za, zi = zscore(acc), zscore(ivd)
    zsa, zsd = zscore(sacc), zscore(sdon)

    rows = []
    rows.append(fit_or(y, [("acc", za)], "acc", "acceptor 3'ss", "MaxEnt"))
    rows.append(fit_or(y, [("acc", zsa)], "acc", "acceptor 3'ss", "SpliceAI"))
    rows.append(fit_or(y, [("acc", za), ("don", zi)], "don",
                       "donor 5'ss | acceptor", "MaxEnt"))
    rows.append(fit_or(y, [("acc", zsa), ("don", zsd)], "don",
                       "donor 5'ss | acceptor", "SpliceAI"))
    rows.append(fit_or(y, [("acc", za), ("don", zi), ("int", za * zi)], "int",
                       "acceptor x donor", "MaxEnt"))
    rows.append(fit_or(y, [("acc", zsa), ("don", zsd), ("int", zsa * zsd)], "int",
                       "acceptor x donor", "SpliceAI"))

    def ranking_row(score, scorer, summary_path=None):
        r = within_gene_rates(score, y, genes)
        est = float(r.mean())
        half = Z95 * float(r.std(ddof=1)) / math.sqrt(r.size)
        p = wilcoxon_signed_rank_p(r - 0.5)
        src = "computed"
        if summary_path and os.path.isfile(summary_path):
            s = read_tsv(summary_path)
            if s and "within_gene_auroc" in s[0]:
                est = fnum(s[0]["within_gene_auroc"])
                if "p_wilcox" in s[0]:
                    p = fnum(s[0]["p_wilcox"])
                src = os.path.basename(summary_path)
                log("  within-gene %s read from %s (CI computed here)" % (scorer, src))
        return {"block": "ranking", "effect_label": "within-gene win-rate, acceptor",
                "scorer": scorer, "estimate": est,
                "ci_lo": max(0.0, est - half), "ci_hi": min(1.0, est + half),
                "p": p, "p_label": p_label(p), "null_value": 0.5,
                "n": int(r.size), "source": src}
    rows.append(ranking_row(acc, "MaxEnt", pf("acc_paired_summary.tsv")))
    rows.append(ranking_row(sacc, "SpliceAI"))

    fp = os.path.join(outdir, "stats_forest.tsv")
    with open(fp, "w") as fh:
        fh.write("block\teffect_label\tscorer\testimate\tci_lo\tci_hi\tp\tp_label\t"
                 "null_value\tn\tsource\n")
        for r in rows:
            fh.write("%s\t%s\t%s\t%.6g\t%.6g\t%.6g\t%.6g\t%s\t%.1f\t%d\t%s\n"
                     % (r["block"], r["effect_label"], r["scorer"], r["estimate"],
                        r["ci_lo"], r["ci_hi"], r["p"], r["p_label"],
                        r["null_value"], r["n"], r["source"]))

    log("")
    log("STATS_FOREST")
    log("  %-8s %-30s %-9s %10s %22s %14s %8s"
        % ("block", "effect", "scorer", "estimate", "95% CI", "p", "n"))
    for r in rows:
        log("  %-8s %-30s %-9s %10.4f  [%8.4f,%8.4f] %14s %8d"
            % (r["block"], r["effect_label"], r["scorer"], r["estimate"],
               r["ci_lo"], r["ci_hi"], r["p_label"], r["n"]))
    log("")

    A = read_tsv(pf("acc_x_interv.tsv"))
    bh = float(np.median([(fnum(r["p_canonical"]) + 1e-3)
                          / 2 ** fnum(r["log2_enrich"]) - 1e-3 for r in A]))
    hm = os.path.join(outdir, "heat_maxent.tsv")
    with open(hm, "w") as fh:
        fh.write("acceptor\tinterv\tn\tp_canonical\tlog2_or\tbase_rate\n")
        for r in A:
            p = min(max(fnum(r["p_canonical"]), 1e-4), 1 - 1e-4)
            fh.write("%s\t%s\t%s\t%s\t%.6f\t%.6f\n"
                     % (r["acceptor"], r["interv"], r["n"], r["p_canonical"],
                        math.log2((p / (1 - p)) / (bh / (1 - bh))), bh))
    G = read_tsv(os.path.join(args.indir, "sa_acc_x_don_final.tsv"))
    hs = os.path.join(outdir, "heat_spliceai.tsv")
    with open(hs, "w") as fh:
        fh.write("acc_bin\tdon_bin\tn\tp_canonical\tlog2_or\tbase_rate\n")
        for r in G:
            if r.get("log2_or") not in (None, ""):
                l2 = fnum(r["log2_or"])
            else:
                p = min(max(fnum(r["p_canonical"]), 1e-4), 1 - 1e-4)
                l2 = math.log2((p / (1 - p)) / (b / (1 - b)))
            fh.write("%s\t%s\t%s\t%s\t%.6f\t%.6f\n"
                     % (r["acc_bin"], r["don_bin"], r["n"], r["p_canonical"], l2, b))

    log("base rate: heatmap-derived %.5f, class-counts %.5f" % (bh, b))
    log("WROTE")
    for p in (fp, hm, hs):
        log("  %s" % p)


if __name__ == "__main__":
    main()
