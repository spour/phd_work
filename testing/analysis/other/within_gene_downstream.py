#!/usr/bin/env python
# diagnostic only

import os
import sys
import csv
import math
import argparse
from collections import defaultdict

import numpy as np

THRESHOLDS = (0.8, 0.5, 0.2)
CACHE = "sa_donor_presence_cache.tsv"


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


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_label(p):
    if not (p > 1e-300):
        return "< 1e-300"
    return "%.1e" % p


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


def win_rates(score, y, genes, lower_is_canonical=True):
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
        wins = (c < k).sum() if lower_is_canonical else (c > k).sum()
        out.append(float(wins + 0.5 * (c == k).sum()) / (c.size * k.size))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=".")
    ap.add_argument("--peaks", default="fig3_ucube_meta.tsv")
    args = ap.parse_args()

    cache_p = os.path.join(args.indir, CACHE)
    peaks_p = args.peaks if os.path.isfile(args.peaks) \
        else os.path.join(args.indir, args.peaks)
    for p in (cache_p, peaks_p):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)

    C = read_tsv(cache_p)
    P = read_tsv(peaks_p)
    if len(C) != len(P):
        sys.exit("%s has %d rows, %s has %d" % (CACHE, len(C), peaks_p, len(P)))
    if [r["gene"] for r in C] != [r["gene"] for r in P]:
        sys.exit("%s gene order does not match %s" % (CACHE, peaks_p))

    genes = [r["gene"] for r in P]
    y = np.array([int(float(r["is_canonical"])) for r in P], dtype=np.int8)
    dn = np.array([fnum(r["don_max_dn"]) for r in C])
    center = np.array([fnum(r["center"]) for r in P])
    strand = np.array([r["strand"] for r in P])
    n_can, n_comp = int((y == 1).sum()), int((y == 0).sum())
    log("peaks %d (%d real ends / %d competitors), genes %d"
        % (len(P), n_can, n_comp, len(set(genes))))
    log("downstream donor max: %d NaN" % int(np.isnan(dn).sum()))
    log("")

    rows = []
    log("=" * 100)
    log("DOWNSTREAM-PAST-pA DONOR: pooled contrast vs within-gene win-rate")
    log("=" * 100)
    log("  %-6s %9s %9s %8s | %11s %11s %11s %7s"
        % ("thr", "canon", "comp", "pooled", "win-rate", "Wilcoxon p", "n genes", "shrink"))
    for t in THRESHOLDS:
        pres = (np.nan_to_num(dn, nan=-1) >= t).astype(float)
        fc = pres[y == 1].mean()
        fk = pres[y == 0].mean()
        ratio = fk / fc if fc > 0 else float("nan")
        r = win_rates(pres, y, genes, lower_is_canonical=True)
        wr = float(r.mean())
        p = wilcoxon_signed_rank_p(r - 0.5)
        shrink = (wr - 0.5) / 0.5
        rows.append((t, fc, fk, ratio, wr, p, r.size))
        log("  %-6.1f %9.4f %9.4f %7.2fx | %11.4f %11s %11d %6.0f%%"
            % (t, fc, fk, ratio, wr, p_label(p), r.size, 100 * shrink))
    log("")
    log("  win-rate = P(canonical has LOWER downstream donor than its competitor,")
    log("  same gene), gene-averaged, ties 0.5. Continuous check at each threshold's")
    log("  underlying value follows.")
    rc = win_rates(dn, y, genes, lower_is_canonical=True)
    log("  continuous don_max_dn win-rate = %.4f  (Wilcoxon %s, n genes %d)"
        % (rc.mean(), p_label(wilcoxon_signed_rank_p(rc - 0.5)), rc.size))
    log("")

    # distance-stratified check: is this class, or position in the gene?
    can_pos = {}
    for i in range(len(P)):
        if y[i] == 1:
            can_pos[genes[i]] = (center[i], strand[i])
    d_signed = np.full(len(P), np.nan)
    for i in range(len(P)):
        if y[i] == 1:
            continue
        cp = can_pos.get(genes[i])
        if cp is None:
            continue
        d_signed[i] = (cp[0] - center[i]) if cp[1] == "+" else (center[i] - cp[0])
    log("=" * 100)
    log("DISTANCE CHECK: competitor downstream-donor presence (thr 0.8) vs distance")
    log("upstream of that gene's canonical end. If the effect is positional rather")
    log("than class-driven, competitors near the canonical end look like it.")
    log("=" * 100)
    pres8 = (np.nan_to_num(dn, nan=-1) >= 0.8).astype(float)
    log("  canonical peaks                       : %.4f  (n = %d)"
        % (pres8[y == 1].mean(), n_can))
    edges = [0, 500, 1000, 2000, 5000, 10000, 25000, 50000, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y == 0) & np.isfinite(d_signed) & (d_signed >= lo) & (d_signed < hi)
        if m.sum() < 50:
            continue
        log("  competitors %6.0f-%-8s upstream of end : %.4f  (n = %d)"
            % (lo, ("inf" if hi == np.inf else "%.0f" % hi), pres8[m].mean(), int(m.sum())))
    log("")

    out = os.path.join(args.indir, "within_gene_downstream.tsv")
    with open(out, "w") as fh:
        fh.write("threshold\tfrac_canonical\tfrac_competitor\tpooled_ratio\t"
                 "within_gene_winrate\twilcoxon_p\tn_genes\n")
        for t, fc, fk, ratio, wr, p, ng in rows:
            fh.write("%.1f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6g\t%d\n"
                     % (t, fc, fk, ratio, wr, p, ng))

    # verdict
    wr8 = rows[0][4]
    log("=" * 100)
    if wr8 >= 0.65:
        log("VERDICT: within-gene win-rate %.3f at thr 0.8 stays well above 0.5." % wr8)
        log("  The contrast survives gene-level control, so it is not explained by")
        log("  gene-to-gene differences in annotation density.")
    elif wr8 <= 0.55:
        log("VERDICT: within-gene win-rate %.3f at thr 0.8 collapses toward 0.5." % wr8)
        log("  The pooled contrast was gene-level annotation density, not a per-site")
        log("  signal. Do not build the panel.")
    else:
        log("VERDICT: within-gene win-rate %.3f at thr 0.8 is intermediate." % wr8)
    log("")
    log("  CAVEAT, read before concluding 'real sequence signal': surviving the")
    log("  within-gene control does NOT rule out annotation leakage. A canonical")
    log("  peak sits at the annotated gene end by construction, so the region")
    log("  downstream of it is post-gene while the region downstream of a competitor")
    log("  is still gene body. Class and position-in-gene stay confounded inside a")
    log("  gene. The distance table above is the relevant read: if competitors close")
    log("  to the canonical end already look canonical, the effect tracks position,")
    log("  not class.")
    log("wrote %s" % out)


if __name__ == "__main__":
    main()
