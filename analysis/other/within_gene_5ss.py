#!/usr/bin/env python
# diagnostic only, no panel, no re-scan

import os
import sys
import csv
import math
import argparse
from collections import defaultdict

import numpy as np

GATE_RATE, GATE_P = 0.55, 1e-10


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
    if np.isnan(p):
        return "n/a"
    if not (p > 1e-300):
        return "< 1e-300"
    return "%.1e" % p


def wilcoxon_signed_rank_p(d):
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
    if var <= 0:
        return float("nan")
    return two_sided_p((W - mu) / math.sqrt(var))


def win_rate(score, y, genes, canonical_lower):
    by = defaultdict(lambda: ([], []))
    for i in range(len(score)):
        if np.isnan(score[i]):
            continue
        by[genes[i]][0 if y[i] == 1 else 1].append(score[i])
    rates, tot_w, tot_p = [], 0.0, 0
    for _g, (can, comp) in by.items():
        if not can or not comp:
            continue
        c = np.asarray(can)[:, None]
        k = np.asarray(comp)[None, :]
        wins = float((c < k).sum() if canonical_lower else (c > k).sum())
        wins += 0.5 * float((c == k).sum())
        p = c.size * k.size
        rates.append(wins / p)
        tot_w += wins
        tot_p += p
    r = np.asarray(rates)
    return r, (tot_w / tot_p if tot_p else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", default="fig3_ucube_meta.tsv")
    ap.add_argument("--indir", default=".")
    args = ap.parse_args()

    p_peaks = args.peaks if os.path.isfile(args.peaks) \
        else os.path.join(args.indir, args.peaks)
    if not os.path.isfile(p_peaks):
        sys.exit("missing required input: %s" % p_peaks)
    R = read_tsv(p_peaks)
    cols = set(R[0].keys())
    for c in ("gene", "is_canonical", "acc"):
        if c not in cols:
            sys.exit("%s lacks column %s" % (p_peaks, c))
    genes = [r["gene"] for r in R]
    y = np.array([int(float(r["is_canonical"])) for r in R], dtype=np.int8)
    acc = np.array([fnum(r["acc"]) for r in R])
    n_can = int((y == 1).sum())
    log("peaks %d (%d real ends / %d competitors), genes %d"
        % (len(R), n_can, len(R) - n_can, len(set(genes))))

    pub = os.path.join(args.indir, "fig3_acc_paired_summary.tsv")
    pub_val = pub_p = None
    if os.path.isfile(pub):
        s = read_tsv(pub)
        if s:
            pub_val = fnum(s[0].get("within_gene_auroc", "nan"))
            pub_p = fnum(s[0].get("p_wilcox", "nan"))
            log("fig3_acc_paired_summary.tsv: within_gene_auroc %.4f, p_wilcox %s, n_genes %s"
                % (pub_val, s[0].get("p_wilcox"), s[0].get("n_genes")))

    tests = [("acceptor 3'ss", "MaxEnt", acc, False,
              "canonical HIGHER (definition signal)")]
    don_col = "interv_donor" if "interv_donor" in cols else None
    if don_col:
        iv = np.array([fnum(r[don_col]) for r in R])
        tests.append(("intervening 5'ss", "MaxEnt", iv, True,
                      "canonical LOWER (competitor should carry it)"))
    else:
        log("NOTE: interv_donor absent from %s; MaxEnt 5'ss test skipped" % p_peaks)

    for fn, col, lab in (("sa_peak_scalars_final.tsv", "sa_don_sum", "donor sum 2kb"),
                         ("sa_interv_donor_cache.tsv", "sa_interv_donor", "intervening (matched)")):
        p = os.path.join(args.indir, fn)
        if not os.path.isfile(p):
            continue
        C = read_tsv(p)
        if len(C) != len(R) or [c["gene"] for c in C] != genes:
            log("NOTE: %s does not align with %s; skipped" % (fn, p_peaks))
            continue
        tests.append(("5'ss %s" % lab, "SpliceAI",
                      np.array([fnum(c[col]) for c in C]), True,
                      "canonical LOWER (competitor should carry it)"))

    print("=" * 96)
    print("WITHIN-GENE WIN-RATES   (gene-fixed; each canonical vs its own gene's competitors)")
    print("=" * 96)
    print("  %-26s %-9s %10s %10s %12s %8s  %s"
          % ("feature", "scorer", "gene-avg", "pair-wtd", "Wilcoxon p", "genes", "expected side"))
    res = {}
    for name, scorer, v, lower, expect in tests:
        r, pw = win_rate(v, y, genes, lower)
        ga = float(r.mean())
        p = wilcoxon_signed_rank_p(r - 0.5)
        res[(name, scorer)] = (ga, pw, p, r.size)
        print("  %-26s %-9s %10.4f %10.4f %12s %8d  %s"
              % (name, scorer, ga, pw, p_label(p), r.size, expect))
    print()
    if pub_val is not None:
        got = res[("acceptor 3'ss", "MaxEnt")][0]
        print("  cross-check: acceptor gene-avg here %.4f vs published %.4f (diff %+.4f)"
              % (got, pub_val, got - pub_val))
    print()

    # decision gate
    a_ga, _, a_p, _ = res[("acceptor 3'ss", "MaxEnt")]
    key5 = ("intervening 5'ss", "MaxEnt")
    print("=" * 96)
    print("DECISION GATE   (both must be > %.2f with p < %g on their expected side)"
          % (GATE_RATE, GATE_P))
    print("=" * 96)
    a_ok = (a_ga > GATE_RATE) and (a_p < GATE_P)
    print("  acceptor, canonical HIGHER : %.4f  %s  -> %s"
          % (a_ga, p_label(a_p), "PASS" if a_ok else "FAIL"))
    if key5 not in res:
        print("  intervening 5'ss           : NOT COMPUTABLE (interv_donor absent)")
        print("\nB+ CANNOT BE TESTED")
        return
    d_ga, _, d_p, _ = res[key5]
    d_ok = (d_ga > GATE_RATE) and (d_p < GATE_P)
    print("  5'ss, canonical LOWER      : %.4f  %s  -> %s"
          % (d_ga, p_label(d_p), "PASS" if d_ok else "FAIL"))
    print()
    if a_ok and d_ok:
        print("B+ HOLDS")
        print("  Both slopes are gene-fixed and on their expected side, so a two-slope")
        print("  panel is valid.")
    else:
        print("B+ FAILS, use acceptor-only")
        if not d_ok:
            side = "wrong direction" if d_ga < 0.5 else "too close to 0.5"
            print("  The 5'ss win-rate is %.4f (%s, %s). Gene-fixed, the intervening"
                  % (d_ga, side, p_label(d_p)))
            print("  donor does not separate classes the way the pooled view suggests.")
        if not a_ok:
            print("  The acceptor win-rate also failed: %.4f" % a_ga)


if __name__ == "__main__":
    main()
