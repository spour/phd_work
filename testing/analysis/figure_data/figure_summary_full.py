#!/usr/bin/env python
# computes every statistic behind figure_summary_full and writes tidy tsvs

import os
import sys
import csv
import math
import argparse
from collections import defaultdict

import numpy as np

MIN_CELL = 200
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


# statistics


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_label(p):
    if p == 0.0:
        return "p < 1e-300"
    if p < 1e-300:
        return "p < 1e-300"
    if p < 0.001:
        return "p = %.0e" % p
    return "p = %.3f" % p


def logistic_irls(X, y, max_iter=60, tol=1e-10):
    n, k = X.shape
    beta = np.zeros(k)
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
    Xc, yc = M[ok], y[ok].astype(float)
    X = np.column_stack([np.ones(Xc.shape[0]), Xc])
    beta, se = logistic_irls(X, yc)
    j = names.index(want) + 1
    b, s = beta[j], se[j]
    p = two_sided_p(b / s)
    return {
        "block": "effect", "effect_label": label, "scorer": scorer,
        "estimate": math.exp(b), "ci_lo": math.exp(b - Z95 * s),
        "ci_hi": math.exp(b + Z95 * s), "p": p, "p_label": p_label(p),
        "null_value": 1.0, "n": int(ok.sum()),
    }


def within_gene_rates(score, y, genes):
    by = defaultdict(lambda: ([], []))
    for i in range(len(score)):
        if np.isnan(score[i]):
            continue
        by[genes[i]][0 if y[i] == 1 else 1].append(score[i])
    rates = []
    for _g, (can, comp) in by.items():
        if not can or not comp:
            continue
        c = np.asarray(can)[:, None]
        k = np.asarray(comp)[None, :]
        rates.append(float((c > k).sum() + 0.5 * (c == k).sum()) / (c.size * k.size))
    return np.asarray(rates)


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


# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_ucube_meta.tsv")
    ap.add_argument("--indir", default=".")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--prefix", default="fig3")
    args = ap.parse_args()
    outdir = args.outdir or args.indir

    def pf(n):
        return os.path.join(args.indir, "%s_%s" % (args.prefix, n))

    required = [pf("acc_x_interv.tsv"),
                os.path.join(args.indir, "sa_acc_x_don_final.tsv"),
                os.path.join(args.indir, "sa_peak_scalars_final.tsv")]
    peak_tbl = args.table if os.path.isfile(args.table) \
        else os.path.join(args.indir, args.table)
    required.append(peak_tbl)
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        print("MISSING REQUIRED INPUTS (%d):" % len(missing), file=sys.stderr)
        for m in missing:
            print("  %s" % m, file=sys.stderr)
        sys.exit(1)
    log("inputs present (%d)" % len(required))

    # per-peak maxent
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

    # per-peak spliceai (identical row order, verified on write)
    S = read_tsv(os.path.join(args.indir, "sa_peak_scalars_final.tsv"))
    if len(S) != len(P):
        sys.exit("sa_peak_scalars_final.tsv has %d rows, %s has %d"
                 % (len(S), peak_tbl, len(P)))
    if [r["gene"] for r in S] != genes:
        sys.exit("sa_peak_scalars_final.tsv gene order does not match %s" % peak_tbl)
    sacc = np.array([fnum(r["sa_acc_max"]) for r in S])
    sdon = np.array([fnum(r["sa_don_sum"]) for r in S])

    n_can = int((y == 1).sum())
    n_comp = int((y == 0).sum())
    b = n_can / float(n_can + n_comp)
    log("peaks %d (%d real / %d competitor), base rate %.5f"
        % (len(P), n_can, n_comp, b))

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
                log("  within-gene %s read from %s (CI still computed here)"
                    % (scorer, src))
        return {
            "block": "ranking", "effect_label": "within-gene win-rate, acceptor",
            "scorer": scorer, "estimate": est,
            "ci_lo": max(0.0, est - half), "ci_hi": min(1.0, est + half),
            "p": p, "p_label": p_label(p), "null_value": 0.5, "n": int(r.size),
            "source": src,
        }
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
                        r["null_value"], r["n"], r.get("source", "computed")))

    log("")
    log("STATS_FOREST")
    log("  %-8s %-28s %-9s %10s %20s %14s %8s"
        % ("block", "effect", "scorer", "estimate", "95% CI", "p", "n"))
    for r in rows:
        log("  %-8s %-28s %-9s %10.4f  [%8.4f,%8.4f] %14s %8d"
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
            if "log2_or" in r and r["log2_or"] not in ("", None):
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
