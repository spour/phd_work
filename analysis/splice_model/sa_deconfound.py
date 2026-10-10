#!/usr/bin/env python
# can any architecture-removal make spliceai's intervening donor competitor-favouring

import os
import sys
import csv
import math
import argparse
from collections import defaultdict

import numpy as np

RAW_BASELINE = 0.447
MAXENT_TARGET = 0.554
AGREE_MIN = 0.55


def log(m):
    print(m, file=sys.stderr, flush=True)


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_label(p):
    if np.isnan(p):
        return "n/a"
    if not (p > 1e-300):
        return "<1e-300"
    return "%.1e" % p


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
    if var <= 0:
        return float("nan")
    return two_sided_p((W - mu) / math.sqrt(var))


def win_competitor_higher(v, y, genes, match=None, tol=None):
    can = defaultdict(list)
    com = defaultdict(list)
    for i in range(v.size):
        if not np.isfinite(v[i]):
            continue
        if match is not None and not np.isfinite(match[i]):
            continue
        (can if y[i] == 1 else com)[genes[i]].append(i)
    rates, tw, tp = [], 0.0, 0
    for g, ci in can.items():
        ki = com.get(g)
        if not ki:
            continue
        c = v[ci][:, None]
        k = v[ki][None, :]
        if match is None:
            m = np.ones((len(ci), len(ki)), bool)
        else:
            m = np.abs(match[ci][:, None] - match[ki][None, :]) < tol
        n = int(m.sum())
        if n == 0:
            continue
        wins = float(((k > c) & m).sum() + 0.5 * ((k == c) & m).sum())
        rates.append(wins / n)
        tw += wins
        tp += n
    r = np.asarray(rates)
    return r, (tw / tp if tp else float("nan")), int(tp)


def zs(v):
    m = np.isfinite(v)
    mu, sd = v[m].mean(), v[m].std()
    return (v - mu) / (sd if sd > 0 else 1.0)


def ols_resid(target, preds):
    ok = np.isfinite(target)
    for p in preds:
        ok &= np.isfinite(p)
    X = np.column_stack([np.ones(int(ok.sum()))] + [p[ok] for p in preds])
    beta, *_ = np.linalg.lstsq(X, target[ok], rcond=None)
    out = np.full(target.size, np.nan)
    out[ok] = target[ok] - X @ beta
    return out, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="gridsearch_cache.npz")
    ap.add_argument("--up", default="sa_upstream_anchor_donor_cache.tsv")
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    a = ap.parse_args()
    for p in (a.cache, a.up, a.pos):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)

    z = np.load(a.cache, allow_pickle=True)
    M = z["M"]
    cols = {tuple(c.split("|")): i for i, c in enumerate(z["cols"])}
    genes = np.array(list(z["gene"]))
    y = z["is_canonical"].astype(np.int8)

    with open(a.up, newline="") as fh:
        U = list(csv.DictReader(fh, delimiter="\t"))
    if len(U) != genes.size or [r["gene"] for r in U] != list(genes):
        sys.exit("row alignment mismatch: %s vs cache" % a.up)

    def col(scorer, site, reg, enc):
        k = (scorer, site, reg, enc)
        if k not in cols:
            sys.exit("cache lacks column %s" % (k,))
        return M[:, cols[k]].astype(float)

    sa_acc = col("SpliceAI", "acceptor", "up_2000", "max")
    sa_int = col("SpliceAI", "donor", "interv", "max")
    me_int = col("MaxEnt", "donor", "interv", "max")
    sa_up = np.array([float(r["sa_up_anchor_donor"]) if r["sa_up_anchor_donor"] not in ("", "nan")
                      else np.nan for r in U])

    log("peaks %d (%d canonical / %d competitor), genes %d"
        % (genes.size, int((y == 1).sum()), int((y == 0).sum()), len(set(genes))))
    for nm, v in (("sa_acc", sa_acc), ("sa_don_up", sa_up), ("sa_don_int", sa_int),
                  ("me_don_int", me_int)):
        log("  %-11s finite %d (%.2f%%)" % (nm, int(np.isfinite(v).sum()),
                                            100 * np.isfinite(v).mean()))
    log("  corr(sa_don_int, sa_don_up) = %.4f   corr(sa_don_int, sa_acc) = %.4f"
        % (np.corrcoef(*[x[np.isfinite(sa_int) & np.isfinite(sa_up)]
                         for x in (sa_int, sa_up)])[0, 1],
           np.corrcoef(*[x[np.isfinite(sa_int) & np.isfinite(sa_acc)]
                         for x in (sa_int, sa_acc)])[0, 1]))

    rows = []

    def add(fix, feat, v, match=None, tol=None, note=""):
        r, pw, npair = win_competitor_higher(v, y, genes, match, tol)
        if r.size < 20:
            rows.append((fix, feat, float("nan"), float("nan"), float("nan"),
                         "too few genes", r.size, npair, note))
            return
        ga = float(r.mean())
        p = wilcoxon_p(r - 0.5)
        d = "CORRECT (comp higher)" if ga > 0.5 else "wrong (canon higher)"
        rows.append((fix, feat, ga, pw, p, d, r.size, npair, note))

    add("--", "sa_don_int (raw)", sa_int)
    add("--", "me_don_int (MaxEnt target)", me_int)

    r_both, b_both = ols_resid(sa_int, [sa_acc, sa_up])
    r_acc, _ = ols_resid(sa_int, [sa_acc])
    r_up, b_up = ols_resid(sa_int, [sa_up])
    add("1", "resid ~ sa_acc + sa_don_up", r_both,
        note="beta acc %.3f, up %.3f" % (b_both[1], b_both[2]))
    add("1", "resid ~ sa_acc only", r_acc)
    add("1", "resid ~ sa_don_up only", r_up)

    gap = zs(me_int) - zs(sa_int)
    add("2", "z(me_don_int) - z(sa_don_int)", gap)

    for tol in (0.1, 0.2):
        add("3", "sa_don_int | matched sa_don_up", sa_int, match=sa_up, tol=tol,
            note="tol %.1f" % tol)

    ok = np.isfinite(sa_int) & np.isfinite(sa_up)
    beta = np.cov(sa_int[ok], sa_up[ok])[0, 1] / np.var(sa_up[ok])
    orth = sa_int - beta * sa_up
    add("4", "sa_don_int - beta*sa_don_up", orth, note="beta %.4f" % beta)

    print("=" * 118)
    print("SPLICEAI INTERVENING DONOR: architecture-removal attempts")
    print("target for every row: P(competitor > canonical | same gene) > 0.5")
    print("=" * 118)
    print("  %-4s %-32s %9s %9s %11s  %-22s %7s %10s" %
          ("fix", "feature", "gene-avg", "pair-wtd", "Wilcoxon p", "direction", "genes", "pairs"))
    for fix, feat, ga, pw, p, d, ng, npair, note in rows:
        print("  %-4s %-32s %9s %9s %11s  %-22s %7d %10d%s"
              % (fix, feat,
                 ("%.4f" % ga) if np.isfinite(ga) else "n/a",
                 ("%.4f" % pw) if np.isfinite(pw) else "n/a",
                 p_label(p), d, ng, npair, ("   [%s]" % note if note else "")))
    print()
    print("  reference: raw sa_don_int %.3f (wrong)  |  MaxEnt intervening %.3f (correct)"
          % (RAW_BASELINE, MAXENT_TARGET))
    print()

    print("=" * 118)
    print("VERDICTS")
    print("=" * 118)
    flipped = []
    for fix, feat, ga, pw, p, d, ng, npair, note in rows:
        if fix == "--" or not np.isfinite(ga):
            continue
        if ga > 0.5 and np.isfinite(p) and p < 1e-5:
            v = "FLIPPED"
            flipped.append((ga, fix, feat))
        elif ga > 0.5:
            v = "flipped but not significant"
        else:
            v = "STILL WRONG"
        print("  fix %-3s %-32s %s (%.4f)" % (fix, feat, v, ga))
    print()
    agree = [t for t in flipped if t[0] >= AGREE_MIN]
    if agree:
        agree.sort(reverse=True)
        ga, fix, feat = agree[0]
        print("  ANY FIX REACHES AGREEMENT (>= %.2f): YES" % AGREE_MIN)
        print("  best: fix %s, %s, win-rate %.4f" % (fix, feat, ga))
        print("  -> B+ can be built two-scorer on this feature.")
    elif flipped:
        ga, fix, feat = max(flipped)
        print("  ANY FIX REACHES AGREEMENT (>= %.2f): NO" % AGREE_MIN)
        print("  best flip is fix %s, %s, at %.4f, above 0.5 but below the %.2f bar."
              % (fix, feat, ga, AGREE_MIN))
        print("  -> not enough to claim the scorers agree; MaxEnt carries the signal.")
    else:
        print("  ANY FIX REACHES AGREEMENT (>= %.2f): NO" % AGREE_MIN)
        print("  No fix moves SpliceAI's intervening donor above 0.5.")
        print("  -> SpliceAI has no recoverable telescripting signal; MaxEnt carries it.")


if __name__ == "__main__":
    main()
