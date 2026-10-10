#!/usr/bin/env python

import os
import sys
import math
import argparse
from collections import defaultdict

import numpy as np

Q_MAX, RATE_MIN = 0.01, 0.55


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


def bh_q(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.size, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return q
    o = idx[np.argsort(p[idx])]
    m = o.size
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = o[rank]
        val = min(prev, p[i] * m / (rank + 1))
        q[i] = prev = val
    return q


def win_rates(v, y, gidx, n_genes, competitor_higher):
    ok = np.isfinite(v)
    rates = np.full(n_genes, np.nan)
    tw = tp = 0.0
    can = defaultdict(list)
    com = defaultdict(list)
    for i in np.flatnonzero(ok):
        (can if y[i] == 1 else com)[gidx[i]].append(v[i])
    for g, cs in can.items():
        ks = com.get(g)
        if not ks:
            continue
        c = np.asarray(cs)[:, None]
        k = np.asarray(ks)[None, :]
        wins = float((c < k).sum() if competitor_higher else (c > k).sum())
        wins += 0.5 * float((c == k).sum())
        n = c.size * k.size
        rates[g] = wins / n
        tw += wins
        tp += n
    r = rates[np.isfinite(rates)]
    return r, (tw / tp if tp else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="gridsearch_cache.npz")
    ap.add_argument("--out", default="gridsearch_direction.tsv")
    a = ap.parse_args()
    if not os.path.isfile(a.cache):
        sys.exit("missing %s (run gridsearch_scan.py first)" % a.cache)
    z = np.load(a.cache, allow_pickle=True)
    M = z["M"]
    cols = [tuple(c.split("|")) for c in z["cols"]]
    genes = list(z["gene"])
    y = z["is_canonical"].astype(np.int8)
    uniq = {g: i for i, g in enumerate(sorted(set(genes)))}
    gidx = np.array([uniq[g] for g in genes])
    n_genes = len(uniq)
    print("[grid] %d peaks, %d genes, %d cells" % (len(genes), n_genes, len(cols)),
          file=sys.stderr)

    rows = []
    for j, (scorer, site, reg, enc) in enumerate(cols):
        comp_hi = (site == "donor")
        r, pw = win_rates(M[:, j].astype(float), y, gidx, n_genes, comp_hi)
        if r.size < 100:
            continue
        ga = float(r.mean())
        p = wilcoxon_p(r - 0.5)
        rows.append(dict(scorer=scorer, site=site, region=reg, encoding=enc,
                         win_gene_avg=ga, win_pair_wtd=pw, p=p, n_genes=int(r.size),
                         direction=("correct" if ga > 0.5 else "wrong")))
        if len(rows) % 100 == 0:
            print("  %d/%d cells" % (len(rows), len(cols)), file=sys.stderr)
    qs = bh_q([r["p"] for r in rows])
    for r, q in zip(rows, qs):
        r["q"] = q

    with open(a.out, "w") as fh:
        k = ["scorer", "site", "region", "encoding", "win_gene_avg", "win_pair_wtd",
             "p", "q", "n_genes", "direction"]
        fh.write("\t".join(k) + "\n")
        for r in rows:
            fh.write("\t".join(("%.6g" % r[c]) if isinstance(r[c], float) else str(r[c])
                               for c in k) + "\n")
    print("[grid] wrote %s (%d cells)" % (a.out, len(rows)), file=sys.stderr)

    by = {(r["scorer"], r["site"], r["region"], r["encoding"]): r for r in rows}
    keys = sorted({(r["site"], r["region"], r["encoding"]) for r in rows})

    def agree_table(site, title):
        print("\n" + "=" * 104)
        print(title)
        print("=" * 104)
        hits = []
        for s, reg, enc in keys:
            if s != site:
                continue
            m = by.get(("MaxEnt", s, reg, enc))
            sa = by.get(("SpliceAI", s, reg, enc))
            if not m or not sa:
                continue
            if (m["win_gene_avg"] >= RATE_MIN and sa["win_gene_avg"] >= RATE_MIN
                    and m["q"] < Q_MAX and sa["q"] < Q_MAX):
                hits.append((min(m["win_gene_avg"], sa["win_gene_avg"]), reg, enc, m, sa))
        if hits:
            hits.sort(reverse=True)
            print("  %-14s %-18s %9s %9s %10s %10s" % ("region", "encoding", "MaxEnt", "SpliceAI",
                                                       "q(ME)", "q(SA)"))
            for _w, reg, enc, m, sa in hits:
                print("  %-14s %-18s %9.4f %9.4f %10.1e %10.1e"
                      % (reg, enc, m["win_gene_avg"], sa["win_gene_avg"], m["q"], sa["q"]))
            return hits
        print("  NO AGREEING %s ENCODING" % site.upper())
        print("  (no cell has BOTH scorers on the correct side at >= %.2f with q < %.2f)"
              % (RATE_MIN, Q_MAX))
        worst = "SpliceAI"
        cand = [r for r in rows if r["site"] == site and r["scorer"] == worst]
        cand.sort(key=lambda r: -r["win_gene_avg"])
        print("\n  5 cells where %s comes closest to the correct direction:" % worst)
        print("  %-14s %-18s %9s %10s %9s" % ("region", "encoding", worst, "q", "MaxEnt"))
        for r in cand[:5]:
            m = by.get(("MaxEnt", site, r["region"], r["encoding"]))
            print("  %-14s %-18s %9.4f %10.1e %9s"
                  % (r["region"], r["encoding"], r["win_gene_avg"], r["q"],
                     ("%.4f" % m["win_gene_avg"]) if m else "n/a"))
        return []

    d_hits = agree_table("donor", "5'ss (DONOR) AGREEING CELLS   competitor-higher, both scorers")
    a_hits = agree_table("acceptor", "3'ss (ACCEPTOR) AGREEING CELLS   canonical-higher, both scorers")

    print("\n" + "=" * 104)
    print("HEADLINE")
    print("=" * 104)
    both = set((r, e) for _w, r, e, _m, _s in d_hits) & set((r, e) for _w, r, e, _m, _s in a_hits)
    if both:
        for reg, enc in sorted(both):
            print("  SINGLE ENCODING AGREEING ON BOTH SITE TYPES: region=%s encoding=%s" % (reg, enc))
    else:
        print("  No single (region, encoding) makes both scorers agree on BOTH site types.")
        if a_hits:
            w, reg, enc, m, sa = a_hits[0]
            print("  best 3'ss: region=%s encoding=%s  MaxEnt %.4f / SpliceAI %.4f"
                  % (reg, enc, m["win_gene_avg"], sa["win_gene_avg"]))
        else:
            print("  best 3'ss: none agreeing")
        if d_hits:
            w, reg, enc, m, sa = d_hits[0]
            print("  best 5'ss: region=%s encoding=%s  MaxEnt %.4f / SpliceAI %.4f"
                  % (reg, enc, m["win_gene_avg"], sa["win_gene_avg"]))
        else:
            print("  best 5'ss: none agreeing")

    print("\n" + "=" * 104)
    print("EACH SCORER'S SINGLE BEST 5'ss ENCODING (correct direction), regardless of agreement")
    print("=" * 104)
    for scorer in ("MaxEnt", "SpliceAI"):
        cand = [r for r in rows if r["site"] == "donor" and r["scorer"] == scorer
                and np.isfinite(r["q"])]
        if not cand:
            continue
        b = max(cand, key=lambda r: r["win_gene_avg"])
        print("  %-9s %-14s %-18s win %.4f (pair %.4f)  q %.1e  n_genes %d  [%s]"
              % (scorer, b["region"], b["encoding"], b["win_gene_avg"], b["win_pair_wtd"],
                 b["q"], b["n_genes"], b["direction"]))
    print("\n  and for the 3'ss:")
    for scorer in ("MaxEnt", "SpliceAI"):
        cand = [r for r in rows if r["site"] == "acceptor" and r["scorer"] == scorer
                and np.isfinite(r["q"])]
        if not cand:
            continue
        b = max(cand, key=lambda r: r["win_gene_avg"])
        print("  %-9s %-14s %-18s win %.4f (pair %.4f)  q %.1e  n_genes %d  [%s]"
              % (scorer, b["region"], b["encoding"], b["win_gene_avg"], b["win_pair_wtd"],
                 b["q"], b["n_genes"], b["direction"]))


if __name__ == "__main__":
    main()
