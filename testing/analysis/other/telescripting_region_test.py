#!/usr/bin/env python
# diagnostic only, no panel

import os
import sys
import csv
import math
import time
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

UP = 2000
CACHE = "sa_upstream_anchor_donor_cache.tsv"
BW = {"+": "spliceAiDonorPlus.bw", "-": "spliceAiDonorMinus.bw"}


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


def win_rate(score, y, genes, competitor_higher):
    by = defaultdict(lambda: ([], []))
    for i in range(len(score)):
        if np.isnan(score[i]):
            continue
        by[genes[i]][0 if y[i] == 1 else 1].append(score[i])
    rates, tw, tp = [], 0.0, 0
    for _g, (can, comp) in by.items():
        if not can or not comp:
            continue
        c = np.asarray(can)[:, None]
        k = np.asarray(comp)[None, :]
        wins = float((c < k).sum() if competitor_higher else (c > k).sum())
        wins += 0.5 * float((c == k).sum())
        n = c.size * k.size
        rates.append(wins / n)
        tw += wins
        tp += n
    r = np.asarray(rates)
    return r, (tw / tp if tp else float("nan"))


def scan_upstream_of_anchor(rows, bw_dir, threads):
    import pyBigWig
    paths = {}
    for s, nm in BW.items():
        p = os.path.join(bw_dir, nm)
        if not os.path.isfile(p):
            sys.exit("missing %s" % p)
        paths[s] = p
    n = len(rows)
    out = np.full(n, np.nan)

    def run(idxs):
        h = {}
        try:
            for i in idxs:
                r = rows[i]
                anc = fnum(r["anchor_3ss_pos"])
                if not math.isfinite(anc):
                    continue
                anc = int(anc)
                strand = r["strand"]
                lo, hi = (anc - UP, anc) if strand == "+" else (anc, anc + UP)
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
                    continue
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
    log("scanning donor bigWigs over (anchor-2kb, anchor) windows ...")
    t0 = time.time()
    done = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            for got in ex.map(run, jobs):
                done += got
                log("  scanned %d/%d" % (done, n))
    log("scan done in %.1f min" % ((time.time() - t0) / 60.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--sa-interv", default="sa_interv_donor_cache.tsv")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    for p in (args.pos, args.sa_interv):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)
    P = read_tsv(args.pos)
    S = read_tsv(args.sa_interv)
    if len(P) != len(S) or [r["gene"] for r in P] != [r["gene"] for r in S]:
        sys.exit("row alignment mismatch between %s and %s" % (args.pos, args.sa_interv))

    genes = [r["gene"] for r in P]
    y = np.array([int(float(r["is_canonical"])) for r in P], dtype=np.int8)
    me_iv = np.array([fnum(r["interv_donor"]) for r in P])
    sa_iv = np.array([fnum(r["sa_interv_donor"]) for r in S])
    log("peaks %d (%d real / %d competitor), genes %d"
        % (len(P), int((y == 1).sum()), int((y == 0).sum()), len(set(genes))))

    if os.path.isfile(CACHE):
        C = read_tsv(CACHE)
        if len(C) == len(P) and [c["gene"] for c in C] == genes:
            sa_up = np.array([fnum(c["sa_up_anchor_donor"]) for c in C])
            log("reused %s" % CACHE)
        else:
            sa_up = None
    else:
        sa_up = None
    if sa_up is None:
        sa_up = scan_upstream_of_anchor(P, args.bw_dir, max(1, args.threads))
        with open(CACHE, "w") as fh:
            fh.write("gene\tchrom\tcenter\tstrand\tis_canonical\tanchor_3ss_pos\tsa_up_anchor_donor\n")
            for i, r in enumerate(P):
                fh.write("%s\t%s\t%s\t%s\t%d\t%s\t%.6g\n"
                         % (r["gene"], r["chrom"], r["center"], r["strand"], y[i],
                            r["anchor_3ss_pos"], sa_up[i]))
        log("cached -> %s" % CACHE)

    tests = [
        ("MaxEnt   intervening (anchor,pA)", me_iv, True,  "competitor higher"),
        ("SpliceAI intervening (anchor,pA)", sa_iv, True,  "competitor higher"),
        ("SpliceAI upstream-of-anchor 2kb",  sa_up, False, "canonical higher"),
    ]
    print("=" * 100)
    print("WITHIN-GENE WIN-RATE BY REGION  (each canonical vs its own gene's competitors)")
    print("=" * 100)
    print("  %-34s %10s %10s %12s %8s  %s"
          % ("feature / region", "gene-avg", "pair-wtd", "Wilcoxon p", "genes", "expected"))
    res = {}
    for name, v, comp_hi, exp in tests:
        r, pw = win_rate(v, y, genes, comp_hi)
        ga = float(r.mean())
        p = wilcoxon_signed_rank_p(r - 0.5)
        res[name] = (ga, pw, p)
        flag = "as expected" if ga > 0.5 else "OPPOSITE"
        print("  %-34s %10.4f %10.4f %12s %8d  %s (%s)"
              % (name, ga, pw, p_label(p), r.size, exp, flag))
    print()
    print("  coverage: MaxEnt intervening %d, SpliceAI intervening %d, SpliceAI upstream %d"
          % (int(np.isfinite(me_iv).sum()), int(np.isfinite(sa_iv).sum()),
             int(np.isfinite(sa_up).sum())))
    print()

    m_ga = res["MaxEnt   intervening (anchor,pA)"][0]
    s_ga = res["SpliceAI intervening (anchor,pA)"][0]
    u_ga = res["SpliceAI upstream-of-anchor 2kb"][0]
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print("  Q1. Do both scorers agree (competitor higher) in the STRICT intervening region?")
    if m_ga > 0.5 and s_ga > 0.5:
        print("      YES. MaxEnt %.4f and SpliceAI %.4f are both above 0.5." % (m_ga, s_ga))
    else:
        who = "SpliceAI" if s_ga <= 0.5 else "MaxEnt"
        print("      NO. MaxEnt %.4f, SpliceAI %.4f. %s is on the wrong side of 0.5:"
              % (m_ga, s_ga, who))
        print("      within a gene the CANONICAL peak carries the stronger SpliceAI"
              if who == "SpliceAI" else "")
        print("      intervening donor %.1f%% of the time." % (100 * (1 - s_ga))
              if who == "SpliceAI" else "")
    print()
    print("  Q2. Does the upstream-of-anchor donor explain SpliceAI's marginal sign?")
    print("      upstream-of-anchor, canonical higher: %.4f (%s)"
          % (u_ga, p_label(res["SpliceAI upstream-of-anchor 2kb"][2])))
    if u_ga > 0.55 and s_ga <= 0.5:
        print("      PARTLY. The upstream donor is canonical-favouring, consistent with the")
        print("      2 kb sum being dominated by the exon-DEFINING donor. But it is not the")
        print("      whole story: the strict intervening region is ALSO canonical-favouring")
        print("      in SpliceAI (%.4f), so removing the upstream donor does not flip it."
              % s_ga)
    elif u_ga > 0.55:
        print("      The upstream donor is canonical-favouring as predicted.")
    else:
        print("      NO. The upstream-of-anchor donor is %.4f, not clearly canonical-favouring,"
              % u_ga)
        print("      so it does not explain the marginal sign.")


if __name__ == "__main__":
    main()
