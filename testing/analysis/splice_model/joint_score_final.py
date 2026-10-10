#!/usr/bin/env python
# joint "acceptor defines minus donor suppresses" score for both scorers

import os
import sys
import csv
import time
import hashlib
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import pyBigWig
except ImportError:
    sys.exit("pyBigWig not installed. Run: pip install pyBigWig")

BASE_URL = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/spliceAi/wildtype/"
BW_NAMES = {
    ("acceptor", "+"): "spliceAiAcceptorPlus.bw",
    ("acceptor", "-"): "spliceAiAcceptorMinus.bw",
    ("donor", "+"): "spliceAiDonorPlus.bw",
    ("donor", "-"): "spliceAiDonorMinus.bw",
}
BIGWIGS = {}
SIGNALS = ("acceptor", "donor")
UP = 2000
CACHE = "sa_peak_scalars_final.tsv"

EXPECT_SA_ACC = 0.737
EXPECT_ME_ACC = 0.606
TOL = 0.02
N_THREADS = 12
HOLDOUT_SEED = 0


def log(m):
    print(m, file=sys.stderr, flush=True)


def auroc(scores, labels):
    ok = ~np.isnan(scores)
    s, y = scores[ok], labels[ok]
    n = s.size
    n_pos = int((y == 1).sum())
    if n_pos == 0 or n_pos == n:
        return float("nan"), n
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    newgrp = np.r_[True, ss[1:] != ss[:-1]]
    grp = np.cumsum(newgrp)
    starts = np.r_[np.nonzero(newgrp)[0], n]
    avg = (starts[:-1] + starts[1:] + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[grp - 1]
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * (n - n_pos)), n


def zfit(v, mask):
    t = v[mask & ~np.isnan(v)]
    mu = float(t.mean()) if t.size else 0.0
    sd = float(t.std())
    return mu, (sd if sd > 0 else 1.0)


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


def scan(peaks):
    n = len(peaks)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    order = sorted(range(n), key=lambda i: (peaks[i][1], peaks[i][2]))
    jobs = [order[k::N_THREADS] for k in range(N_THREADS)]
    jobs = [j for j in jobs if j]

    def run(idxs):
        handles = {}
        try:
            for k, i in enumerate(idxs):
                _g, chrom, center, strand, _y = peaks[i]
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
                    a = oriented(bw, chrom, center, strand, clen)
                    cov = ~np.isnan(a)
                    if not cov.any():
                        continue
                    v = a[cov]
                    out[i, s_i] = v.max() if sig == "acceptor" else v.sum()
                if k and k % 5000 == 0:
                    log("    %d/%d" % (k, len(idxs)))
        finally:
            for bw, _ in handles.values():
                bw.close()
        return len(idxs)

    done = 0
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for got in ex.map(run, jobs):
            done += got
            log("  scanned %d/%d" % (done, n))
    return out[:, 0], out[:, 1]


def within_gene(score, y, genes, mask):
    by = defaultdict(lambda: ([], []))
    for i in np.nonzero(mask)[0]:
        if np.isnan(score[i]):
            continue
        by[genes[i]][0 if y[i] == 1 else 1].append(score[i])
    rates, tot_w, tot_p = [], 0.0, 0
    for _g, (can, comp) in by.items():
        if not can or not comp:
            continue
        c = np.asarray(can)[:, None]
        k = np.asarray(comp)[None, :]
        w = float((c > k).sum() + 0.5 * (c == k).sum())
        p = c.size * k.size
        rates.append(w / p)
        tot_w += w
        tot_p += p
    if not rates:
        return float("nan"), float("nan"), 0, 0
    return float(np.mean(rates)), tot_w / tot_p, len(rates), tot_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="fig3_ucube_meta.tsv")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--threads", type=int, default=N_THREADS)
    args = ap.parse_args()
    globals()["N_THREADS"] = max(1, args.threads)

    if not os.path.isfile(args.table):
        sys.exit("missing input: %s" % args.table)
    rows = list(csv.DictReader(open(args.table, newline=""), delimiter="\t"))
    cols = set(rows[0].keys()) if rows else set()

    don_col, fallback = "interv_donor", False
    if don_col not in cols:
        alt = [c for c in ("interv_sum5", "interv_density5") if c in cols]
        if not alt:
            sys.exit("no interv_donor and no fallback (interv_sum5) in %s" % args.table)
        don_col, fallback = alt[0], True

    need = {"gene", "is_canonical", "acc"}
    miss = need - cols
    if miss:
        sys.exit("%s lacks columns: %s" % (args.table, sorted(miss)))

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    genes = [r["gene"] for r in rows]
    y = np.array([int(float(r["is_canonical"])) for r in rows], dtype=np.int8)
    acc = np.array([f(r["acc"]) for r in rows])
    ivd = np.array([f(r[don_col]) for r in rows])
    n = len(rows)

    split = np.array([int(hashlib.md5(("%d:%s" % (HOLDOUT_SEED, g)).encode())
                          .hexdigest()[:8], 16) & 1 for g in genes], dtype=np.int8)
    tr, te = split == 0, split == 1

    # spliceai scalars: cache or scan
    have_coords = {"chrom", "center", "strand"} <= cols
    sa_acc = sa_don = None
    if os.path.isfile(args.cache) and not args.rescan:
        c = list(csv.DictReader(open(args.cache, newline=""), delimiter="\t"))
        if len(c) == n:
            sa_acc = np.array([f(r["sa_acc_max"]) for r in c])
            sa_don = np.array([f(r["sa_don_sum"]) for r in c])
            log("reused cached per-peak scalars: %s" % args.cache)
        else:
            log("cache %s has %d rows, table has %d; rescanning"
                % (args.cache, len(c), n))
    if sa_acc is None:
        if not have_coords:
            sys.exit("no cache and %s lacks chrom/center/strand; run with "
                     "fig3_ucube_meta.tsv" % args.table)
        missing = []
        for key, nm in BW_NAMES.items():
            p = os.path.join(args.bw_dir, nm)
            if os.path.isfile(p):
                BIGWIGS[key] = p
            else:
                missing.append(p)
        if missing:
            print("MISSING bigWigs:", file=sys.stderr)
            for m in missing:
                print("  %s" % m, file=sys.stderr)
            sys.exit(1)
        peaks = [(r["gene"], r["chrom"], int(float(r["center"])), r["strand"],
                  int(float(r["is_canonical"]))) for r in rows]
        log("no cache; scanning SpliceAI once (%d peaks) ..." % n)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sa_acc, sa_don = scan(peaks)
        log("scan done in %.1f min" % ((time.time() - t0) / 60.0))
        with open(args.cache, "w") as fh:
            fh.write("gene\tchrom\tcenter\tstrand\tis_canonical\tsa_acc_max\tsa_don_sum\n")
            for i, r in enumerate(rows):
                fh.write("%s\t%s\t%s\t%s\t%d\t%.6g\t%.6g\n"
                         % (r["gene"], r["chrom"], r["center"], r["strand"],
                            y[i], sa_acc[i], sa_don[i]))
        log("cached -> %s" % args.cache)

    # composites, z fit on train only
    def z(v):
        mu, sd = zfit(v, tr)
        return (v - mu) / sd
    sa_joint = z(sa_acc) - z(sa_don)
    me_joint = z(acc) - z(ivd)

    # sanity
    a_sa_acc, _ = auroc(sa_acc[te], y[te])
    a_me_acc, _ = auroc(acc[te], y[te])
    log("")
    log("SANITY (test half)")
    log("  sa_acc_max %.4f (expect ~%.3f) | acc %.4f (expect ~%.3f)"
        % (a_sa_acc, EXPECT_SA_ACC, a_me_acc, EXPECT_ME_ACC))
    bad = []
    if abs(a_sa_acc - EXPECT_SA_ACC) > TOL:
        bad.append("sa_acc_max %.4f vs %.3f" % (a_sa_acc, EXPECT_SA_ACC))
    if abs(a_me_acc - EXPECT_ME_ACC) > TOL:
        bad.append("acc %.4f vs %.3f" % (a_me_acc, EXPECT_ME_ACC))
    if bad:
        sys.exit("STOP: marginal(s) off expectation: %s" % "; ".join(bad))
    log("  -> marginals reproduce.")

    print("=" * 78)
    print("Joint 'acceptor defines minus donor suppresses' score, both scorers")
    print("=" * 78)
    print("peaks %d (%d canonical / %d competitor), genes %d"
          % (n, int((y == 1).sum()), int((y == 0).sum()), len(set(genes))))
    print("split: md5(gene) seed %d -> train %d / test %d peaks"
          % (HOLDOUT_SEED, int(tr.sum()), int(te.sum())))
    print("z parameters fit on TRAIN only, applied to test.")
    print("donor-suppression column: %s%s"
          % (don_col, "   <-- FALLBACK" if fallback else ""))
    if fallback:
        print("MISMATCH NOTE: interv_donor absent from %s; used %s instead."
              % (args.table, don_col))
        print("  This is a different feature. Do not compare me_joint against runs")
        print("  that used interv_donor.")
    nan_iv = int(np.isnan(ivd).sum())
    print("%s missing on %d peaks (%.2f%%); those drop from me_joint."
          % (don_col, nan_iv, 100.0 * nan_iv / n))
    print()

    print("1. POOLED TEST AUROC (raw, canonical = positive)")
    print("  %-28s %9s" % ("score", "auroc"))
    items = [
        ("sa_acc_max  (SpliceAI marg)", sa_acc),
        ("sa_don_sum  (SpliceAI marg)", sa_don),
        ("sa_joint    = z(acc)-z(don)", sa_joint),
        ("acc         (MaxEnt marg)", acc),
        ("%-11s (MaxEnt marg)" % don_col[:11], ivd),
        ("me_joint    = z(acc)-z(don)", me_joint),
    ]
    res = {}
    for nm, v in items:
        a, nn = auroc(v[te], y[te])
        res[nm] = a
        print("  %-28s %9.4f" % (nm, a))
    print("  baselines: sa_acc_max %.3f, MaxEnt acc %.3f" % (EXPECT_SA_ACC, EXPECT_ME_ACC))
    print()

    a_sa_joint, _ = auroc(sa_joint[te], y[te])
    a_me_joint, _ = auroc(me_joint[te], y[te])

    print("2. DOES SUBTRACTING THE DONOR TERM ADD DISCRIMINATION?")
    d_sa = a_sa_joint - a_sa_acc
    d_me = a_me_joint - a_me_acc
    for nm, mg, jt, d in (("SpliceAI", a_sa_acc, a_sa_joint, d_sa),
                          ("MaxEnt", a_me_acc, a_me_joint, d_me)):
        verdict = "ADDS" if d > 0.002 else ("NEUTRAL" if d > -0.002 else "HURTS")
        print("  %-9s acceptor %.4f -> joint %.4f   delta %+.4f   %s"
              % (nm, mg, jt, d, verdict))
    print()

    print("3. WITHIN-GENE WIN-RATE OF THE COMPOSITE SCORE")
    print("   P(canonical > competitor | same gene), 0.5 credit for ties.")
    print("   This is the COMPOSITE-score win-rate. It is NOT the acceptor-only")
    print("   acc_paired headline and should not be quoted as it.")
    print()
    print("  %-26s %10s %10s %8s %10s" % ("score / half", "gene-avg", "pair-wtd",
                                          "genes", "pairs"))
    wg = {}
    for nm, v in (("sa_joint", sa_joint), ("me_joint", me_joint),
                  ("sa_acc_max (ref)", sa_acc), ("acc (ref)", acc)):
        for half, mask in (("test", te), ("all", np.ones(n, dtype=bool))):
            ga, pw, ng, npair = within_gene(v, y, genes, mask)
            wg[(nm, half)] = ga
            print("  %-26s %10.4f %10.4f %8d %10d"
                  % ("%s / %s" % (nm, half), ga, pw, ng, npair))
    print()

    # compact final block
    print("=" * 78)
    print("  %-9s %14s %9s %9s %14s" % ("scorer", "marginal_acc", "joint",
                                        "delta", "within_gene"))
    print("  %-9s %14.3f %9.3f %+9.3f %14.3f"
          % ("SpliceAI", a_sa_acc, a_sa_joint, d_sa, wg[("sa_joint", "test")]))
    print("  %-9s %14.3f %9.3f %+9.3f %14.3f"
          % ("MaxEnt", a_me_acc, a_me_joint, d_me, wg[("me_joint", "test")]))
    print("  within_gene = composite score, gene-averaged, test half")
    print("=" * 78)


if __name__ == "__main__":
    main()
