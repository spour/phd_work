#!/usr/bin/env python
# is pas-proximal u1 density@4 real, or sequence composition?

import os
import sys
import csv
import math
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

WIN = 500
U1_STRONG = 4.0
CACHE = "u1_composition_cache.tsv"
OBSERVED = 0.619


def log(m):
    print(m, file=sys.stderr, flush=True)


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_label(p):
    if np.isnan(p):
        return "n/a"
    return "<1e-300" if not (p > 1e-300) else "%.1e" % p


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
    return two_sided_p((W - mu) / math.sqrt(var)) if var > 0 else float("nan")


def win(v, y, genes, match=None, tol=None):
    can, com = defaultdict(list), defaultdict(list)
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
        m = (np.ones((len(ci), len(ki)), bool) if match is None
             else np.abs(match[ci][:, None] - match[ki][None, :]) < tol)
        n = int(m.sum())
        if n == 0:
            continue
        w = float(((k > c) & m).sum() + 0.5 * ((k == c) & m).sum())
        rates.append(w / n)
        tw += w
        tp += n
    r = np.asarray(rates)
    return r, (tw / tp if tp else float("nan")), int(tp)


def resid(t, p):
    ok = np.isfinite(t) & np.isfinite(p)
    X = np.column_stack([np.ones(int(ok.sum())), p[ok]])
    b, *_ = np.linalg.lstsq(X, t[ok], rcond=None)
    o = np.full(t.size, np.nan)
    o[ok] = t[ok] - X @ b
    return o


_G = {}


def _init(cfg):
    from extract_splicing_features import GenomeFasta, load_u1_lookup
    _G["fa"] = GenomeFasta(cfg["fasta"])
    u1 = load_u1_lookup(cfg["u1_lookup"])
    L = len(next(iter(u1)))
    MAP = np.full(256, -1, np.int64)
    for i, ch in enumerate("ACGT"):
        MAP[ord(ch)] = i
    dense = np.full(4 ** L, np.nan, np.float32)
    powv = (4 ** np.arange(L - 1, -1, -1)).astype(np.int64)
    for kmer, val in u1.items():
        idx = 0
        bad = False
        for k, ch in enumerate(kmer):
            c = MAP[ord(ch)]
            if c < 0:
                bad = True
                break
            idx += int(c) * int(powv[k])
        if not bad:
            dense[idx] = -val
    _G.update(MAP=MAP, U1LEN=L, powv=powv, dense=dense, rng=np.random.default_rng(0))


def _u1(seq_arr):
    L, powv, dense = _G["U1LEN"], _G["powv"], _G["dense"]
    n = len(seq_arr)
    if n < L:
        return np.empty(0)
    codes = np.zeros(n - L + 1, np.int64)
    bad = np.zeros(n - L + 1, bool)
    for k in range(L):
        c = seq_arr[k:k + (n - L + 1)]
        codes += c * powv[k]
        bad |= (c < 0)
    return np.where(bad, np.nan, dense[np.clip(codes, 0, len(dense) - 1)])


def _work(rec):
    chrom, center, strand = rec
    if strand == "+":
        glo, ghi = center - WIN, center
    else:
        glo, ghi = center + 1, center + WIN + 1
    if glo < 0:
        return (np.nan,) * 4
    try:
        seq = _G["fa"].fetch(chrom, glo, ghi, strand=strand)
    except Exception:
        return (np.nan,) * 4
    if not seq or len(seq) < 50:
        return (np.nan,) * 4
    seq = seq.upper()
    arr = _G["MAP"][np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]
    good = arr >= 0
    if good.sum() < 50:
        return (np.nan,) * 4
    gc = float(((arr == 1) | (arr == 2)).sum()) / float(good.sum())
    v = _u1(arr)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan,) * 4
    kb = WIN / 1000.0
    dens = float((v >= U1_STRONG).sum()) / kb
    mx = float(v.max())
    sh = _G["rng"].permutation(arr)
    vs = _u1(sh)
    vs = vs[np.isfinite(vs)]
    dens_sh = float((vs >= U1_STRONG).sum()) / kb if vs.size else np.nan
    return (dens, mx, gc, dens_sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--fasta", default="../train/data/hg38.fa")
    ap.add_argument("--u1_lookup", default="../train/splice_train/data/u1_lookup.npy")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rescan", action="store_true")
    a = ap.parse_args()
    for p in (a.pos, a.fasta, a.u1_lookup):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)
    with open(a.pos, newline="") as fh:
        R = list(csv.DictReader(fh, delimiter="\t"))
    genes = np.array([r["gene"] for r in R])
    y = np.array([int(float(r["is_canonical"])) for r in R], dtype=np.int8)

    if os.path.isfile(CACHE) and not a.rescan:
        C = list(csv.DictReader(open(CACHE, newline=""), delimiter="\t"))
        if len(C) == len(R):
            dens = np.array([float(c["u1_density4"]) for c in C])
            gc = np.array([float(c["gc"]) for c in C])
            dsh = np.array([float(c["u1_density4_shuffled"]) for c in C])
            log("reused %s" % CACHE)
        else:
            C = None
    else:
        C = None
    if C is None:
        recs = [(r["chrom"], int(float(r["center"])), r["strand"]) for r in R]
        log("scanning (pA-%d, pA) windows for U1 + GC + shuffle ..." % WIN)
        t0 = time.time()
        import multiprocessing as mp
        out = np.full((len(recs), 4), np.nan)
        with mp.get_context("fork").Pool(a.jobs, initializer=_init,
                                         initargs=({"fasta": a.fasta,
                                                    "u1_lookup": a.u1_lookup},)) as pool:
            for i, v in enumerate(pool.imap(_work, recs, chunksize=64)):
                out[i] = v
                if i and i % 20000 == 0:
                    log("  %d/%d (%.1f min)" % (i, len(recs), (time.time() - t0) / 60))
        log("scan done in %.1f min" % ((time.time() - t0) / 60))
        dens, mx, gc, dsh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        with open(CACHE, "w") as fh:
            fh.write("gene\tis_canonical\tu1_density4\tu1_max\tgc\tu1_density4_shuffled\n")
            for i, r in enumerate(R):
                fh.write("%s\t%d\t%.6g\t%.6g\t%.6g\t%.6g\n"
                         % (r["gene"], y[i], dens[i], mx[i], gc[i], dsh[i]))
        log("cached -> %s" % CACHE)

    ok = np.isfinite(dens) & np.isfinite(gc)
    log("coverage: %d/%d peaks with U1 and GC" % (int(ok.sum()), len(R)))
    log("corr(U1 density@4, GC) = %+.4f" % np.corrcoef(dens[ok], gc[ok])[0, 1])
    log("GC mean canonical %.4f, competitor %.4f"
        % (np.nanmean(gc[y == 1]), np.nanmean(gc[y == 0])))

    def row(name, v, match=None, tol=None, note=""):
        r, pw, npair = win(v, y, genes, match, tol)
        if r.size < 20:
            print("  %-42s %9s %9s %11s %8s %10s %s"
                  % (name, "n/a", "n/a", "n/a", r.size, npair, note))
            return float("nan")
        ga = float(r.mean())
        p = wilcoxon_p(r - 0.5)
        print("  %-42s %9.4f %9.4f %11s %8d %10d %s"
              % (name, ga, pw, p_label(p), r.size, npair, note))
        return ga

    print("=" * 116)
    print("COMPOSITION CONTROL for PAS-proximal U1 density@4   (published within-gene %.3f)"
          % OBSERVED)
    print("  all rows: within-gene P(competitor > canonical | same gene)")
    print("=" * 116)
    print("  %-42s %9s %9s %11s %8s %10s" % ("feature", "gene-avg", "pair-wtd",
                                             "Wilcoxon p", "genes", "pairs"))
    v_u1 = row("U1 density@4 (the result)", dens)
    v_gc = row("TEST 1: GC content alone", gc)
    v_res = row("TEST 2: U1 density@4 residualised on GC", resid(dens, gc))
    rng = np.random.default_rng(0)
    rho = np.corrcoef(dens[ok], gc[ok])[0, 1]
    zg = (gc - np.nanmean(gc)) / np.nanstd(gc)
    synth = rho * zg + math.sqrt(max(0.0, 1 - rho ** 2)) * rng.normal(size=gc.size)
    v_syn = row("  synthetic control: rho*GC + noise", synth)
    v_syn_r = row("  synthetic, residualised on GC", resid(synth, gc))
    for tol in (0.05, 0.10):
        v_m = row("TEST 3: U1 density@4 | matched GC", dens, match=gc, tol=tol,
                  note="tol %.2f" % tol)
        if tol == 0.05:
            v_match = v_m
    v_sh = row("TEST 4: observed minus shuffled U1 density", dens - dsh)
    print()

    print("=" * 116)
    print("VERDICT")
    print("=" * 116)
    print("  TEST 1  GC alone: %.4f" % v_gc)
    if v_gc > 0.55:
        print("          GC by itself is competitor-higher. Composition is doing real work;")
        print("          the U1 result cannot be read at face value.")
    else:
        print("          GC alone is weak, so composition is not sufficient on its own.")
    print("  TEST 2  residualised: %.4f  |  synthetic %.4f -> residualised %.4f"
          % (v_res, v_syn, v_syn_r))
    valid = abs(v_syn_r - 0.5) < 0.03
    print("          synthetic %s -> residualising is %s here"
          % ("stays ~0.5" if valid else "ALSO MOVES", "valid" if valid else "NOT trustworthy"))
    print("  TEST 3  matched GC (tol 0.05): %.4f   <- DECIDES" % v_match)
    print("  TEST 4  observed minus shuffle: %.4f" % v_sh)
    print()
    if v_match >= 0.55:
        print("  SURVIVES. At matched GC the competitor still carries more strong-U1 density")
        print("  (%.4f). The PAS-proximal U1 signal is not sequence composition." % v_match)
    elif v_match > 0.5:
        print("  PARTLY SURVIVES. Matched-GC win-rate %.4f is above chance but below 0.55;" % v_match)
        print("  composition explains part of the raw %.3f." % OBSERVED)
    else:
        print("  FAILS. At matched GC the effect is gone (%.4f). The raw %.3f was composition."
              % (v_match, OBSERVED))


if __name__ == "__main__":
    main()
