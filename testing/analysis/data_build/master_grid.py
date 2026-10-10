#!/usr/bin/env python
# score maxent, u1 and spliceai identically across every region x reduction x threshold

import os
import sys
import csv
import math
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

U_DN = 1000
U_ANCH_EXTRA = 2000
U1_THR = (4.0, 6.0, 8.0, 10.0)
BASE_RED = ("max", "mean", "sum")
THR_RED = ("density_per_kb", "count_above", "nearest_dist")
MIN_GENES = 100
NOISE_BAND = 0.02

REGION_SPECS = [
    ("interv",       lambda A: (1, max(1, A - 1))),
    ("up_100",       lambda A: (1, 100)),
    ("up_250",       lambda A: (1, 250)),
    ("up_500",       lambda A: (1, 500)),
    ("up_1000",      lambda A: (1, 1000)),
    ("up_2000",      lambda A: (1, 2000)),
    ("upanch_500",   lambda A: (A + 1, A + 500)),
    ("upanch_1000",  lambda A: (A + 1, A + 1000)),
    ("upanch_2000",  lambda A: (A + 1, A + 2000)),
    ("dn_100",       lambda A: (-100, -1)),
    ("dn_250",       lambda A: (-250, -1)),
    ("dn_500",       lambda A: (-500, -1)),
    ("dn_1000",      lambda A: (-1000, -1)),
]
U1_ENC = list(BASE_RED) + ["%s@%g" % (r, t) for r in THR_RED for t in U1_THR]
U1_COLS = [(r, e) for r, _ in REGION_SPECS for e in U1_ENC]


def log(m):
    print(m, file=sys.stderr, flush=True)


# statistics

def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_label(p):
    if not np.isfinite(p):
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


def bh_q(p):
    p = np.asarray(p, float)
    q = np.full(p.size, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return q
    o = idx[np.argsort(p[idx])]
    m = o.size
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = prev
    return q


def auroc(v, y):
    ok = np.isfinite(v)
    x, yy = v[ok], y[ok]
    n1 = int((yy == 1).sum())
    n0 = int((yy == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), n1 + n0
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    n = xs.size
    st = np.r_[True, xs[1:] != xs[:-1]]
    g = np.cumsum(st)
    bnd = np.r_[np.nonzero(st)[0], n]
    av = (bnd[:-1] + bnd[1:] + 1) / 2.0
    r = np.empty(n)
    r[o] = av[g - 1]
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0), n1 + n0


def fold_enrich(v, y, base, nbin=10):
    ok = np.isfinite(v)
    x, yy = v[ok], y[ok]
    if x.size < 100 or np.unique(x).size < 3:
        return float("nan")
    qs = np.quantile(x, np.linspace(0, 1, nbin + 1)[1:-1])
    b = np.searchsorted(np.unique(qs), x, side="right")
    best = float("nan")
    for k in np.unique(b):
        m = b == k
        if int(m.sum()) < 50:
            continue
        f = float(yy[m].mean()) / base
        if not np.isfinite(best) or f > best:
            best = f
    return best


# u1 scan

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
        idx, bad = 0, False
        for k, ch in enumerate(kmer):
            c = MAP[ord(ch)]
            if c < 0:
                bad = True
                break
            idx += int(c) * int(powv[k])
        if not bad:
            dense[idx] = -val
    _G.update(MAP=MAP, U1LEN=L, powv=powv, dense=dense)


def _u1_full(seq):
    MAP, L, powv, dense = _G["MAP"], _G["U1LEN"], _G["powv"], _G["dense"]
    arr = MAP[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]
    n = len(arr)
    if n < L:
        return np.empty(0)
    codes = np.zeros(n - L + 1, np.int64)
    bad = np.zeros(n - L + 1, bool)
    for k in range(L):
        c = arr[k:k + (n - L + 1)]
        codes += c * powv[k]
        bad |= (c < 0)
    return np.where(bad, np.nan, dense[np.clip(codes, 0, len(dense) - 1)])


def _reduce_u1(u, v, lo, hi, out, base):
    n_enc = len(U1_ENC)
    out[base:base + n_enc] = np.nan
    if u.size == 0 or hi < lo:
        return
    m = (u >= lo) & (u <= hi)
    kb = (hi - lo + 1) / 1000.0
    nt = len(U1_THR)
    k = base + len(BASE_RED)
    if not m.any():
        for ti in range(nt):
            out[k + ti] = 0.0
            out[k + nt + ti] = 0.0
            out[k + 2 * nt + ti] = np.nan
        return
    vv, uu = v[m], u[m]
    out[base] = vv.max()
    out[base + 1] = vv.mean()
    out[base + 2] = vv.sum()
    for ti, t in enumerate(U1_THR):
        s = vv >= t
        c = float(s.sum())
        out[k + ti] = c / kb
        out[k + nt + ti] = c
        out[k + 2 * nt + ti] = -float(np.abs(uu[s]).min()) if c > 0 else np.nan


def _work(rec):
    chrom, center, strand, A = rec
    A = int(A)
    u_max, u_min = A + U_ANCH_EXTRA, -U_DN
    out = np.full(len(U1_COLS), np.nan)
    if strand == "+":
        glo, ghi = center - u_max, center - u_min + 1
    else:
        glo, ghi = center + u_min, center + u_max + 1
    if glo < 0:
        return out
    try:
        seq = _G["fa"].fetch(chrom, glo, ghi, strand=strand)
    except Exception:
        return out
    if not seq or len(seq) < 30:
        return out
    v = _u1_full(seq.upper())
    if v.size == 0:
        return out
    u = u_max - np.arange(v.size, dtype=float)
    ok = np.isfinite(v)
    u, v = u[ok], v[ok]
    j = 0
    for _reg, fn in REGION_SPECS:
        lo, hi = fn(A)
        _reduce_u1(u, v, lo, hi, out, j)
        j += len(U1_ENC)
    return out


def u1_matrix(a, R):
    if os.path.isfile(a.u1_cache):
        z = np.load(a.u1_cache, allow_pickle=True)
        if z["M"].shape == (len(R), len(U1_COLS)) and list(z["gene"]) == [r["gene"] for r in R]:
            log("[u1] reusing %s" % a.u1_cache)
            return z["M"]
        log("[u1] %s stale (shape/gene mismatch), rescanning" % a.u1_cache)
    if not os.path.isfile(a.u1_lookup):
        sys.exit("missing required input: %s" % a.u1_lookup)
    recs = [(r["chrom"], int(float(r["center"])), r["strand"], float(r["d_anchor_to_pa"]))
            for r in R]
    log("[u1] scanning %d peaks x %d cells" % (len(recs), len(U1_COLS)))
    t0 = time.time()
    M = np.full((len(recs), len(U1_COLS)), np.nan, dtype=np.float32)
    cfg = {"fasta": a.fasta, "u1_lookup": a.u1_lookup}
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(cfg,)) as pool:
        for i, v in enumerate(pool.imap(_work, recs, chunksize=64)):
            M[i] = v
            if i and i % 20000 == 0:
                log("  %d/%d (%.1f min)" % (i, len(recs), (time.time() - t0) / 60))
    log("[u1] scan done in %.1f min" % ((time.time() - t0) / 60))
    np.savez_compressed(a.u1_cache, M=M,
                        cols=np.array(["|".join(c) for c in U1_COLS]),
                        gene=np.array([r["gene"] for r in R]))
    log("[u1] wrote %s" % a.u1_cache)
    return M


# pairing setup

def build_pairs(genes, y):
    order = {}
    by = {}
    for i, g in enumerate(genes):
        by.setdefault(g, ([], []))[0 if y[i] == 1 else 1].append(i)
    ci, ki, slot = [], [], []
    for g, (can, com) in by.items():
        if not can or not com:
            continue
        s = order.setdefault(g, len(order))
        for c in can:
            for k in com:
                ci.append(c)
                ki.append(k)
                slot.append(s)
    return (np.asarray(ci, np.int64), np.asarray(ki, np.int64),
            np.asarray(slot, np.int64), len(order))


def win_rate(v, ci, ki, slot, n_slot):
    c, k = v[ci], v[ki]
    ok = np.isfinite(c) & np.isfinite(k)
    if not ok.any():
        return np.empty(0), float("nan"), 0
    w = np.where(c > k, 1.0, np.where(c == k, 0.5, 0.0)) * ok
    num = np.bincount(slot, weights=w, minlength=n_slot)
    den = np.bincount(slot, weights=ok.astype(float), minlength=n_slot)
    m = den > 0
    return num[m] / den[m], float(num.sum() / den.sum()), int(den.sum())


# driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--cache", default="gridsearch_cache.npz")
    ap.add_argument("--u1_cache", default="u1_grid_cache.npz")
    ap.add_argument("--fasta", default="../train/data/hg38.fa")
    ap.add_argument("--u1_lookup", default="../train/splice_train/data/u1_lookup.npy")
    ap.add_argument("--out", default="master_grid.tsv")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    for p in (a.pos, a.cache):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)
    with open(a.pos, newline="") as fh:
        R = list(csv.DictReader(fh, delimiter="\t"))

    z = np.load(a.cache, allow_pickle=True)
    M = z["M"]
    cache_cols = [tuple(c.split("|")) for c in z["cols"]]
    genes = list(z["gene"])
    y = z["is_canonical"].astype(np.int8)
    if len(genes) != len(R) or [r["gene"] for r in R] != genes:
        sys.exit("row alignment mismatch: %s vs %s" % (a.pos, a.cache))

    U = u1_matrix(a, R)

    base = float((y == 1).mean())
    ci, ki, slot, n_slot = build_pairs(genes, y)
    log("[grid] %d peaks, %d genes, base rate %.4f" % (len(genes), len(set(genes)), base))
    log("[grid] %d comparable genes, %d canonical/competitor pairs" % (n_slot, ci.size))

    cells = []
    for j, (scorer, site, reg, enc) in enumerate(cache_cols):
        if enc.startswith("present"):
            continue
        red, _, thr = enc.partition("@")
        cells.append((scorer, site, reg, red, thr, M[:, j].astype(float)))
    for j, (reg, enc) in enumerate(U1_COLS):
        red, _, thr = enc.partition("@")
        cells.append(("U1", "donor", reg, red, thr, U[:, j].astype(float)))

    log("[grid] %d cells to evaluate" % len(cells))

    rows = []
    skipped_sparse = 0
    all_nan = 0
    t0 = time.time()
    for n, (scorer, site, reg, red, thr, v) in enumerate(cells):
        n_def = int(np.isfinite(v).sum())
        if n_def == 0:
            all_nan += 1
            continue
        rates, pw, npair = win_rate(v, ci, ki, slot, n_slot)
        if rates.size < MIN_GENES:
            skipped_sparse += 1
            continue
        raw = float(rates.mean())
        comp_hi = (site == "donor")
        exp_rates = (1.0 - rates) if comp_hi else rates
        expd = float(exp_rates.mean())
        pwd = (1.0 - pw) if comp_hi else pw
        p = wilcoxon_p(exp_rates - 0.5)
        au, _ = auroc(v, y)
        fe = fold_enrich(v, y, base)
        rows.append(dict(scorer=scorer, signal=site, region=reg, reduction=red,
                         threshold=(thr if thr else "-"), base_rate=base,
                         max_fold_enrich=fe, pooled_auroc=au,
                         within_gene_winrate_expected_dir=expd, winrate_raw=raw,
                         pair_wtd=pwd, wilcoxon_p=p, n_genes=int(rates.size),
                         n_peaks_defined=n_def, n_pairs=npair))
        if (n + 1) % 250 == 0:
            log("  %d/%d cells (%.1f s)" % (n + 1, len(cells), time.time() - t0))

    qs = bh_q([r["wilcoxon_p"] for r in rows])
    for r, q in zip(rows, qs):
        r["bh_q"] = q
        d = r["within_gene_winrate_expected_dir"]
        if abs(d - 0.5) <= NOISE_BAND:
            r["direction"] = "NULL"
        elif d > 0.5 and np.isfinite(q) and q < 0.01:
            r["direction"] = "CORRECT"
        elif d > 0.5:
            r["direction"] = "correct_ns"
        else:
            r["direction"] = "WRONG"

    KEYS = ["scorer", "signal", "region", "reduction", "threshold", "base_rate",
            "max_fold_enrich", "pooled_auroc", "within_gene_winrate_expected_dir",
            "winrate_raw", "pair_wtd", "wilcoxon_p", "bh_q", "direction",
            "n_genes", "n_peaks_defined"]
    with open(a.out, "w") as fh:
        fh.write("\t".join(KEYS) + "\n")
        for r in rows:
            fh.write("\t".join(("%.6g" % r[c]) if isinstance(r[c], float) else str(r[c])
                               for c in KEYS) + "\n")
    log("[grid] wrote %s (%d cells)" % (a.out, len(rows)))

    def fmt(r):
        return ("  %-9s %-8s %-12s %-15s %-6s %8.4f %8.4f %8.4f %9.3f %10s %8d"
                % (r["scorer"], r["signal"], r["region"], r["reduction"], r["threshold"],
                   r["within_gene_winrate_expected_dir"], r["pair_wtd"], r["pooled_auroc"],
                   r["max_fold_enrich"] if np.isfinite(r["max_fold_enrich"]) else float("nan"),
                   p_label(r["bh_q"]), r["n_genes"]))

    HDR = ("  %-9s %-8s %-12s %-15s %-6s %8s %8s %8s %9s %10s %8s"
           % ("scorer", "signal", "region", "reduction", "thr", "win_exp", "pair", "auroc",
              "fold", "q", "n_genes"))

    print("=" * 118)
    print("A. TOP 15 CELLS BY WITHIN-GENE WIN-RATE (expected direction), q < 0.01, win >= 0.55")
    print("   expected: acceptor = canonical higher (definition); donor/U1 = competitor higher (suppression)")
    print("=" * 118)
    print(HDR)
    top = [r for r in rows if r["within_gene_winrate_expected_dir"] >= 0.55
           and np.isfinite(r["bh_q"]) and r["bh_q"] < 0.01]
    top.sort(key=lambda r: -r["within_gene_winrate_expected_dir"])
    if not top:
        print("  none")
    for r in top[:15]:
        print(fmt(r))

    print()
    print("=" * 118)
    print("B. BEST CELL PER (SCORER x SIGNAL)")
    print("=" * 118)
    print(HDR)
    for scorer in ("MaxEnt", "U1", "SpliceAI"):
        for site in ("acceptor", "donor"):
            cand = [r for r in rows if r["scorer"] == scorer and r["signal"] == site]
            if not cand:
                print("  %-9s %-8s  (no cells; U1 has no acceptor analog)" % (scorer, site))
                continue
            print(fmt(max(cand, key=lambda r: r["within_gene_winrate_expected_dir"])))

    print()
    print("=" * 118)
    print("C. BEST CELL PER SIGNAL, ACROSS SCORERS   (where the scorers agree vs disagree)")
    print("=" * 118)
    for site in ("acceptor", "donor"):
        print("  %s:" % site.upper())
        print(HDR)
        for scorer in ("MaxEnt", "U1", "SpliceAI"):
            cand = [r for r in rows if r["scorer"] == scorer and r["signal"] == site]
            if cand:
                print(fmt(max(cand, key=lambda r: r["within_gene_winrate_expected_dir"])))
        print()

    print("=" * 118)
    print("D. NOISE CELLS   win-rate within %.2f of 0.5, regardless of p" % NOISE_BAND)
    print("=" * 118)
    noise = [r for r in rows if r["direction"] == "NULL"]
    print("  %d / %d cells (%.1f%%) are indistinguishable from a coin flip." %
          (len(noise), len(rows), 100.0 * len(noise) / max(1, len(rows))))
    from collections import Counter
    cnt = Counter((r["scorer"], r["signal"]) for r in noise)
    for (scorer, site), c in sorted(cnt.items()):
        tot = sum(1 for r in rows if r["scorer"] == scorer and r["signal"] == site)
        print("    %-9s %-8s %4d / %4d" % (scorer, site, c, tot))
    sig_noise = [r for r in noise if np.isfinite(r["bh_q"]) and r["bh_q"] < 0.01]
    print("  of those, %d reach q < 0.01 anyway (significant and meaningless)." % len(sig_noise))

    print()
    print("=" * 118)
    print("TOTALS")
    print("=" * 118)
    print("  peaks %d   genes %d   comparable genes %d   pairs %d   base rate %.4f"
          % (len(genes), len(set(genes)), n_slot, ci.size, base))
    print("  cells evaluated %d   reported %d" % (len(cells), len(rows)))
    print("  dropped, all-NaN column          : %d" % all_nan)
    print("  dropped, sparse (n_genes < %d)  : %d" % (MIN_GENES, skipped_sparse))
    dc = Counter(r["direction"] for r in rows)
    for k in ("CORRECT", "correct_ns", "NULL", "WRONG"):
        print("  direction %-11s: %d" % (k, dc.get(k, 0)))
    zero = sum(1 for r in rows if r["n_peaks_defined"] == 0)
    print("  0-drop check: cells with n_peaks_defined == 0 reported: %d" % zero)


if __name__ == "__main__":
    main()
