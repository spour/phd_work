#!/usr/bin/env python
# is u1 hybridization a within-gene telescripting scorer?

import os
import sys
import csv
import math
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

U_DN = 0
U_EXTRA = 2000
U1_THR = (0.0, 4.0, 8.0)
MAXENT_TARGET, SPLICEAI_FAIL = 0.554, 0.447
AGREE_MIN = 0.55
REGION_SPECS = [
    ("interv", lambda A: (1, max(1, A - 1))),
    ("pas500", lambda A: (1, 500)),
    ("pas250", lambda A: (1, 250)),
    ("pas100", lambda A: (1, 100)),
]
ENC = ["max", "mean", "sum"] + ["density@%g" % t for t in U1_THR]
COLS = [(r, e) for r, _ in REGION_SPECS for e in ENC]


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


_G = {}


def _init(cfg):
    from extract_splicing_features import GenomeFasta, init_maxentscan, load_u1_lookup
    o = init_maxentscan()
    _G["s5"] = o[0]
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


def _work(rec):
    chrom, center, strand, A, want_sign = rec
    A = int(A)
    u_max = A + U_EXTRA
    out = np.full(len(COLS), np.nan)
    if strand == "+":
        glo, ghi = center - u_max, center + 1
    else:
        glo, ghi = center, center + u_max + 1
    if glo < 0:
        return out, None
    try:
        seq = _G["fa"].fetch(chrom, glo, ghi, strand=strand)
    except Exception:
        return out, None
    if not seq or len(seq) < 30:
        return out, None
    seq = seq.upper()
    u1v = _u1_full(seq)
    if u1v.size == 0:
        return out, None
    u = u_max - np.arange(u1v.size, dtype=float)
    ok = np.isfinite(u1v)
    u, v = u[ok], u1v[ok]

    sign = None
    if want_sign:
        arr = np.frombuffer(seq.encode("ascii", "replace"), np.uint8)
        s5 = _G["s5"]
        st, bg = [], []
        gts = np.where((arr[:-1] == 71) & (arr[1:] == 84))[0]
        for i in gts[:400]:
            p = int(i)
            if p - 3 < 0 or p + 6 > len(seq):
                continue
            try:
                x = s5(seq[p - 3:p + 6])
                x = float(np.nanmax(np.asarray(x, float))) if not np.isscalar(x) else float(x)
            except Exception:
                continue
            k = p - 3
            if 0 <= k < u1v.size and np.isfinite(u1v[k]):
                (st if x >= 8.0 else bg).append(float(u1v[k]))
        sign = (st, bg)

    j = 0
    for _reg, fn in REGION_SPECS:
        lo, hi = fn(A)
        m = (u >= lo) & (u <= hi)
        if m.any():
            vv = v[m]
            kb = (hi - lo + 1) / 1000.0
            out[j] = vv.max()
            out[j + 1] = vv.mean()
            out[j + 2] = vv.sum()
            for ti, t in enumerate(U1_THR):
                out[j + 3 + ti] = float((vv >= t).sum()) / kb
        j += len(ENC)
    return out, sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--fasta", default="../train/data/hg38.fa")
    ap.add_argument("--u1_lookup", default="../train/splice_train/data/u1_lookup.npy")
    ap.add_argument("--cache", default="gridsearch_cache.npz")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    for p in (a.pos, a.fasta, a.u1_lookup):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)
    with open(a.pos, newline="") as fh:
        R = list(csv.DictReader(fh, delimiter="\t"))
    genes = np.array([r["gene"] for r in R])
    y = np.array([int(float(r["is_canonical"])) for r in R], dtype=np.int8)
    recs = [(r["chrom"], int(float(r["center"])), r["strand"], float(r["d_anchor_to_pa"]),
             i < 300) for i, r in enumerate(R)]
    log("peaks %d, %d U1 cells" % (len(recs), len(COLS)))

    t0 = time.time()
    M = np.full((len(recs), len(COLS)), np.nan, dtype=np.float32)
    strong, back = [], []
    cfg = {"fasta": a.fasta, "u1_lookup": a.u1_lookup}
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(cfg,)) as pool:
        for i, (v, sg) in enumerate(pool.imap(_work, recs, chunksize=64)):
            M[i] = v
            if sg:
                strong.extend(sg[0])
                back.extend(sg[1])
            if i and i % 20000 == 0:
                log("  %d/%d (%.1f min)" % (i, len(recs), (time.time() - t0) / 60))
    log("scan done in %.1f min" % ((time.time() - t0) / 60))

    print("=" * 104)
    print("SIGN CHECK  (must hold, or every result below inverts)")
    print("=" * 104)
    s, b = np.asarray(strong), np.asarray(back)
    print("  U1 at MaxEnt-strong donors (>=8): mean %.4f  n %d" % (s.mean(), s.size))
    print("  U1 at other GT positions        : mean %.4f  n %d" % (b.mean(), b.size))
    okname = "CORRECT (higher = stronger U1)" if s.mean() > b.mean() else "INVERTED"
    print("  delta %+.4f  ->  %s" % (s.mean() - b.mean(), okname))
    print("  lookup negated at load (dense[idx] = -val), matching figure3_data.py")
    if s.mean() <= b.mean():
        print("\n  SIGN IS INVERTED. Stopping.")
        return
    print()

    def win(v):
        can, com = defaultdict(list), defaultdict(list)
        for i in range(v.size):
            if np.isfinite(v[i]):
                (can if y[i] == 1 else com)[genes[i]].append(v[i])
        rates, tw, tp = [], 0.0, 0
        for g, cs in can.items():
            ks = com.get(g)
            if not ks:
                continue
            c = np.asarray(cs)[:, None]
            k = np.asarray(ks)[None, :]
            w = float((k > c).sum()) + 0.5 * float((c == k).sum())
            n = c.size * k.size
            rates.append(w / n)
            tw += w
            tp += n
        r = np.asarray(rates)
        return r, (tw / tp if tp else float("nan"))

    print("=" * 104)
    print("U1 WITHIN-GENE WIN-RATE   P(competitor U1 > canonical U1 | same gene)")
    print("  anchors:  MaxEnt intervening donor %.3f (correct)   SpliceAI donor %.3f (wrong)"
          % (MAXENT_TARGET, SPLICEAI_FAIL))
    print("=" * 104)
    print("  %-9s %-13s %9s %9s %11s  %s" % ("region", "encoding", "gene-avg", "pair-wtd",
                                             "Wilcoxon p", "direction"))
    res = {}
    for j, (reg, enc) in enumerate(COLS):
        r, pw = win(M[:, j].astype(float))
        if r.size < 100:
            continue
        ga = float(r.mean())
        p = wilcoxon_p(r - 0.5)
        res[(reg, enc)] = (ga, pw, p)
        print("  %-9s %-13s %9.4f %9.4f %11s  %s"
              % (reg, enc, ga, pw, p_label(p),
                 "CORRECT (comp higher)" if ga > 0.5 else "wrong (canon higher)"))
    print()

    if os.path.isfile(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        cols = {tuple(c.split("|")): i for i, c in enumerate(z["cols"])}
        me_int = z["M"][:, cols[("MaxEnt", "donor", "interv", "max")]].astype(float)
        print("=" * 104)
        print("IS PAS-PROXIMAL U1 JUST THE DONOR IN DISGUISE?")
        print("=" * 104)
        for reg in ("pas500", "pas250", "pas100", "interv"):
            j = COLS.index((reg, "max"))
            v = M[:, j].astype(float)
            m = np.isfinite(v) & np.isfinite(me_int)
            print("  corr(%s U1 max, MaxEnt intervening donor) = %+.4f  (n %d)"
                  % (reg, np.corrcoef(v[m], me_int[m])[0, 1], int(m.sum())))
        print()

    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    pas = [(k, v) for k, v in res.items() if k[0].startswith("pas")]
    best_pas = max(pas, key=lambda kv: kv[1][0]) if pas else None
    corr_ok = True
    if best_pas:
        (reg, enc), (ga, pw, p) = best_pas
        print("  best PAS-proximal cell: %s/%s  win %.4f  %s" % (reg, enc, ga, p_label(p)))
        if ga >= AGREE_MIN and p < 1e-5:
            print("  -> PAS-proximal U1 IS competitor-higher at >= %.2f. Mechanism-distinct"
                  % AGREE_MIN)
            print("     telescripting signal, agreeing with MaxEnt's donor direction.")
        elif ga > 0.5:
            print("  -> competitor-higher but below the %.2f bar; suggestive, not sufficient."
                  % AGREE_MIN)
        else:
            print("  -> WRONG direction: canonical carries more PAS-proximal U1.")
            print("     U1 does not supply a telescripting signal here.")
    n_corr = sum(1 for k, v in res.items() if v[0] > 0.5)
    print("  cells competitor-higher: %d / %d" % (n_corr, len(res)))


if __name__ == "__main__":
    main()
