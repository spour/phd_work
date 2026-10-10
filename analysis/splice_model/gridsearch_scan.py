#!/usr/bin/env python

import os
import sys
import csv
import math
import time
import argparse
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

U_DN = 1000
U_ANCH_EXTRA = 2000
ME_THR = (4.0, 6.0, 8.0, 10.0)
SA_THR = (0.2, 0.5, 0.8)
BASE_RED = ("max", "mean", "sum")
THR_RED = ("density_per_kb", "count_above", "present", "nearest_dist")
BW = {("acceptor", "+"): "spliceAiAcceptorPlus.bw", ("acceptor", "-"): "spliceAiAcceptorMinus.bw",
      ("donor", "+"): "spliceAiDonorPlus.bw", ("donor", "-"): "spliceAiDonorMinus.bw"}

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
REGIONS = [r[0] for r in REGION_SPECS]


def enc_names(scorer):
    thrs = ME_THR if scorer == "MaxEnt" else SA_THR
    out = list(BASE_RED)
    for red in THR_RED:
        for t in thrs:
            out.append("%s@%g" % (red, t))
    return out


def column_index():
    cols = []
    for scorer in ("MaxEnt", "SpliceAI"):
        for site in ("acceptor", "donor"):
            for reg in REGIONS:
                for enc in enc_names(scorer):
                    cols.append((scorer, site, reg, enc))
    return cols


COLS = column_index()
COL_OF = {c: i for i, c in enumerate(COLS)}


def reduce_region(u, v, lo, hi, thrs, out, base):
    n_enc = len(BASE_RED) + len(THR_RED) * len(thrs)
    out[base:base + n_enc] = np.nan
    if u.size == 0 or hi < lo:
        return n_enc
    m = (u >= lo) & (u <= hi)
    if not m.any():
        kb = (hi - lo + 1) / 1000.0
        k = base + len(BASE_RED)
        for _t in thrs:
            out[k] = 0.0
            out[k + len(thrs)] = 0.0
            out[k + 2 * len(thrs)] = 0.0
            out[k + 3 * len(thrs)] = np.nan
            k += 1
        return n_enc
    vv = v[m]
    uu = u[m]
    out[base] = vv.max()
    out[base + 1] = vv.mean()
    out[base + 2] = vv.sum()
    kb = (hi - lo + 1) / 1000.0
    k = base + len(BASE_RED)
    nt = len(thrs)
    for ti, t in enumerate(thrs):
        s = vv >= t
        c = float(s.sum())
        out[k + ti] = c / kb
        out[k + nt + ti] = c
        out[k + 2 * nt + ti] = 1.0 if c > 0 else 0.0
        out[k + 3 * nt + ti] = -float(np.abs(uu[s]).min()) if c > 0 else np.nan
    return n_enc


_G = {}


def _init(cfg):
    from extract_splicing_features import init_maxentscan, GenomeFasta
    import pyBigWig
    o = init_maxentscan()
    _G["s5"], _G["s3"] = o[0], o[1]
    _G["fa"] = GenomeFasta(cfg["fasta"])
    _G["bw"] = {k: pyBigWig.open(os.path.join(cfg["bw_dir"], v)) for k, v in BW.items()}
    _G["chroms"] = {k: _G["bw"][k].chroms() for k in _G["bw"]}


def _work(rec):
    import numpy as np
    chrom, center, strand, A = rec
    A = int(A)
    u_max = A + U_ANCH_EXTRA
    u_min = -U_DN
    out = np.full(len(COLS), np.nan, dtype=np.float64)
    # genomic span
    if strand == "+":
        glo, ghi = center - u_max, center - u_min + 1
    else:
        glo, ghi = center + u_min, center + u_max + 1
    if glo < 0:
        return out
    try:
        seq = _G["fa"].fetch(chrom, glo, ghi, strand=strand)
    except Exception:
        seq = None
    if not seq or len(seq) < 60:
        return out
    seq = seq.upper()
    L = len(seq)
    arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)

    # maxent acceptor (ag) and donor (gt), per position, no upstream-only filter
    s3, s5 = _G["s3"], _G["s5"]
    au, av = [], []
    for i in np.where((arr[:-1] == 65) & (arr[1:] == 71))[0]:
        p = int(i) + 2
        if p - 20 < 0 or p + 3 > L:
            continue
        try:
            x = s3(seq[p - 20:p + 3])
            x = float(np.nanmax(np.asarray(x, float))) if not np.isscalar(x) else float(x)
        except Exception:
            continue
        if np.isfinite(x) and x > -99:
            au.append(u_max - p)
            av.append(x)
    du, dv = [], []
    for i in np.where((arr[:-1] == 71) & (arr[1:] == 84))[0]:
        p = int(i)
        if p - 3 < 0 or p + 6 > L:
            continue
        try:
            x = s5(seq[p - 3:p + 6])
            x = float(np.nanmax(np.asarray(x, float))) if not np.isscalar(x) else float(x)
        except Exception:
            continue
        if np.isfinite(x) and x > -99:
            du.append(u_max - p)
            dv.append(x)
    me = {"acceptor": (np.asarray(au, float), np.asarray(av, float)),
          "donor": (np.asarray(du, float), np.asarray(dv, float))}

    # spliceai, same span
    sa = {}
    u_grid = np.arange(u_min, u_max + 1, dtype=float)
    for site in ("acceptor", "donor"):
        bw = _G["bw"][(site, strand)]
        clen = _G["chroms"][(site, strand)].get(chrom)
        vals = np.full(u_grid.size, np.nan)
        if clen is not None:
            lo_c, hi_c = max(0, glo), min(ghi, clen)
            if hi_c > lo_c:
                try:
                    raw = np.asarray(bw.values(chrom, lo_c, hi_c), dtype=np.float32)
                except Exception:
                    raw = None
                if raw is not None:
                    if strand == "+":
                        u_of = center - np.arange(lo_c, hi_c)
                    else:
                        u_of = np.arange(lo_c, hi_c) - center
                    keep = (u_of >= u_min) & (u_of <= u_max) & np.isfinite(raw)
                    idx = (u_of[keep] - u_min).astype(np.int64)
                    vals[idx] = raw[keep]
        ok = np.isfinite(vals)
        sa[site] = (u_grid[ok], vals[ok])

    # reduce every cell
    for scorer, data, thrs in (("MaxEnt", me, ME_THR), ("SpliceAI", sa, SA_THR)):
        for site in ("acceptor", "donor"):
            u, v = data[site]
            for reg, fn in REGION_SPECS:
                lo, hi = fn(A)
                base = COL_OF[(scorer, site, reg, enc_names(scorer)[0])]
                reduce_region(u, v, lo, hi, thrs, out, base)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--fasta", default="../train/data/hg38.fa")
    ap.add_argument("--bw-dir", default="spliceai_bw")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="gridsearch_cache.npz")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    for p in (a.pos, a.fasta):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)
    with open(a.pos, newline="") as fh:
        R = list(csv.DictReader(fh, delimiter="\t"))
    if a.limit:
        R = R[:a.limit]
    recs = [(r["chrom"], int(float(r["center"])), r["strand"], float(r["d_anchor_to_pa"]))
            for r in R]
    print("[scan] %d peaks, %d grid columns" % (len(recs), len(COLS)), file=sys.stderr)
    print("[scan] regions %d, MaxEnt enc %d, SpliceAI enc %d"
          % (len(REGIONS), len(enc_names("MaxEnt")), len(enc_names("SpliceAI"))), file=sys.stderr)

    t0 = time.time()
    cfg = {"fasta": a.fasta, "bw_dir": a.bw_dir}
    M = np.full((len(recs), len(COLS)), np.nan, dtype=np.float32)
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(cfg,)) as pool:
            for i, v in enumerate(pool.imap(_work, recs, chunksize=64)):
                M[i] = v
                if i and i % 10000 == 0:
                    print("  %d/%d  (%.1f min)" % (i, len(recs), (time.time() - t0) / 60),
                          file=sys.stderr)
    else:
        _init(cfg)
        for i, r in enumerate(recs):
            M[i] = _work(r)
    print("[scan] done in %.1f min" % ((time.time() - t0) / 60), file=sys.stderr)

    np.savez_compressed(
        a.out, M=M,
        cols=np.array(["|".join(c) for c in COLS]),
        gene=np.array([r["gene"] for r in R]),
        is_canonical=np.array([int(float(r["is_canonical"])) for r in R], dtype=np.int8))
    print("[scan] wrote %s" % a.out, file=sys.stderr)

    # checks
    y = np.array([int(float(r["is_canonical"])) for r in R])
    print("\nCHECKS", file=sys.stderr)
    print("  rows %d (%d canonical / %d competitor)" % (len(R), int((y == 1).sum()),
                                                        int((y == 0).sum())), file=sys.stderr)
    allnan = int(np.isnan(M).all(axis=1).sum())
    print("  peaks with NO data at all (dropped): %d" % allnan, file=sys.stderr)
    for scorer in ("MaxEnt", "SpliceAI"):
        for site in ("acceptor", "donor"):
            idx = [i for i, c in enumerate(COLS) if c[0] == scorer and c[1] == site]
            cov = float(np.isfinite(M[:, idx]).any(axis=1).mean())
            print("  %-9s %-9s: peaks with >=1 finite value  %.4f" % (scorer, site, cov),
                  file=sys.stderr)
    ai = [i for i, c in enumerate(COLS) if c[0] == "MaxEnt" and c[1] == "acceptor"]
    regs_pop = sorted({COLS[i][2] for i in ai if np.isfinite(M[:, i]).mean() > 0.5})
    print("  MaxEnt ACCEPTOR regional data populated in %d/%d regions: %s"
          % (len(regs_pop), len(REGIONS), ", ".join(regs_pop)), file=sys.stderr)


if __name__ == "__main__":
    main()
