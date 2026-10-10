#!/usr/bin/env python3
# can the spl_* featuriser be re-run on arbitrary sequence?
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, "scripts")

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=200)
P.add_argument("--seed", type=int, default=0)
A = P.parse_args()

import u1_5ss_landscape_data as L
from fastparquet import ParquetFile
from extract_splicing_features import init_maxentscan, load_me5_lookup, load_u1_lookup
from compute_gene_baselines import derive_axes
import splice_extra_features as XF
from splice_common import compute_splice_row, SPLICE_COLS

UP, DN, NB = 2000, 500, 8
ME5P = "../train/splice_train/data/me5_lookup.npy"
U1P  = "../train/splice_train/data/u1_lookup.npy"
_, score3, _, m3 = init_maxentscan()
me5 = load_me5_lookup(ME5P); u1 = load_u1_lookup(U1P)
log(f"lookups: me5 {len(me5)} 9-mers, u1 {len(u1)} 9-mers")

T = ParquetFile("shortlist_feat_all.parquet").to_pandas()
COLS = [c for c in SPLICE_COLS if c in T.columns]
_missing = [c for c in SPLICE_COLS if c not in T.columns]
if _missing: log(f"[note] not stored in the parquet, excluded from the check: {_missing}")
G = L.G_["G"]

def fetch(chrom, c, st, up=UP, dn=DN):
    a, b = (c - up, c + dn) if st == "+" else (c - dn, c + up)
    s = G.fetch(chrom, a, b, st)
    return s if s and len(s) == up + dn else None

def row_for(seq):
    return compute_splice_row(seq, UP, DN, NB, me5, score3, m3, u1, derive_axes, XF)

if __name__ == "__main__":
    rng = np.random.default_rng(A.seed)
    idx = rng.choice(len(T), A.n, replace=False)
    got = {c: [] for c in COLS}; want = {c: [] for c in COLS}
    t0 = time.time(); nfetch = 0
    for i in idx:
        r = T.iloc[i]
        seq = fetch(r.chrom, int(r.center), r.strand)
        if seq is None: continue
        nfetch += 1
        rw = row_for(seq)
        for c in COLS:
            got[c].append(rw[c]); want[c].append(float(r[c]))
    log(f"re-derived {nfetch}/{A.n} sites in {time.time()-t0:.0f}s "
        f"({(time.time()-t0)/max(nfetch,1)*1000:.0f} ms/site)")
    log(f"{'column':28s} {'exact':>7s} {'both-NaN':>9s} {'r':>7s}")
    nfull = 0
    rows = []
    for c in COLS:
        g = np.array(got[c], float); w = np.array(want[c], float)
        bn = (~np.isfinite(g)) & (~np.isfinite(w))
        fin = np.isfinite(g) & np.isfinite(w)
        ex = int((np.abs(g[fin] - w[fin]) <= 1e-5 * np.maximum(1, np.abs(w[fin]))).sum()) + int(bn.sum())
        rr = float(np.corrcoef(g[fin], w[fin])[0, 1]) if fin.sum() > 2 and np.std(g[fin]) > 0 and np.std(w[fin]) > 0 else np.nan
        frac = ex / len(g)
        if frac >= 0.99: nfull += 1
        log(f"{c:28s} {frac:7.3f} {int(bn.sum()):9d} {rr:7.3f}")
        rows.append((c, frac, int(bn.sum()), rr))
    log(f"\n[verdict] {nfull}/{len(COLS)} columns reproduce at >=99% exact agreement "
        f"over {nfetch} sites")
    with open("splice_refeaturise_check.tsv", "w") as fh:
        fh.write("column\tfrac_exact\tn_both_nan\tpearson_r\tn_sites\n")
        for c, f_, b_, r_ in rows: fh.write(f"{c}\t{f_:.5f}\t{b_}\t{r_:.5f}\t{nfetch}\n")
    log("[out] splice_refeaturise_check.tsv")
