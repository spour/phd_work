#!/usr/bin/env python3
# precompute maxentscan 5'ss scores for all 9-mers
import argparse
import itertools as it
import os
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="me5_lookup.npy",
                     help="Output .npy path. Saved as numpy 0-d object array "
                          "containing a dict; load with np.load(p, "
                          "allow_pickle=True).item().")
    args = ap.parse_args()

    try:
        from maxentpy import maxent
        from maxentpy.maxent import load_matrix5
    except ImportError:
        raise SystemExit(
            "maxentpy is required. Install via install_maxentpy_noc.sh."
        )

    m5 = load_matrix5()
    print(f"[me5_lookup] enumerating {4 ** 9:,} 9-mers")

    bases = "ACGT"
    all_kmers = ["".join(p) for p in it.product(bases, repeat=9)]

    t0 = time.time()
    scores = np.empty(len(all_kmers), dtype=np.float32)
    for i, kmer in enumerate(all_kmers):
        try:
            scores[i] = maxent.score5(kmer, matrix=m5)
        except Exception:
            scores[i] = -99.0
        if (i + 1) % 50_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(all_kmers) - i - 1) / rate
            print(f"[me5_lookup]   {i + 1:,}/{len(all_kmers):,}  "
                   f"rate={rate:,.0f}/s  eta={eta:.0f}s")

    lookup = dict(zip(all_kmers, scores.tolist()))
    print(f"[me5_lookup] done in {time.time() - t0:.1f}s")
    print(f"[me5_lookup] score range: "
           f"min={min(lookup.values()):.2f}  max={max(lookup.values()):.2f}  "
           f"median={float(np.median(list(lookup.values()))):.2f}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, lookup, allow_pickle=True)
    print(f"[me5_lookup] wrote {args.out}")


if __name__ == "__main__":
    main()