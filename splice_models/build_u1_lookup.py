#!/usr/bin/env python3
# precompute u1 snrna hybridization energies for all 9-mers
import argparse
import itertools as it
import os
import time
import numpy as np


U1 = "ACUUACCUG"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="u1_lookup.npy",
                     help="Output .npy path. Saved as numpy 0-d object array "
                          "containing a dict; load with np.load(p, "
                          "allow_pickle=True).item().")
    ap.add_argument("--u1", default=U1,
                     help=f"U1 snRNA hybridization region (5'->3'). "
                          f"Default: {U1}")
    ap.add_argument("--k", type=int, default=9,
                     help="K-mer length to enumerate. Default 9.")
    args = ap.parse_args()

    try:
        import RNA
    except ImportError:
        raise SystemExit(
            "ViennaRNA Python bindings not available. On Compute Canada:\n"
            "    module load StdEnv/2023 viennarna/2.6.4\n"
            "Or:    pip install ViennaRNA"
        )

    print(f"[u1_lookup] U1 sequence: {args.u1} (5'->3')")
    print(f"[u1_lookup] enumerating {4 ** args.k:,} {args.k}-mers")

    dna_bases = "ACTG"
    rna_bases = "ACUG"

    all_dna = ["".join(p) for p in it.product(dna_bases, repeat=args.k)]
    all_rna = ["".join(p) for p in it.product(rna_bases, repeat=args.k)]

    t0 = time.time()
    energies = np.empty(len(all_rna), dtype=np.float32)
    for i, rna in enumerate(all_rna):
        energies[i] = RNA.duplexfold(args.u1, rna).energy
        if (i + 1) % 50_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(all_rna) - i - 1) / rate
            print(f"[u1_lookup]   {i + 1:,}/{len(all_rna):,}  "
                   f"rate={rate:,.0f}/s  eta={eta/60:.1f}m")

    lookup = dict(zip(all_dna, energies.tolist()))

    print(f"[u1_lookup] done in {(time.time() - t0)/60:.1f} min")
    print(f"[u1_lookup] energy range: "
           f"min={min(lookup.values()):.2f}  max={max(lookup.values()):.2f}  "
           f"median={float(np.median(list(lookup.values()))):.2f}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, lookup, allow_pickle=True)
    print(f"[u1_lookup] wrote {args.out}")


if __name__ == "__main__":
    main()