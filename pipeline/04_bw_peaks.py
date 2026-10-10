#!/usr/bin/env python3
# call peaks from a stranded bigwig pair and write a table that liftover_table.sh can

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import pyBigWig
except ImportError:
    sys.exit("pyBigWig needed:  pip install pyBigWig")


def _self_id():
    import hashlib
    try:
        h = hashlib.md5(open(__file__, "rb").read()).hexdigest()[:8]
        print(f"[{os.path.basename(__file__)}  build {h}]", file=sys.stderr)
    except OSError:
        pass


def describe(path):
    bw = pyBigWig.open(path)
    h = bw.header()
    chroms = bw.chroms()
    chr1 = chroms.get("chr1") or chroms.get("1")
    build = {248956422: "hg38", 249250621: "hg19"}.get(chr1, f"unknown ({chr1})")
    cov = h.get("nBasesCovered", 0)
    tot = h.get("sumData", 0.0)
    print(f"{os.path.basename(path)}")
    print(f"  build            {build}")
    print(f"  bases covered    {cov:,}")
    print(f"  total signal     {tot:,.0f}")
    print(f"  mean per base    {tot / max(cov, 1):.3f}")
    print(f"  max              {h.get('maxVal', float('nan')):.1f}")
    kind = "SPARSE site track" if cov < 5_000_000 else "COVERAGE track"
    print(f"  looks like       {kind}")
    if kind.startswith("COVERAGE"):
        print("  Peak calling on a coverage track will not give you 3' ends.")
        bw.close()
        return kind

    vals = []
    for c in list(chroms)[:6]:
        try:
            ivs = bw.intervals(c)
        except RuntimeError:
            continue
        if ivs:
            vals.extend(abs(v) for _, _, v in ivs)
        if len(vals) > 400_000:
            break
    bw.close()
    if not vals:
        return kind
    v = np.asarray(vals)
    qs = [50, 75, 90, 95, 99, 99.9]
    print("  interval value quantiles (sampled from the first chromosomes)")
    print("    " + "  ".join(f"{q}%={np.percentile(v, q):.3g}" for q in qs))
    print("  peaks retained at each threshold, extrapolated genome-wide")
    for thr in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        frac = float((v >= thr).mean())
        print(f"    >= {thr:<6g} keeps {100 * frac:6.2f}% of intervals")
    print("  Pick a threshold from this, or use --top-n to fix the count.")
    return kind


def call_peaks(path, strand, min_value, merge_within):
    bw = pyBigWig.open(path)
    rows = []
    for chrom in bw.chroms():
        try:
            ivs = bw.intervals(chrom)
        except RuntimeError:
            continue
        if not ivs:
            continue
        keep = [(s, e, abs(v)) for s, e, v in ivs if abs(v) >= min_value]
        if not keep:
            continue
        b_end, best_v, best_p = keep[0][1], keep[0][2], keep[0][0]
        for s, e, v in keep[1:]:
            if s - b_end <= merge_within:
                b_end = max(b_end, e)
                if v > best_v:
                    best_v, best_p = v, s
            else:
                rows.append((chrom, best_p + 1, strand, best_v))
                b_end, best_v, best_p = e, v, s
        rows.append((chrom, best_p + 1, strand, best_v))
    bw.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", nargs="+", help="one or more bigWigs")
    ap.add_argument("--report", nargs="*", metavar="BW",
                    help="describe the files and exit")
    ap.add_argument("--min-value", type=float, default=5.0)
    ap.add_argument("--merge-within", type=int, default=25)
    ap.add_argument("--strand", default=None, choices=["+", "-"])
    ap.add_argument("--top-n", type=int, default=None,
                    help="keep the N strongest peaks, overriding --min-value")
    ap.add_argument("--target", type=int, default=None,
                    help="published peak count, printed alongside for comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    _self_id()

    if args.report is not None:
        for p in (args.report or args.bw or []):
            describe(p)
            print()
        return
    if not args.bw or not args.out:
        sys.exit("need --bw and --out (or --report)")

    min_value = args.min_value
    if args.top_n:
        if min_value != 5.0:
            print(f"--top-n given, ignoring --min-value {min_value}")
        min_value = 0.0

    all_rows = []
    for path in args.bw:
        kind = describe(path)
        if kind.startswith("COVERAGE"):
            print("  refusing to call peaks on this file\n")
            continue
        rev = path.replace(".fwd.bw", ".rev.bw")
        if path.endswith(".fwd.bw") and os.path.exists(rev):
            all_rows += call_peaks(path, "+", min_value, args.merge_within)
            all_rows += call_peaks(rev, "-", min_value, args.merge_within)
        elif args.strand:
            all_rows += call_peaks(path, args.strand, min_value, args.merge_within)
        else:
            r = call_peaks(path, "+", min_value, args.merge_within)
            all_rows += r + [(c, p, "-", v) for c, p, _, v in r]
        print()

    if not all_rows:
        sys.exit("no peaks called")
    df = pd.DataFrame(all_rows, columns=["seqnames", "start", "strand", "value"])
    n_unique = df.drop_duplicates(["seqnames", "start"]).shape[0]
    if n_unique * 2 == len(df):
        print("NOTE: this bigWig is unstranded, so every site was emitted on")
        print("      both strands. The strand column is not real and matching")
        print("      against it is effectively unstranded.")
    if args.top_n:
        top = (df.drop_duplicates(["seqnames", "start"])
                 .nlargest(args.top_n, "value")[["seqnames", "start"]])
        df = df.merge(top, on=["seqnames", "start"])
    df = df.sort_values(["seqnames", "start"]).reset_index(drop=True)

    v = df["value"].to_numpy()
    how = f"top {args.top_n:,}" if args.top_n else f"min-value {min_value}"
    print(f"called {len(df):,} rows / "
          f"{df.drop_duplicates(['seqnames', 'start']).shape[0]:,} unique "
          f"positions ({how})")
    print(f"  value quantiles  "
          f"50% {np.percentile(v, 50):.1f}  "
          f"90% {np.percentile(v, 90):.1f}  "
          f"99% {np.percentile(v, 99):.1f}")
    if args.target:
        n_u = df.drop_duplicates(["seqnames", "start"]).shape[0]
        print(f"  reference count  {args.target:,}   ratio {n_u / args.target:.2f}x")
        print("  Compare like with like: a polyA_sites track holds ALL 3' ends,")
        print("  so expect tens of thousands. Published intronic PCPA counts")
        print("  (So 2019: 1,485) match the INTRONIC SUBSET after intersection,")
        print("  not the raw call.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"\nwrote {args.out}")
    print("If this bigWig is hg19, lift it before overlapping:")
    print(f"  external/liftover_table.sh {args.out} seqnames start strand "
          f"{args.out.replace('.tsv', '_hg38.tsv')}")


if __name__ == "__main__":
    main()