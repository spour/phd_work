#!/usr/bin/env python3
# confirm part b extracts splicing features on the same window as training
import argparse
import os
import sys
import numpy as np
import pandas as pd

PROX_NT, PAD, U1_TAU = 1000, 12, 200


def burden_prox_u1(fa, u1_lookup, scan_u1, chrom, center, strand):
    if strand == "+":
        gs, ge = center - PROX_NT, center + PAD
    else:
        gs, ge = center - PAD, center + PROX_NT
    seq = fa.fetch(chrom, gs, ge, strand=strand)
    if seq is None or len(seq) < PROX_NT:
        return np.nan
    u1 = scan_u1(seq, u1_lookup)
    if u1.size == 0:
        return np.nan
    p = np.arange(u1.size); up = p < PROX_NT
    d = (PROX_NT - p[up]).astype(float); vals = u1[up]
    fin = np.isfinite(vals)
    if not fin.any():
        return np.nan
    d, vals = d[fin], vals[fin]
    bg = np.percentile(vals, 10)
    return float(np.sum(np.maximum(vals - bg, 0.0) * np.exp(-d / float(U1_TAU))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--rf_dir", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--genes", nargs="+", default=["PUDP"])
    ap.add_argument("--upstream_nt", type=int, default=3000)
    ap.add_argument("--downstream_nt", type=int, default=1000)
    ap.add_argument("--scripts_dir", default="scripts")
    args = ap.parse_args()

    sys.path.insert(0, args.scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup,
                                           init_maxentscan, extract_one_site, scan_u1)
    fa = GenomeFasta(args.reference_fasta)
    u1 = load_u1_lookup(args.u1_lookup); me5 = load_me5_lookup(args.me5_lookup)
    _, score3, _, m3 = init_maxentscan()

    df = pd.read_csv(args.features_tsv, sep="\t")
    print(f"{'gene':<12s}{'y':>2s}{'feature':<22s}{'stored':>12s}{'recomputed':>12s}{'absdiff':>10s}", file=sys.stderr)
    for gene in args.genes:
        sub = df[df["gene"] == gene]
        if sub.empty:
            print(f"{gene}: not in features_tsv", file=sys.stderr); continue
        for _, r in sub.iterrows():
            c = int(r["center"]); strand = r["strand"]; chrom = r["chrom"]
            if strand == "+":
                ws, we = c - args.upstream_nt, c + args.downstream_nt
            else:
                ws, we = c - args.downstream_nt, c + args.upstream_nt
            seq = fa.fetch(chrom, ws, we, strand=strand)
            feats = extract_one_site(seq, args.upstream_nt, args.downstream_nt, 8, me5, score3, m3, u1)
            me3_max = float(np.nanmax([feats.get(f"bin{b}_me3_max", np.nan) for b in range(1, 7)]))
            burden = burden_prox_u1(fa, u1, scan_u1, chrom, c, strand)
            for fname, stored, recomp in [("upstream_max_me3", r.get("upstream_max_me3", np.nan), me3_max),
                                          ("prox_u1_sum_t200", r.get("prox_u1_sum_t200", np.nan), burden)]:
                diff = abs(float(stored) - recomp) if pd.notna(stored) else np.nan
                print(f"{gene:<12s}{int(r['y']):>2d}{fname:<22s}{float(stored):>12.4f}{recomp:>12.4f}{diff:>10.4f}",
                      file=sys.stderr)

        rfp = os.path.join(args.rf_dir, f"{gene}.parquet")
        canon_center = sub.loc[sub["y"] == 1, "center"]
        if os.path.exists(rfp) and not canon_center.empty:
            rf = pd.read_parquet(rfp, columns=["window_start_genomic", "is_constitutive_cpa"])
            rf_canon = rf.loc[rf["is_constitutive_cpa"].astype(bool), "window_start_genomic"].to_numpy()
            if rf_canon.size:
                tc = int(canon_center.iloc[0])
                nearest = int(rf_canon[np.argmin(np.abs(rf_canon - tc))])
                print(f"  {gene}: training canonical center={tc}  RF is_constitutive_cpa={nearest}  "
                      f"offset={tc - nearest}  -> pass --center_offset {tc - nearest} if nonzero", file=sys.stderr)

    print("\n[check] absdiff ~0 on check 1 means extraction reproduces training. "
          "A constant nonzero offset on check 2 is what to pass to the gate.", file=sys.stderr)


if __name__ == "__main__":
    main()