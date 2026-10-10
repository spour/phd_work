#!/usr/bin/env python3
# rebuild the manifest_all.parquet from the per-gene parquets directly
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--genes_bed", default=None,
                    help="Optional: original gene BED to join chrom/strand info")
    args = ap.parse_args()

    per_gene_dir = os.path.join(args.scan_dir, "per_gene")
    parquets = sorted(glob.glob(os.path.join(per_gene_dir, "*.parquet")))
    print(f"Found {len(parquets):,} per-gene parquets")
    if not parquets:
        sys.exit("No per-gene parquets found.")

    # read each parquet's metadata (the small stuff)
    rows = []
    n_no_canonical = 0
    for i, pq in enumerate(parquets):
        gene = os.path.basename(pq).replace(".parquet", "")
        try:
            df = pd.read_parquet(pq, columns=[
                "window_offset", "rf_prob", "lr_prob", "cnn_prob",
                "is_constitutive_cpa",
            ])
        except Exception as e:
            print(f"  WARN: failed to read {pq}: {e}")
            continue

        n_windows = len(df)
        canon_idx = df.index[df["is_constitutive_cpa"]]
        if len(canon_idx) == 0:
            canon_offset = None
            n_no_canonical += 1
            rf_picks = lr_picks = cnn_picks = None
        else:
            ci = canon_idx[0]
            canon_offset = int(df.loc[ci, "window_offset"])
            rf_picks  = bool(int(df["rf_prob"].idxmax())  == ci)
            lr_picks  = bool(int(df["lr_prob"].idxmax())  == ci)
            cnn_picks = bool(int(df["cnn_prob"].idxmax()) == ci)

        rows.append({
            "gene":                   gene,
            "n_windows":              int(n_windows),
            "canonical_window_offset": canon_offset,
            "rf_top_offset":          int(df["rf_prob"].idxmax()),
            "lr_top_offset":          int(df["lr_prob"].idxmax()),
            "cnn_top_offset":         int(df["cnn_prob"].idxmax()),
            "rf_top_prob":            float(df["rf_prob"].max()),
            "lr_top_prob":            float(df["lr_prob"].max()),
            "cnn_top_prob":           float(df["cnn_prob"].max()),
            "rf_picks_canonical":     rf_picks,
            "lr_picks_canonical":     lr_picks,
            "cnn_picks_canonical":    cnn_picks,
        })

        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{len(parquets):,}")

    manifest = pd.DataFrame(rows)

    if args.genes_bed:
        gene_info = pd.read_csv(
            args.genes_bed, sep="\t", header=None,
            names=["chrom", "start", "end", "name", "score", "strand"],
            usecols=range(6),
        )[["chrom", "start", "end", "name", "strand"]].rename(columns={"name": "gene"})
        manifest = manifest.merge(gene_info, on="gene", how="left")

    out = os.path.join(args.scan_dir, "manifest_all.parquet")
    manifest.to_parquet(out, index=False, compression="snappy")
    print(f"\nWrote {out}  ({len(manifest):,} rows)")
    print(f"Genes without canonical CPA window: {n_no_canonical:,}")

    # headline metrics
    scored = manifest[manifest["canonical_window_offset"].notna()]
    print(f"Genes with valid canonical window: {len(scored):,}")

    if len(scored):
        rf_hit  = scored["rf_picks_canonical"].mean()
        lr_hit  = scored["lr_picks_canonical"].mean()
        cnn_hit = scored["cnn_picks_canonical"].mean()
        print(f"\nGene-end specificity (top-1 picks canonical CPA window):")
        print(f"  RF : {rf_hit*100:5.2f}%   ({int(scored['rf_picks_canonical'].sum()):,}/{len(scored):,})")
        print(f"  LR : {lr_hit*100:5.2f}%   ({int(scored['lr_picks_canonical'].sum()):,}/{len(scored):,})")
        print(f"  CNN: {cnn_hit*100:5.2f}%   ({int(scored['cnn_picks_canonical'].sum()):,}/{len(scored):,})")

        print(f"\nTolerance-based (|top_offset - canonical_offset| <= tol):")
        print(f"  {'tol':>4}  {'RF':>7}  {'LR':>7}  {'CNN':>7}")
        for tol in [0, 5, 10, 25, 50, 100, 250, 500]:
            row = {}
            for m in ["rf", "lr", "cnn"]:
                diff = (scored[f"{m}_top_offset"] - scored["canonical_window_offset"]).abs()
                row[m] = float((diff <= tol).mean())
            print(f"  {tol:>4}  {row['rf']*100:6.2f}%  {row['lr']*100:6.2f}%  {row['cnn']*100:6.2f}%")


if __name__ == "__main__":
    main()