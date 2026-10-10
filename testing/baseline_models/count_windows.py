#!/usr/bin/env python3
# report how many 500-nt windows each model scored
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc_compute


KNOWN_PROB_COLS = {
    "rf":       "rf_prob",
    "lr":       "lr_prob",
    "cnn":      "cnn_prob",
    "aparent2": "aparent2_prob",
}


def fmt_int(n):
    return f"{n:,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True,
                    help="Directory containing per_gene/*.parquet")
    ap.add_argument("--pc_genes_bed", default=None,
                    help="Optional: pc_genes BED for genome-body coverage")
    ap.add_argument("--window_len", type=int, default=500,
                    help="Window length in nt (for coverage math; default 500)")
    ap.add_argument("--json", default=None,
                    help="Write summary to JSON instead of printing")
    args = ap.parse_args()

    per_gene = os.path.join(args.scan_dir, "per_gene")
    parquets = sorted(glob.glob(os.path.join(per_gene, "*.parquet")))
    if not parquets:
        raise SystemExit(f"No parquets under {per_gene}")
    print(f"Scanning {len(parquets):,} parquets under {per_gene}\n")

    n_genes = len(parquets)
    n_windows_total = 0
    per_model = {m: 0 for m in KNOWN_PROB_COLS}
    per_gene_window_counts = []
    detected_cols = set()
    n_errors = 0

    for i, p in enumerate(parquets):
        try:
            pf = pq.ParquetFile(p)
        except Exception as e:
            n_errors += 1
            continue
        n_rows = pf.metadata.num_rows
        n_windows_total += n_rows
        per_gene_window_counts.append(n_rows)
        cols = set(pf.schema_arrow.names)
        detected_cols.update(cols)

        for m, col in KNOWN_PROB_COLS.items():
            if col not in cols:
                continue
            try:
                table = pf.read(columns=[col])
                arr = table.column(col)
                n_null = arr.null_count
                per_model[m] += (n_rows - n_null)
            except Exception:
                per_model[m] += n_rows

        if (i + 1) % 2000 == 0:
            print(f"  ...scanned {i+1:,}/{n_genes:,}")

    arr = np.array(per_gene_window_counts)
    stats = {
        "n_genes":        int(n_genes),
        "n_windows_total": int(n_windows_total),
        "n_errors":       int(n_errors),
        "per_gene_min":   int(arr.min()) if len(arr) else 0,
        "per_gene_max":   int(arr.max()) if len(arr) else 0,
        "per_gene_median": float(np.median(arr)) if len(arr) else 0,
        "per_gene_mean":  float(arr.mean()) if len(arr) else 0,
        "per_model":      {m: int(v) for m, v in per_model.items() if v > 0},
        "models_detected": sorted([m for m in KNOWN_PROB_COLS
                                     if KNOWN_PROB_COLS[m] in detected_cols]),
    }

    if args.pc_genes_bed:
        bp_total = 0
        with open(args.pc_genes_bed) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                p2 = line.rstrip("\n").split("\t")
                bp_total += int(p2[2]) - int(p2[1])
        stats["pc_gene_bp_total"] = int(bp_total)

    # print summary
    print(f"\n========= Window count summary =========")
    print(f"  parquets:           {fmt_int(stats['n_genes'])}")
    if stats["n_errors"]:
        print(f"  read errors:        {fmt_int(stats['n_errors'])}")
    print(f"  total windows:      {fmt_int(stats['n_windows_total'])}")
    print(f"  per-gene windows:")
    print(f"    min     {fmt_int(stats['per_gene_min']):>15}")
    print(f"    median  {stats['per_gene_median']:>15,.0f}")
    print(f"    mean    {stats['per_gene_mean']:>15,.1f}")
    print(f"    max     {fmt_int(stats['per_gene_max']):>15}")

    print(f"\n  Models detected: {stats['models_detected']}")
    print(f"\n  Windows scored per model (non-null prob cells):")
    for m in KNOWN_PROB_COLS:
        if m in stats["per_model"]:
            v = stats["per_model"][m]
            pct = v / max(stats["n_windows_total"], 1) * 100
            print(f"    {m:<10} {fmt_int(v):>15}  ({pct:5.1f}% of total windows)")
        else:
            print(f"    {m:<10} {'(no column)':>15}")

    print(f"\n  Compute volume (windows × {args.window_len} nt, "
          f"DOUBLE-COUNTS overlapping windows):")
    for m in stats["per_model"]:
        v = stats["per_model"][m]
        cells = v * args.window_len
        print(f"    {m:<10}  {fmt_int(cells):>20} nt-cells  ({cells/1e9:.2f} Gnt)")

    if args.pc_genes_bed:
        bp_total = stats["pc_gene_bp_total"]
        print(f"\n  Protein-coding gene-body footprint: {fmt_int(bp_total)} nt "
              f"({bp_total/1e6:.1f} Mb)")
        for m in stats["per_model"]:
            v = stats["per_model"][m]
            print(f"    {m:<10} {v/bp_total:.2f}x average scoring depth "
                  f"(windows per gene-body bp, assuming 1-nt step)")

    print(f"\n========================================")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()