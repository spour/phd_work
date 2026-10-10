#!/usr/bin/env python3
# look up scores for a list of cryptic sites in the per-gene parquet scans
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd


def parse_gene_name(name_field):
    return name_field.split("::")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_bed", required=True,
                    help="BED of cryptic sites (chrom, start, end, name, score, strand)")
    ap.add_argument("--scan_dir",  required=True,
                    help="Dir containing per_gene/*.parquet")
    ap.add_argument("--out_tsv",   required=True)
    ap.add_argument("--paper_prob_col", type=int, default=4,
                    help="0-based column index of paper's RF prob (default: 4, the 5th column)")
    args = ap.parse_args()

    sites = pd.read_csv(
        args.sites_bed, sep="\t", header=None,
        names=["chrom", "start", "end", "name", "paper_prob", "strand"],
        usecols=range(6),
    )
    print(f"Loaded {len(sites):,} sites from {args.sites_bed}")
    sites["gene"] = sites["name"].map(parse_gene_name)

    per_gene_dir = os.path.join(args.scan_dir, "per_gene")
    if not os.path.isdir(per_gene_dir):
        sys.exit(f"ERROR: {per_gene_dir} does not exist")

    out_rows = []
    n_missing_gene  = 0
    n_missing_window = 0
    grouped = sites.groupby("gene")
    print(f"Looking up {len(sites):,} sites across {len(grouped):,} genes...")
    for i, (gene, group) in enumerate(grouped):
        pq = os.path.join(per_gene_dir, f"{gene}.parquet")
        if not os.path.exists(pq):
            n_missing_gene += len(group)
            for _, s in group.iterrows():
                out_rows.append({**s.to_dict(),
                                 "rf_prob": np.nan, "lr_prob": np.nan,
                                 "cnn_prob": np.nan, "found": False})
            continue

        df = pd.read_parquet(pq, columns=[
            "window_start_genomic", "rf_prob", "lr_prob", "cnn_prob",
            "is_constitutive_cpa",
        ])
        lookup = pd.Series(df.index, index=df["window_start_genomic"].values)

        for _, s in group.iterrows():
            idx = lookup.get(s["start"])
            if idx is None:
                n_missing_window += 1
                out_rows.append({**s.to_dict(),
                                 "rf_prob": np.nan, "lr_prob": np.nan,
                                 "cnn_prob": np.nan, "found": False})
                continue
            row = df.iloc[idx]
            out_rows.append({**s.to_dict(),
                             "rf_prob":  float(row["rf_prob"]),
                             "lr_prob":  float(row["lr_prob"]),
                             "cnn_prob": float(row["cnn_prob"]),
                             "is_constitutive_cpa": bool(row["is_constitutive_cpa"]),
                             "found": True})

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(grouped)} genes processed")

    out = pd.DataFrame(out_rows)
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\nWrote {args.out_tsv}: {len(out):,} rows")
    if n_missing_gene:
        print(f"  Missing gene parquets: {n_missing_gene:,} sites")
    if n_missing_window:
        print(f"  Missing window in parquet: {n_missing_window:,} sites")

    # summary
    found = out[out["found"]].copy()
    n_constitutive = int(found.get("is_constitutive_cpa", pd.Series(dtype=bool)).sum())
    if n_constitutive:
        print(f"\nNote: {n_constitutive:,} of these sites overlap a CONSTITUTIVE CPA "
              f"window (not cryptic in your data). They will be included in summary "
              f"unless you filter them out.")

    summary = {
        "n_sites_input":       int(len(out)),
        "n_sites_scored":      int(len(found)),
        "n_sites_missing":     int(len(out) - len(found)),
        "n_overlapping_canonical": n_constitutive,
        "paper_prob": {
            "mean":   float(found["paper_prob"].mean()),
            "median": float(found["paper_prob"].median()),
            "min":    float(found["paper_prob"].min()),
        },
    }
    print(f"\nMean scores on the {len(found):,} sites successfully looked up:")
    print(f"  Paper RF (input)  : mean={found['paper_prob'].mean():.4f}  "
          f"median={found['paper_prob'].median():.4f}")
    for m in ["rf", "lr", "cnn"]:
        col = f"{m}_prob"
        summary[col] = {
            "mean":   float(found[col].mean()),
            "median": float(found[col].median()),
            "frac_above_0.5":  float((found[col] > 0.5).mean()),
            "frac_above_0.69": float((found[col] > 0.69).mean()),
            "frac_above_0.9":  float((found[col] > 0.9).mean()),
        }
        print(f"  New {m.upper():4s}            : "
              f"mean={found[col].mean():.4f}  "
              f"median={found[col].median():.4f}  "
              f">0.69={(found[col]>0.69).mean()*100:5.1f}%  "
              f">0.9={(found[col]>0.9).mean()*100:5.1f}%")

    summary_path = args.out_tsv.replace(".tsv", "_summary.json")
    if summary_path == args.out_tsv:
        summary_path = args.out_tsv + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()