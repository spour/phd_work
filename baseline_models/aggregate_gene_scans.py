#!/usr/bin/env python3
# after all score_genes.py shards finish
import argparse
import glob
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True,
                    help="Directory containing manifest_shard*.parquet")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.scan_dir, "manifest_shard*.parquet")))
    print(f"Found {len(paths)} shard manifests")
    if not paths:
        raise SystemExit("No manifests to aggregate.")

    manifest = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    n = len(manifest)
    n_skipped = int(manifest.get("skipped", pd.Series([False] * n)).sum())
    print(f"Total genes: {n}  ({n_skipped} skipped)")

    scored = manifest[manifest["canonical_window_offset"].notna()].copy()
    print(f"Genes with valid canonical window: {len(scored)}")

    if len(scored) == 0:
        print("WARNING: no genes have a canonical CPA window — was --cpa_bed passed?")
        summary = {
            "n_genes_total":   int(n),
            "n_genes_skipped": n_skipped,
            "n_genes_scored":  0,
        }
    else:
        rf_hit  = scored["rf_picks_canonical"].mean()
        lr_hit  = scored["lr_picks_canonical"].mean()
        cnn_hit = scored["cnn_picks_canonical"].mean()
        print(f"\nGene-end specificity (top-1 picks canonical CPA window):")
        print(f"  RF : {rf_hit*100:5.2f}%   ({int(scored['rf_picks_canonical'].sum())}/{len(scored)})")
        print(f"  LR : {lr_hit*100:5.2f}%   ({int(scored['lr_picks_canonical'].sum())}/{len(scored)})")
        print(f"  CNN: {cnn_hit*100:5.2f}%   ({int(scored['cnn_picks_canonical'].sum())}/{len(scored)})")
        print(f"  (Paper baseline RF reported: 59%)")

        summary = {
            "n_genes_total":          int(n),
            "n_genes_skipped":        n_skipped,
            "n_genes_scored":         int(len(scored)),
            "gene_end_specificity": {
                "rf":  float(rf_hit),
                "lr":  float(lr_hit),
                "cnn": float(cnn_hit),
            },
            "median_n_windows":       int(scored["n_windows"].median()),
            "mean_n_windows":         float(scored["n_windows"].mean()),
        }

    out_manifest = os.path.join(args.scan_dir, "manifest_all.parquet")
    manifest.to_parquet(out_manifest, index=False, compression="snappy")
    print(f"\nWrote {out_manifest}")

    summary_path = os.path.join(args.scan_dir, "gene_end_specificity.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()