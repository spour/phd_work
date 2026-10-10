#!/usr/bin/env python3
# rewrite __wg columns in features.tsv using sequence-derived per-gene baselines instead
import argparse
import os
import numpy as np
import pandas as pd


AXES = [
    "upstream_mean_u1",
    "upstream_max_me3",
    "term_3ss",
    "local_me5",
    "pas_prox_u1",
    "dn5ss_has_strong",
    "interv_me5_max",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--baselines_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--filter_to_baseline_genes", action="store_true",
                     help="Drop sites whose gene has no usable baseline "
                          "(default: keep all sites; missing baselines fall "
                          "back to global median for that axis).")
    ap.add_argument("--min_samples", type=int, default=10,
                     help="Treat baselines with fewer than this many usable "
                          "samples as missing. Default 10.")
    args = ap.parse_args()

    print(f"[apply] reading features from {args.features_tsv}")
    df = pd.read_csv(args.features_tsv, sep="\t")
    print(f"[apply]   {len(df):,} sites, {df['gene'].nunique():,} unique genes")

    print(f"[apply] reading baselines from {args.baselines_tsv}")
    base = pd.read_csv(args.baselines_tsv, sep="\t")
    print(f"[apply]   {len(base):,} gene rows")

    dup_count = base["gene"].duplicated(keep=False).sum()
    if dup_count > 0:
        print(f"[apply] {dup_count:,} duplicate gene rows; combining by "
               f"sample-weighted mean")
        baseline_cols = [c for c in base.columns if c.startswith("baseline_")]
        def combine(group):
            w = group["n_samples_used"].astype(float).values
            if w.sum() == 0:
                return pd.Series({"n_samples_used": 0,
                                   **{c: np.nan for c in baseline_cols}})
            out = {"n_samples_used": int(w.sum())}
            for c in baseline_cols:
                v = group[c].astype(float).values
                mask = np.isfinite(v) & (w > 0)
                if mask.any():
                    out[c] = float(np.average(v[mask], weights=w[mask]))
                else:
                    out[c] = np.nan
            return pd.Series(out)
        base = base.groupby("gene", sort=False).apply(combine).reset_index()
        print(f"[apply]   deduplicated to {len(base):,} unique gene rows")

    for ax in AXES:
        col = f"baseline_{ax}"
        if col not in base.columns:
            raise SystemExit(f"Missing column {col} in {args.baselines_tsv}")

    print(f"[apply] deriving 7 raw axes from bin/engineered features")
    bin_u1_mean_cols = [f"bin{i}_u1_mean" for i in range(1, 7)]
    bin_me3_max_cols = [f"bin{i}_me3_max" for i in range(1, 7)]

    df["upstream_mean_u1"] = df[bin_u1_mean_cols].mean(axis=1, skipna=True)
    df["upstream_max_me3"] = df[bin_me3_max_cols].max(axis=1, skipna=True)
    df["term_3ss"] = df.get("term_me3_top3mean_500_25", np.nan)
    df["local_me5"] = df.get("bin6_me5_mean", np.nan)
    df["pas_prox_u1"] = df.get("bin6_u1_mean", np.nan)
    if "dn5ss_nearest_strong_dist" in df.columns:
        df["dn5ss_has_strong"] = np.where(
            df["dn5ss_nearest_strong_dist"].notna(), 1.0, 0.0)
    else:
        df["dn5ss_has_strong"] = 0.0
    df["interv_me5_max"] = df.get("interv_me5_max_500", pd.Series(np.nan, index=df.index))
    if "interv_me5_max_1000" in df.columns:
        m = df["interv_me5_max"].isna()
        df.loc[m, "interv_me5_max"] = df.loc[m, "interv_me5_max_1000"]

    for ax in AXES:
        n_finite = int(np.isfinite(df[ax]).sum())
        print(f"[apply]   raw {ax}: {n_finite:,}/{len(df):,} finite, "
               f"mean={float(np.nanmean(df[ax])):+.4f} "
               f"std={float(np.nanstd(df[ax])):.4f}")

    insufficient = base["n_samples_used"] < args.min_samples
    if insufficient.any():
        print(f"[apply] {insufficient.sum():,} baselines have <{args.min_samples} "
               f"samples; marking as NaN")
        for ax in AXES:
            base.loc[insufficient, f"baseline_{ax}"] = np.nan

    feat_genes = set(df["gene"].unique())
    base_genes = set(base["gene"].unique())
    overlap = feat_genes & base_genes
    missing = feat_genes - base_genes
    print(f"[apply] coverage: {len(overlap):,}/{len(feat_genes):,} feature "
           f"genes have a baseline ({100*len(overlap)/len(feat_genes):.1f}%)")
    if missing:
        print(f"[apply] {len(missing):,} feature genes missing baseline; "
               f"will use global median fallback")

    if args.filter_to_baseline_genes:
        keep_genes = set(base.loc[~insufficient, "gene"]) & feat_genes
        before = len(df)
        df = df[df["gene"].isin(keep_genes)].copy()
        print(f"[apply] filtered to {len(keep_genes):,} genes with usable "
               f"baselines: {before:,} -> {len(df):,} sites")

    print(f"[apply] computing new __wg columns")
    base_indexed = base.set_index("gene")
    for ax in AXES:
        if ax not in df.columns:
            print(f"[apply]   WARNING: raw axis '{ax}' not in features.tsv; "
                   f"can't compute __wg version")
            continue
        gene_to_baseline = base_indexed[f"baseline_{ax}"]
        baseline_per_site = df["gene"].map(gene_to_baseline)
        global_median = float(np.nanmedian(df[ax]))
        baseline_per_site = baseline_per_site.fillna(global_median)
        new_wg_col = f"{ax}__wg"
        df[new_wg_col] = df[ax] - baseline_per_site

        n_used_baseline = (df["gene"].map(gene_to_baseline).notna()).sum()
        n_used_fallback = len(df) - n_used_baseline
        finite_mean = float(np.nanmean(df[new_wg_col]))
        finite_std = float(np.nanstd(df[new_wg_col]))
        print(f"[apply]   {new_wg_col}: "
               f"{n_used_baseline:,} from baseline, {n_used_fallback:,} from "
               f"global-median fallback. "
               f"mean={finite_mean:+.4f} std={finite_std:.4f}")

    out_dir = os.path.dirname(os.path.abspath(args.out_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[apply] wrote {args.out_tsv}  ({len(df):,} rows, "
           f"{len(df.columns)} columns)")


if __name__ == "__main__":
    main()