#!/usr/bin/env python3
# fits gbt with per-bin features gene-baseline centered (instead of raw)
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def fit_gbt_oof(df, feature_cols, y, groups, label):
    X = df[feature_cols].values
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gbt", HistGradientBoostingClassifier(
            max_iter=500, max_depth=4, min_samples_leaf=50, learning_rate=0.05,
            class_weight="balanced", early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42)),
    ])
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    ap = float(average_precision_score(y, oof))
    print(f"  {label:>45s}: AUROC={auc:.4f}  AUPRC={ap:.4f}")
    return auc, ap, oof


def per_gene_metrics(oof, df):
    df = df.copy(); df["pred"] = oof
    n_paired = 0; sum_top1 = 0.0; sum_rr = 0.0; sum_wgauc = 0.0
    for _, g in df.groupby("gene", sort=False):
        y = g["y"].values.astype(int); p = g["pred"].values.astype(float)
        if not (y == 1).any() or not (y == 0).any():
            continue
        n_paired += 1
        sum_top1 += int(y[np.argmax(p)] == 1)
        cs = float(np.max(p[y == 1]))
        rank = 1 + int((p[y == 0] > cs).sum())
        sum_rr += 1.0 / rank
        try:
            sum_wgauc += float(roc_auc_score(y, p))
        except ValueError:
            pass
    if n_paired == 0:
        return np.nan, np.nan, np.nan, 0
    return (sum_top1 / n_paired, sum_rr / n_paired,
             sum_wgauc / n_paired, n_paired)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--baselines_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--min_samples", type=int, default=10,
                     help="Treat baselines with <N samples as NaN.")
    args = ap.parse_args()

    print(f"[test] loading features")
    df = pd.read_csv(args.features_tsv, sep="\t")
    bin_cols = [c for c in df.columns if c.startswith("bin")
                 and (c.endswith("_mean") or c.endswith("_max"))]
    wg_axis_cols = [c for c in df.columns if c.endswith("__wg")
                     and not c.startswith("bin")]
    print(f"[test]   {len(df):,} sites, {df['gene'].nunique():,} unique genes")
    print(f"[test]   {len(bin_cols)} per-bin features, "
           f"{len(wg_axis_cols)} gene-baselined axes already present")

    print(f"[test] loading baselines v2")
    base = pd.read_csv(args.baselines_tsv, sep="\t")
    print(f"[test]   {len(base):,} gene rows")

    dup = base["gene"].duplicated(keep=False).sum()
    if dup > 0:
        print(f"[test]   deduplicating {dup} duplicate gene rows")
        baseline_cols = [c for c in base.columns if c.startswith("baseline_")]
        def combine(g):
            w = g["n_samples_used"].astype(float).values
            if w.sum() == 0:
                return pd.Series({"n_samples_used": 0,
                                   **{c: np.nan for c in baseline_cols}})
            out = {"n_samples_used": int(w.sum())}
            for c in baseline_cols:
                v = g[c].astype(float).values
                m = np.isfinite(v) & (w > 0)
                out[c] = float(np.average(v[m], weights=w[m])) if m.any() else np.nan
            return pd.Series(out)
        base = base.groupby("gene", sort=False).apply(
            combine, include_groups=False).reset_index()

    insufficient = base["n_samples_used"] < args.min_samples
    if insufficient.any():
        print(f"[test]   {insufficient.sum():,} baselines have <{args.min_samples} "
               f"samples; marking as NaN")
        for c in [c for c in base.columns if c.startswith("baseline_")]:
            base.loc[insufficient, c] = np.nan

    missing_perbin_baselines = [c for c in bin_cols
                                  if f"baseline_{c}" not in base.columns]
    if missing_perbin_baselines:
        print(f"[test] WARNING: {len(missing_perbin_baselines)} per-bin features "
               f"have no baseline. Examples: {missing_perbin_baselines[:5]}")
        print(f"[test] re-run compute_gene_baselines_v2.py to add them.")
        raise SystemExit(1)

    print(f"[test] computing per-bin __wg columns")
    base_indexed = base.set_index("gene")
    bin_wg_cols = []
    for c in bin_cols:
        bcol = f"baseline_{c}"
        baseline_per_site = df["gene"].map(base_indexed[bcol])
        global_median = float(np.nanmedian(df[c]))
        baseline_per_site = baseline_per_site.fillna(global_median)
        new_col = f"{c}__bb"
        df[new_col] = df[c] - baseline_per_site
        bin_wg_cols.append(new_col)

    y = df["y"].values.astype(int)
    groups = df["gene"].astype(str).values
    print(f"[test] running 4 GBT configurations\n")

    rows = []

    auc, ap, oof = fit_gbt_oof(df, bin_cols, y, groups,
                                 "48 per-bin RAW (no centering)")
    p1, mrr, wgauc, np_paired = per_gene_metrics(oof, df)
    rows.append({"config": "48_perbin_raw", "n_features": 48,
                  "auroc": auc, "auprc": ap, "p_at_1": p1, "mrr": mrr,
                  "within_gene_auc": wgauc, "n_paired_genes": np_paired})

    auc, ap, oof = fit_gbt_oof(df, bin_wg_cols, y, groups,
                                 "48 per-bin GENE-BASELINE centered (NEW)")
    p1, mrr, wgauc, np_paired = per_gene_metrics(oof, df)
    rows.append({"config": "48_perbin_genebaseline", "n_features": 48,
                  "auroc": auc, "auprc": ap, "p_at_1": p1, "mrr": mrr,
                  "within_gene_auc": wgauc, "n_paired_genes": np_paired})

    auc, ap, oof = fit_gbt_oof(df, bin_cols + wg_axis_cols, y, groups,
                                 "55 combined: 48 RAW + 7 baselined axes")
    p1, mrr, wgauc, np_paired = per_gene_metrics(oof, df)
    rows.append({"config": "55_combined_raw_bins_baselined_axes", "n_features": 55,
                  "auroc": auc, "auprc": ap, "p_at_1": p1, "mrr": mrr,
                  "within_gene_auc": wgauc, "n_paired_genes": np_paired})

    auc, ap, oof = fit_gbt_oof(df, bin_wg_cols + wg_axis_cols, y, groups,
                                 "55 combined: 48 BASELINED + 7 baselined (NEW)")
    p1, mrr, wgauc, np_paired = per_gene_metrics(oof, df)
    rows.append({"config": "55_combined_all_baselined", "n_features": 55,
                  "auroc": auc, "auprc": ap, "p_at_1": p1, "mrr": mrr,
                  "within_gene_auc": wgauc, "n_paired_genes": np_paired})

    out = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(args.out_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\n[test] wrote {args.out_tsv}\n")
    print("Final comparison:")
    print(out[["config", "auroc", "auprc", "p_at_1", "mrr",
                "within_gene_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()