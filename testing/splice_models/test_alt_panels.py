#!/usr/bin/env python3
# try alternative 7-feature panels with gene-baseline centering and report auroc/auprc
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


PANELS = {
    "original_means": [
        "upstream_mean_u1", "upstream_max_me3", "term_3ss", "local_me5",
        "pas_prox_u1", "dn5ss_has_strong", "interv_me5_max",
    ],
    "peaks_max": [
        "upstream_max_u1", "upstream_max_me3", "term_3ss_max", "local_me5_max",
        "pas_prox_u1_max", "dn5ss_best", "all_u1_max",
    ],
    "decay_burden": [
        "upstream_decay_u1", "downstream_decay_u1", "upstream_decay_3ss",
        "downstream_decay_3ss", "me3_prominence", "u1_gradient", "term_3ss",
    ],
    "gradient_prominence": [
        "u1_gradient", "u1_gradient_decay", "me3_prominence",
        "me3_max_prominence", "local_suppression", "terminal_identity",
        "dn5ss_has_strong",
    ],
    "terminal_exon": [
        "term_3ss", "interv_me5_max", "interv_me5_n_strong", "interv_geometry",
        "exon_clean", "upstream_mean_u1", "dn5ss_has_strong",
    ],
    "winner_take_all": [
        "upstream_max_u1", "term_3ss_max", "local_me5_max", "pas_prox_u1_max",
        "all_u1_max", "upstream_max_me3", "dn5ss_best",
    ],
    "mixed_mean_peak": [
        "upstream_mean_u1", "term_3ss", "local_me5", "pas_prox_u1",
        "me3_prominence", "u1_gradient", "interv_me5_max",
    ],
    "tele_integrity": [
        "upstream_mean_u1", "downstream_mean_u1", "pas_prox_u1",
        "dn5ss_has_strong", "dn5ss_best", "interv_me5_max", "term_3ss",
    ],
}


def derive_all_axes(df):
    bin_u1m = [f"bin{i}_u1_mean" for i in range(1, 7)]
    bin_u1x = [f"bin{i}_u1_max"  for i in range(1, 7)]
    bin_me3 = [f"bin{i}_me3_max" for i in range(1, 7)]
    bin_me3m = [f"bin{i}_me3_max" for i in range(1, 9)]

    df["upstream_mean_u1"] = df[bin_u1m].mean(axis=1, skipna=True)
    df["downstream_mean_u1"] = df[[f"bin{i}_u1_mean" for i in range(7, 9)]
                                   ].mean(axis=1, skipna=True)
    df["upstream_max_u1"] = df[bin_u1x].max(axis=1, skipna=True)
    df["upstream_max_me3"] = df[bin_me3].max(axis=1, skipna=True)
    df["upstream_mean_me3"] = df[bin_me3].mean(axis=1, skipna=True)
    df["pas_prox_u1"] = df.get("bin6_u1_mean", pd.Series(np.nan, index=df.index))
    df["pas_prox_u1_max"] = df.get("bin6_u1_max", pd.Series(np.nan, index=df.index))
    df["local_me5"] = df.get("bin6_me5_mean", pd.Series(np.nan, index=df.index))
    df["local_me5_max"] = df.get("bin6_me5_max", pd.Series(np.nan, index=df.index))

    df["term_3ss"] = df.get("term_me3_top3mean_500_25",
                              pd.Series(np.nan, index=df.index))
    df["term_3ss_max"] = df.get("term_me3_max_500_25",
                                  pd.Series(np.nan, index=df.index))

    if "dn5ss_nearest_strong_dist" in df.columns:
        df["dn5ss_has_strong"] = np.where(
            df["dn5ss_nearest_strong_dist"].notna(), 1.0, 0.0)
    else:
        df["dn5ss_has_strong"] = 0.0
    df["dn5ss_best"] = df.get("dn5ss_best_score",
                                pd.Series(np.nan, index=df.index))

    df["interv_me5_max"] = df.get("interv_me5_max_500",
                                    pd.Series(np.nan, index=df.index))
    df["interv_me5_n_strong"] = df.get("interv_me5_n_strong_500",
                                          pd.Series(np.nan, index=df.index))
    df["interv_geometry"] = df.get("interv_exon_geometry_500",
                                      pd.Series(np.nan, index=df.index))
    df["exon_clean"] = np.where(
        (df["interv_me5_n_strong"].fillna(0) == 0) &
        (df["interv_geometry"].fillna(0) == 1), 1.0, 0.0)

    for tau in (50, 150, 400):
        if f"all_u1_decay_up_tau{tau}" not in df.columns:
            df[f"all_u1_decay_up_tau{tau}"] = df["upstream_mean_u1"]
        if f"all_u1_decay_dn_tau{tau}" not in df.columns:
            df[f"all_u1_decay_dn_tau{tau}"] = df["downstream_mean_u1"]
        if f"all_me3_decay_up_tau{tau}" not in df.columns:
            df[f"all_me3_decay_up_tau{tau}"] = df["upstream_max_me3"]
        if f"all_me3_decay_dn_tau{tau}" not in df.columns:
            df[f"all_me3_decay_dn_tau{tau}"] = df["upstream_mean_me3"]
    df["upstream_decay_u1"] = df["all_u1_decay_up_tau400"]
    df["downstream_decay_u1"] = df["all_u1_decay_dn_tau400"]
    df["upstream_decay_3ss"] = df["all_me3_decay_up_tau400"]
    df["downstream_decay_3ss"] = df["all_me3_decay_dn_tau400"]

    if "all_u1_max" not in df.columns:
        df["all_u1_max"] = df[[f"bin{i}_u1_max" for i in range(1, 9)]
                                 ].max(axis=1, skipna=True)

    df["u1_gradient"] = df["upstream_mean_u1"] - df["downstream_mean_u1"]
    df["u1_gradient_decay"] = (df["all_u1_decay_up_tau400"] -
                                  df["all_u1_decay_dn_tau400"])
    df["me3_prominence"] = df["term_3ss"] - df["upstream_mean_me3"]
    df["me3_max_prominence"] = df["term_3ss"] - df["upstream_max_me3"]
    df["local_suppression"] = df["pas_prox_u1"] + df["local_me5"]
    df["terminal_identity"] = df["upstream_mean_u1"] + df["me3_prominence"]

    return df


def center_panel(df, baselines, panel_cols):
    base_indexed = baselines.set_index("gene")
    out = df.copy()
    for col in panel_cols:
        base_col = f"baseline_{col}"
        if base_col in base_indexed.columns:
            baseline_per_site = df["gene"].map(base_indexed[base_col])
            global_median = float(np.nanmedian(df[col]))
            baseline_per_site = baseline_per_site.fillna(global_median)
            out[f"{col}__wg"] = df[col] - baseline_per_site
        else:
            out[f"{col}__wg"] = df[col] - float(np.nanmedian(df[col]))
    return out


def fit_panel(df, panel_cols, y, groups, label):
    wg_cols = [f"{c}__wg" for c in panel_cols]
    missing = [c for c in wg_cols if c not in df.columns]
    if missing:
        return label, np.nan, np.nan, f"missing: {missing}"

    X = df[wg_cols].values
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gbt", HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, min_samples_leaf=50,
            learning_rate=0.05, class_weight="balanced",
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42)),
    ])
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    ap = float(average_precision_score(y, oof))
    return label, auc, ap, "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--baselines_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    print(f"[panels] reading features")
    df = pd.read_csv(args.features_tsv, sep="\t")
    df = derive_all_axes(df)

    print(f"[panels] reading baselines")
    baselines = pd.read_csv(args.baselines_tsv, sep="\t")
    dup = baselines["gene"].duplicated(keep=False).sum()
    if dup > 0:
        print(f"[panels]   deduplicating {dup} duplicate gene rows")
        baseline_cols = [c for c in baselines.columns if c.startswith("baseline_")]
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
        baselines = baselines.groupby("gene", sort=False).apply(combine).reset_index()

    all_panel_features = set()
    for p in PANELS.values():
        all_panel_features.update(p)
    in_baselines = {c for c in all_panel_features
                     if f"baseline_{c}" in baselines.columns}
    missing_from_baselines = all_panel_features - in_baselines
    if missing_from_baselines:
        print(f"[panels] features needing global-median centering "
              f"(no gene baseline): {sorted(missing_from_baselines)}")

    y = df["y"].values.astype(int)
    groups = df["gene"].astype(str).values

    rows = []
    for name, panel in PANELS.items():
        missing_raw = [c for c in panel if c not in df.columns]
        if missing_raw:
            print(f"[panels] {name}: SKIP (missing raw: {missing_raw})")
            continue

        df_centered = center_panel(df, baselines, panel)
        label, auc, ap, status = fit_panel(df_centered, panel, y, groups, name)
        rows.append({"panel": label, "n_features": len(panel),
                      "auroc": auc, "auprc": ap, "status": status,
                      "features": ",".join(panel)})
        print(f"[panels] {name:>22s}: AUROC={auc:.4f}  AUPRC={ap:.4f}  ({status})")

    out = pd.DataFrame(rows).sort_values("auprc", ascending=False).reset_index(drop=True)
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\n[panels] wrote {args.out_tsv}")
    print()
    print("Ranked by AUPRC:")
    print(out[["panel", "auroc", "auprc"]].to_string(index=False))


if __name__ == "__main__":
    main()