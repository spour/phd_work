#!/usr/bin/env python3
# refit the combined 55-feature splicing gbt on the chromosomal split that matches the cpa
import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


TRAIN_CHROMS = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8",
                 "chr9", "chr10", "chr11", "chr12", "chr14", "chr15", "chr16",
                 "chr17"}
VAL_CHROMS = {"chr3", "chr13"}
TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}


def normalize_chrom(c):
    if not isinstance(c, str):
        return c
    c = c.strip()
    if c.lower().startswith("chr"):
        return "chr" + c[3:].upper().replace("X", "X").replace("Y", "Y")
    return "chr" + c.upper()


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
        return {"p_at_1": np.nan, "mrr": np.nan, "wg_auc": np.nan, "n_paired": 0}
    return {"p_at_1": sum_top1 / n_paired,
            "mrr": sum_rr / n_paired,
            "wg_auc": sum_wgauc / n_paired,
            "n_paired": n_paired}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[chrsplit] loading features")
    df = pd.read_csv(args.features_tsv, sep="\t")
    print(f"[chrsplit]   {len(df):,} sites, {df['gene'].nunique():,} unique genes")

    if "chrom" not in df.columns:
        raise SystemExit("features_tsv missing 'chrom' column")

    df["chrom_norm"] = df["chrom"].astype(str).apply(normalize_chrom)
    chrom_dist = df["chrom_norm"].value_counts()
    print(f"[chrsplit] chromosome distribution:")
    for c, n in chrom_dist.items():
        print(f"  {c}: {n:,}")

    df["split"] = "other"
    df.loc[df["chrom_norm"].isin(TRAIN_CHROMS), "split"] = "train"
    df.loc[df["chrom_norm"].isin(VAL_CHROMS), "split"] = "val"
    df.loc[df["chrom_norm"].isin(TEST_CHROMS), "split"] = "test"

    split_counts = df["split"].value_counts()
    print(f"\n[chrsplit] split sizes:")
    for s in ["train", "val", "test", "other"]:
        n = int(split_counts.get(s, 0))
        n_genes = df[df["split"] == s]["gene"].nunique()
        pos_rate = float((df[df["split"] == s]["y"] == 1).mean()) if n > 0 else 0
        print(f"  {s:>5s}: {n:>7,} sites in {n_genes:>6,} genes "
               f"(pos rate {pos_rate:.3f})")
    if (df["split"] == "other").any():
        other_chroms = df.loc[df["split"] == "other", "chrom_norm"].unique()
        print(f"  WARNING: 'other' chroms {sorted(other_chroms)} excluded from training")

    train_genes = set(df.loc[df["split"] == "train", "gene"])
    val_genes = set(df.loc[df["split"] == "val", "gene"])
    test_genes = set(df.loc[df["split"] == "test", "gene"])
    leak_tv = train_genes & val_genes
    leak_tt = train_genes & test_genes
    leak_vt = val_genes & test_genes
    if leak_tv or leak_tt or leak_vt:
        print(f"  GENE LEAKAGE DETECTED:")
        print(f"    train ∩ val: {len(leak_tv)}")
        print(f"    train ∩ test: {len(leak_tt)}")
        print(f"    val ∩ test: {len(leak_vt)}")
        print(f"  Possible cause: gene spans multiple chromosomes "
               f"(or annotation issue)")
    else:
        print(f"  no gene leakage between splits ✓")

    bin_cols = [c for c in df.columns
                 if c.startswith("bin") and (c.endswith("_mean") or c.endswith("_max"))]
    wg_cols = [c for c in df.columns if c.endswith("__wg")]
    feature_cols = bin_cols + wg_cols
    print(f"\n[chrsplit] feature set: {len(bin_cols)} per-bin + "
           f"{len(wg_cols)} gene-baselined = {len(feature_cols)} features")

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "y"].values.astype(int)
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, "y"].values.astype(int)
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "y"].values.astype(int)

    print(f"\n[chrsplit] fitting model (val-based early stopping)")
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gbt", HistGradientBoostingClassifier(
            max_iter=800, max_depth=4, min_samples_leaf=50, learning_rate=0.05,
            class_weight="balanced", early_stopping=True,
            validation_fraction=None,
            n_iter_no_change=30, random_state=args.seed)),
    ])
    pipe.named_steps["gbt"].validation_fraction = 0.15
    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_val = pipe.predict_proba(X_val)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    print(f"\n[chrsplit] performance:")
    print(f"  {'split':<6s} {'AUROC':>7s} {'AUPRC':>7s} {'P@1':>7s} {'MRR':>7s} {'WG-AUC':>7s} {'n_paired':>9s}")
    rows = []
    for split_name, mask, preds in [("train", train_mask, p_train),
                                       ("val", val_mask, p_val),
                                       ("test", test_mask, p_test)]:
        ydf = df.loc[mask].copy().reset_index(drop=True)
        y = ydf["y"].values.astype(int)
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            print(f"  {split_name:<6s} (insufficient class diversity)")
            continue
        auc = roc_auc_score(y, preds)
        ap = average_precision_score(y, preds)
        gm = per_gene_metrics(preds, ydf)
        print(f"  {split_name:<6s} {auc:>7.4f} {ap:>7.4f} "
               f"{gm['p_at_1']:>7.4f} {gm['mrr']:>7.4f} "
               f"{gm['wg_auc']:>7.4f} {gm['n_paired']:>9d}")
        rows.append({"split": split_name, "auroc": float(auc), "auprc": float(ap),
                      "p_at_1": gm["p_at_1"], "mrr": gm["mrr"],
                      "within_gene_auc": gm["wg_auc"],
                      "n_paired_genes": gm["n_paired"],
                      "n_sites": int(mask.sum())})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "chrsplit_metrics.tsv"),
                    sep="\t", index=False)
    print(f"\n[chrsplit] wrote chrsplit_metrics.tsv")

    out_pkl = os.path.join(args.out_dir, "gbt_combined_chrsplit.joblib")
    joblib.dump({
        "model": pipe,
        "feature_cols": feature_cols,
        "feature_groups": {"per_bin": bin_cols, "gene_baselined_axes": wg_cols},
        "train_chroms": sorted(TRAIN_CHROMS),
        "val_chroms": sorted(VAL_CHROMS),
        "test_chroms": sorted(TEST_CHROMS),
        "n_train_sites": int(train_mask.sum()),
        "n_test_sites": int(test_mask.sum()),
    }, out_pkl)
    print(f"[chrsplit] saved model: {out_pkl}")

    test_preds_df = df.loc[test_mask, ["chrom", "center", "gene", "y"]].copy()
    test_preds_df["splicing_prob"] = p_test
    test_preds_df.to_csv(os.path.join(args.out_dir, "test_predictions.tsv"),
                          sep="\t", index=False)
    print(f"[chrsplit] saved test predictions on {len(test_preds_df):,} sites")


if __name__ == "__main__":
    main()