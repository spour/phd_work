#!/usr/bin/env python3
# restrict to rf candidate windows, learn the optimal combiner on val chroms
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                                roc_curve, precision_recall_curve)
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_CHROMS = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8",
                 "chr9", "chr10", "chr11", "chr12", "chr14", "chr15", "chr16",
                 "chr17"}
VAL_CHROMS = {"chr3", "chr13"}
TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}


def normalize_chrom(c):
    if not isinstance(c, str):
        return c
    c = c.strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def safe_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def find_peaks_nms(scores, positions, min_distance_nt, max_peaks=None):
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.array([], dtype=int)
    order = np.argsort(scores)[::-1]
    chosen = []
    chosen_positions = []
    for idx in order:
        if not finite_mask[idx]:
            continue
        p = positions[idx]
        if any(abs(p - cp) <= min_distance_nt for cp in chosen_positions):
            continue
        chosen.append(int(idx))
        chosen_positions.append(int(p))
        if max_peaks is not None and len(chosen) >= max_peaks:
            break
    return np.array(chosen, dtype=int)


def build_candidate_table(gene, cpa_path, splice_path, canonical_positions,
                           strategy, max_candidates, threshold,
                           min_peak_distance, cpa_model, tolerance_nt):
    try:
        cpa = pd.read_parquet(cpa_path, columns=["window_start_genomic",
                                                   cpa_model,
                                                   "is_constitutive_cpa"])
        sp = pd.read_parquet(splice_path,
                               columns=["window_start_genomic", "splicing_prob"])
    except Exception:
        return None
    merged = cpa.merge(sp, on="window_start_genomic", how="inner")
    merged = merged.dropna(subset=[cpa_model, "splicing_prob"])
    if len(merged) == 0:
        return None
    positions = merged["window_start_genomic"].values
    rf_scores = merged[cpa_model].values
    splice_scores = merged["splicing_prob"].values

    if strategy == "peaks":
        cand_idx = find_peaks_nms(rf_scores, positions, min_peak_distance,
                                     max_peaks=max_candidates)
    elif strategy == "topk":
        if max_candidates >= len(merged):
            cand_idx = np.arange(len(merged))
        else:
            cand_idx = np.argpartition(rf_scores, -max_candidates)[-max_candidates:]
    elif strategy == "threshold":
        cand_idx = np.where(rf_scores > threshold)[0]
        if max_candidates is not None and len(cand_idx) > max_candidates:
            top = np.argsort(rf_scores[cand_idx])[::-1][:max_candidates]
            cand_idx = cand_idx[top]
    else:
        raise ValueError(f"unknown strategy {strategy}")

    if len(cand_idx) == 0:
        return None

    cand_positions = positions[cand_idx]
    cand_rf = rf_scores[cand_idx]
    cand_splice = splice_scores[cand_idx]
    in_zone = np.zeros(len(cand_idx), dtype=bool)
    for cp in canonical_positions:
        in_zone |= np.abs(cand_positions - cp) <= tolerance_nt

    return pd.DataFrame({
        "gene": gene,
        "position": cand_positions,
        "rf_prob": cand_rf,
        "splicing_prob": cand_splice,
        "is_canonical_zone": in_zone.astype(int),
    })


def per_gene_p_at_1(df, score_col):
    if len(df) == 0:
        return np.nan, 0
    n = 0
    hits = 0
    for g, sub in df.groupby("gene", sort=False):
        if (sub["is_canonical_zone"] == 1).sum() == 0:
            continue
        n += 1
        s = sub[score_col].values
        if not np.isfinite(s).any():
            continue
        top = sub.iloc[int(np.nanargmax(s))]
        hits += int(top["is_canonical_zone"] == 1)
    return (hits / n if n > 0 else np.nan), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_parquet_dir", required=True)
    ap.add_argument("--splicing_parquet_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--candidate_strategy",
                     choices=["peaks", "topk", "threshold"], default="peaks")
    ap.add_argument("--max_candidates_per_gene", type=int, default=50)
    ap.add_argument("--cpa_threshold", type=float, default=0.5)
    ap.add_argument("--min_peak_distance", type=int, default=250)
    ap.add_argument("--tolerance_nt", type=int, default=250)
    ap.add_argument("--cpa_model", default="rf_prob")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[combiner] strategy={args.candidate_strategy} "
           f"max_cand={args.max_candidates_per_gene} "
           f"min_peak_dist={args.min_peak_distance}nt "
           f"tolerance={args.tolerance_nt}nt")

    pc = pd.read_csv(args.pc_genes_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    pc["gene"] = pc["name"].astype(str)
    if pc["gene"].str.contains("_").any():
        pc["gene"] = pc["gene"].str.split("_").str[-1]
    pc["chrom_norm"] = pc["chrom"].apply(normalize_chrom)
    pc = pc.drop_duplicates(subset="gene", keep="first")
    pc["split"] = "other"
    pc.loc[pc["chrom_norm"].isin(TRAIN_CHROMS), "split"] = "train"
    pc.loc[pc["chrom_norm"].isin(VAL_CHROMS), "split"] = "val"
    pc.loc[pc["chrom_norm"].isin(TEST_CHROMS), "split"] = "test"
    print(f"[combiner] gene splits: {pc['split'].value_counts().to_dict()}")

    candidate_tables = {"val": [], "test": []}
    n_skipped = {"val": 0, "test": 0}

    for i, row in pc.iterrows():
        split = row["split"]
        if split not in candidate_tables:
            continue
        gene = row["gene"]
        cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
        splice_path = os.path.join(args.splicing_parquet_dir, f"{gene}.parquet")
        if not os.path.exists(cpa_path) or not os.path.exists(splice_path):
            n_skipped[split] += 1
            continue
        try:
            cpa_meta = pd.read_parquet(cpa_path,
                                         columns=["window_start_genomic",
                                                   "is_constitutive_cpa"])
        except Exception:
            n_skipped[split] += 1
            continue
        canonical_positions = cpa_meta.loc[
            cpa_meta["is_constitutive_cpa"] == True,
            "window_start_genomic"].values
        if len(canonical_positions) == 0:
            n_skipped[split] += 1
            continue

        ct = build_candidate_table(
            gene, cpa_path, splice_path, canonical_positions,
            args.candidate_strategy, args.max_candidates_per_gene,
            args.cpa_threshold, args.min_peak_distance,
            args.cpa_model, args.tolerance_nt)
        if ct is None or len(ct) == 0:
            n_skipped[split] += 1
            continue
        candidate_tables[split].append(ct)

        if (i + 1) % 2000 == 0:
            print(f"[combiner]   processed {i+1:,} genes "
                   f"(val: {len(candidate_tables['val']):,} done, "
                   f"test: {len(candidate_tables['test']):,} done)")

    val_df = pd.concat(candidate_tables["val"], ignore_index=True) \
        if candidate_tables["val"] else pd.DataFrame()
    test_df = pd.concat(candidate_tables["test"], ignore_index=True) \
        if candidate_tables["test"] else pd.DataFrame()
    print(f"\n[combiner] val candidates: {len(val_df):,} rows from "
           f"{val_df['gene'].nunique() if len(val_df) > 0 else 0:,} genes "
           f"(skipped: {n_skipped['val']:,})")
    print(f"[combiner] test candidates: {len(test_df):,} rows from "
           f"{test_df['gene'].nunique() if len(test_df) > 0 else 0:,} genes "
           f"(skipped: {n_skipped['test']:,})")
    if len(val_df) > 0:
        print(f"[combiner] val positive rate: "
               f"{float(val_df['is_canonical_zone'].mean()):.4f}")
    if len(test_df) > 0:
        print(f"[combiner] test positive rate: "
               f"{float(test_df['is_canonical_zone'].mean()):.4f}")

    val_df.to_csv(os.path.join(args.out_dir, "val_candidates.tsv"),
                   sep="\t", index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_candidates.tsv"),
                    sep="\t", index=False)

    if len(val_df) < 100 or len(test_df) < 100:
        print("[combiner] ERROR: not enough candidates to fit/evaluate")
        return

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 1: redundancy via correlation of logit(rf) vs logit(splice)")
    print(f"{'='*70}")
    for split_name, sdf in [("val", val_df), ("test", test_df)]:
        l_rf = safe_logit(sdf["rf_prob"].values)
        l_sp = safe_logit(sdf["splicing_prob"].values)
        pear_r, pear_p = pearsonr(l_rf, l_sp)
        spear_r, spear_p = spearmanr(l_rf, l_sp)
        print(f"  [{split_name}] n={len(sdf):,}")
        print(f"    pearson(logit_rf, logit_splice)  = "
               f"{pear_r:>+.4f}  (p={pear_p:.2e})")
        print(f"    spearman(logit_rf, logit_splice) = "
               f"{spear_r:>+.4f}  (p={spear_p:.2e})")
        for label, mask in [("in_zone", sdf["is_canonical_zone"] == 1),
                              ("outside", sdf["is_canonical_zone"] == 0)]:
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(l_rf[mask], l_sp[mask])
            print(f"      ({label:>8s}, n={int(mask.sum()):>6,}): "
                   f"pearson = {r:>+.4f}")

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 2: learned combiner (fit on val, eval on test)")
    print(f"{'='*70}")

    X_val = np.column_stack([safe_logit(val_df["rf_prob"].values),
                                safe_logit(val_df["splicing_prob"].values)])
    y_val = val_df["is_canonical_zone"].values
    X_test = np.column_stack([safe_logit(test_df["rf_prob"].values),
                                 safe_logit(test_df["splicing_prob"].values)])
    y_test = test_df["is_canonical_zone"].values

    clf_full = LogisticRegression(max_iter=1000, class_weight="balanced",
                                    solver="lbfgs")
    clf_full.fit(X_val, y_val)
    w_intercept = float(clf_full.intercept_[0])
    w_rf = float(clf_full.coef_[0, 0])
    w_splice = float(clf_full.coef_[0, 1])

    print(f"  Combiner fit on val (n={len(val_df):,}):")
    print(f"    intercept      = {w_intercept:>+.4f}")
    print(f"    w_logit_rf     = {w_rf:>+.4f}")
    print(f"    w_logit_splice = {w_splice:>+.4f}")
    if abs(w_rf) > 1e-6:
        print(f"    splice/rf ratio= {w_splice/w_rf:>+.4f}")

    clf_rf_only = LogisticRegression(max_iter=1000, class_weight="balanced",
                                       solver="lbfgs")
    clf_rf_only.fit(X_val[:, [0]], y_val)
    print(f"\n  Reference rf-only LR fit on val:")
    print(f"    intercept      = {float(clf_rf_only.intercept_[0]):>+.4f}")
    print(f"    w_logit_rf     = {float(clf_rf_only.coef_[0, 0]):>+.4f}")

    test_df = test_df.copy()
    test_df["score_cpa_only"] = test_df["rf_prob"]
    test_df["score_splicing_only"] = test_df["splicing_prob"]
    test_df["score_product"] = test_df["rf_prob"] * test_df["splicing_prob"]
    test_df["score_logit_cpa_07"] = (0.7 * safe_logit(test_df["rf_prob"]) +
                                       0.3 * safe_logit(test_df["splicing_prob"]))
    test_df["score_learned"] = clf_full.predict_proba(X_test)[:, 1]
    test_df["score_rf_only_lr"] = clf_rf_only.predict_proba(X_test[:, [0]])[:, 1]

    print(f"\n  Test pooled metrics (n={len(test_df):,} candidates, "
           f"positive rate {float(y_test.mean()):.4f}):")
    print(f"  {'scorer':<22s} {'AUROC':>8s} {'AUPRC':>8s} {'lift_vs_rf':>11s}")
    rf_auc = roc_auc_score(y_test, test_df["score_cpa_only"])
    pooled_rows = []
    for label, col in [("cpa_only", "score_cpa_only"),
                          ("splicing_only", "score_splicing_only"),
                          ("product", "score_product"),
                          ("logit_cpa_07", "score_logit_cpa_07"),
                          ("learned_combiner", "score_learned"),
                          ("rf_only_lr_baseline", "score_rf_only_lr")]:
        try:
            auc = roc_auc_score(y_test, test_df[col])
            ap = average_precision_score(y_test, test_df[col])
        except ValueError:
            auc, ap = np.nan, np.nan
        lift = auc - rf_auc
        print(f"  {label:<22s} {auc:>8.4f} {ap:>8.4f} {lift:>+11.4f}")
        pooled_rows.append({"split": "test", "scorer": label,
                             "auroc": float(auc), "auprc": float(ap),
                             "auroc_lift_vs_rf": float(lift)})

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 3: per-gene P@1 (top-1 candidate in canonical zone)")
    print(f"{'='*70}")
    print(f"  {'scorer':<22s} {'P@1':>7s} {'n_genes':>9s}")
    per_gene_rows = []
    for label, col in [("cpa_only", "score_cpa_only"),
                          ("splicing_only", "score_splicing_only"),
                          ("product", "score_product"),
                          ("logit_cpa_07", "score_logit_cpa_07"),
                          ("learned_combiner", "score_learned")]:
        p1, n_genes = per_gene_p_at_1(test_df, col)
        print(f"  {label:<22s} {p1:>7.4f} {n_genes:>9d}")
        per_gene_rows.append({"split": "test", "scorer": label,
                               "p_at_1": float(p1), "n_genes": int(n_genes)})

    all_metrics = pd.DataFrame(pooled_rows + per_gene_rows)
    all_metrics.to_csv(os.path.join(args.out_dir, "combiner_metrics.tsv"),
                        sep="\t", index=False)

    coef_df = pd.DataFrame([{
        "intercept": w_intercept, "w_logit_rf": w_rf,
        "w_logit_splice": w_splice,
        "splice_to_rf_ratio": w_splice / w_rf if abs(w_rf) > 1e-6 else np.nan,
        "n_val_candidates": len(val_df),
        "n_val_positives": int(y_val.sum()),
        "n_test_candidates": len(test_df),
        "n_test_positives": int(y_test.sum()),
    }])
    coef_df.to_csv(os.path.join(args.out_dir, "combiner_coefficients.tsv"),
                    sep="\t", index=False)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"cpa_only": "#888888", "splicing_only": "#3E8C68",
               "product": "#C23B3B", "logit_cpa_07": "#E07B00",
               "learned_combiner": "#1F77B4"}
    for label, col in [("cpa_only", "score_cpa_only"),
                          ("splicing_only", "score_splicing_only"),
                          ("product", "score_product"),
                          ("logit_cpa_07", "score_logit_cpa_07"),
                          ("learned_combiner", "score_learned")]:
        s = test_df[col].values
        fpr, tpr, _ = roc_curve(y_test, s)
        prec, rec, _ = precision_recall_curve(y_test, s)
        auc = roc_auc_score(y_test, s)
        ap = average_precision_score(y_test, s)
        ax_roc.plot(fpr, tpr, color=colors[label], lw=2,
                     label=f"{label} (AUC={auc:.3f})")
        ax_pr.plot(rec, prec, color=colors[label], lw=2,
                    label=f"{label} (AP={ap:.3f})")
    base = float(y_test.mean())
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_pr.axhline(base, color="k", ls="--", lw=1, alpha=0.5,
                   label=f"chance ({base:.3f})")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC (RF candidate windows)", fontweight="bold")
    ax_roc.legend(loc="lower right", frameon=False); ax_roc.grid(color="#EEE")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR (RF candidate windows)", fontweight="bold")
    ax_pr.legend(loc="upper right", frameon=False); ax_pr.grid(color="#EEE")
    fig.suptitle(f"Learned combiner on RF candidate windows "
                  f"(strategy={args.candidate_strategy}, "
                  f"≤{args.max_candidates_per_gene} cands/gene)",
                  fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.out_dir, "figure_learned_combiner.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    sub = test_df.sample(min(20000, len(test_df)), random_state=42)
    l_rf = safe_logit(sub["rf_prob"].values)
    l_sp = safe_logit(sub["splicing_prob"].values)
    is_pos = sub["is_canonical_zone"].values == 1
    ax.scatter(l_rf[~is_pos], l_sp[~is_pos], s=4, alpha=0.3,
                color="#888888", label="outside canonical zone")
    ax.scatter(l_rf[is_pos], l_sp[is_pos], s=10, alpha=0.7,
                color="#C23B3B", label="canonical zone")
    rf_grid = np.linspace(l_rf.min(), l_rf.max(), 100)
    if abs(w_splice) > 1e-6:
        splice_boundary = -(w_intercept + w_rf * rf_grid) / w_splice
        ax.plot(rf_grid, splice_boundary, "k--", lw=2,
                 label="learned combiner decision boundary")
    ax.set_xlabel("logit(rf_prob)")
    ax.set_ylabel("logit(splicing_prob)")
    ax.set_title(f"Test candidate windows\n"
                  f"learned: {w_rf:.2f}·logit_rf + "
                  f"{w_splice:.2f}·logit_splice + {w_intercept:.2f}",
                  fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color="#EEE")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figure_logit_scatter.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[combiner] wrote:")
    print(f"  combiner_metrics.tsv")
    print(f"  combiner_coefficients.tsv")
    print(f"  val_candidates.tsv  test_candidates.tsv")
    print(f"  figure_learned_combiner.png  figure_logit_scatter.png")


if __name__ == "__main__":
    main()