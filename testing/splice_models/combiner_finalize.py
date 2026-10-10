#!/usr/bin/env python3
# two analyses for the paper-ready combiner section
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_and_report(name, X_train, y_train, X_test, y_test,
                     feature_names, class_weight=None):
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight,
                                solver="lbfgs")
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    ap = average_precision_score(y_test, p_test)
    print(f"\n  [{name}] class_weight={class_weight}")
    print(f"    intercept     = {float(clf.intercept_[0]):>+.4f}")
    for i, fn in enumerate(feature_names):
        print(f"    w_{fn:<14s}= {float(clf.coef_[0, i]):>+.4f}")
    print(f"    test AUROC = {auc:.4f}, AUPRC = {ap:.4f}")
    return clf, p_test, auc, ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_candidates", required=True)
    ap.add_argument("--test_candidates", required=True)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--model_joblib", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_shap_samples", type=int, default=5000,
                     help="Subsample candidates for SHAP (cost control)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    val_df = pd.read_csv(args.val_candidates, sep="\t")
    test_df = pd.read_csv(args.test_candidates, sep="\t")
    print(f"[final] val candidates: {len(val_df):,}  test candidates: {len(test_df):,}")
    print(f"[final] val pos rate: {val_df['is_canonical_zone'].mean():.4f}")
    print(f"[final] test pos rate: {test_df['is_canonical_zone'].mean():.4f}")

    X_val = np.column_stack([safe_logit(val_df["rf_prob"].values),
                                safe_logit(val_df["splicing_prob"].values)])
    y_val = val_df["is_canonical_zone"].values
    X_test = np.column_stack([safe_logit(test_df["rf_prob"].values),
                                 safe_logit(test_df["splicing_prob"].values)])
    y_test = test_df["is_canonical_zone"].values

    print(f"\n{'='*70}")
    print(f"ANALYSIS 1: combiner coefficients with and without class_weight")
    print(f"{'='*70}")

    feature_names = ["logit_rf", "logit_splice"]
    clf_bal, p_bal, auc_bal, ap_bal = fit_and_report(
        "WITH class_weight='balanced' (your earlier fit)",
        X_val, y_val, X_test, y_test, feature_names,
        class_weight="balanced")
    clf_unbal, p_unbal, auc_unbal, ap_unbal = fit_and_report(
        "WITHOUT class_weight (clean coefficients for paper)",
        X_val, y_val, X_test, y_test, feature_names,
        class_weight=None)

    print(f"\n  [sanity] AUROC identical between balanced and unbalanced LR? "
           f"{abs(auc_bal - auc_unbal) < 1e-6}")
    print(f"  [verdict] Use UNBALANCED coefficients for paper; ranking unchanged.")

    coef_summary = pd.DataFrame([
        {"variant": "with_balanced_class_weight",
         "intercept": float(clf_bal.intercept_[0]),
         "w_logit_rf": float(clf_bal.coef_[0, 0]),
         "w_logit_splice": float(clf_bal.coef_[0, 1]),
         "ratio_splice_over_rf": float(clf_bal.coef_[0, 1] / clf_bal.coef_[0, 0]),
         "test_auroc": float(auc_bal), "test_auprc": float(ap_bal)},
        {"variant": "no_class_weight_PAPER_COEFFICIENTS",
         "intercept": float(clf_unbal.intercept_[0]),
         "w_logit_rf": float(clf_unbal.coef_[0, 0]),
         "w_logit_splice": float(clf_unbal.coef_[0, 1]),
         "ratio_splice_over_rf": float(clf_unbal.coef_[0, 1] / clf_unbal.coef_[0, 0]),
         "test_auroc": float(auc_unbal), "test_auprc": float(ap_unbal)},
    ])
    coef_summary.to_csv(os.path.join(args.out_dir, "combiner_coefficients_clean.tsv"),
                          sep="\t", index=False)
    print(f"\n[final] wrote combiner_coefficients_clean.tsv")

    print(f"\n{'='*70}")
    print(f"ANALYSIS 2: SHAP on the splicing GBT (candidate-window subset)")
    print(f"{'='*70}")


    print(f"[final] loading features TSV...")
    feat_df = pd.read_csv(args.features_tsv, sep="\t")
    print(f"[final]   {len(feat_df):,} labeled sites total")

    print(f"[final] loading splicing GBT...")
    bundle = joblib.load(args.model_joblib)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    print(f"[final]   model has {len(feature_cols)} features")

    VAL_CHROMS = {"chr3", "chr13"}
    TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
    def norm_chr(c):
        c = str(c).strip()
        return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()
    feat_df["chrom_norm"] = feat_df["chrom"].apply(norm_chr)
    valtest_feat = feat_df[feat_df["chrom_norm"].isin(VAL_CHROMS | TEST_CHROMS)]
    print(f"[final]   val+test labeled sites: {len(valtest_feat):,}")
    print(f"[final]   labels: "
           f"{valtest_feat['y'].value_counts().to_dict()}")

    if len(valtest_feat) > args.n_shap_samples:
        shap_subset = valtest_feat.sample(args.n_shap_samples,
                                             random_state=42).reset_index(drop=True)
    else:
        shap_subset = valtest_feat.reset_index(drop=True)
    print(f"[final]   SHAP subset: {len(shap_subset):,} sites "
           f"({int((shap_subset['y']==1).sum()):,} canonical, "
           f"{int((shap_subset['y']==0).sum()):,} cryptic)")

    X_shap = shap_subset[feature_cols].values

    try:
        import shap
    except ImportError:
        print(f"[final] shap not installed. Try: pip install --user shap")
        return

    print(f"[final] computing SHAP values...")
    if hasattr(model, "named_steps"):
        gbt = model.named_steps["gbt"]
        imputer = model.named_steps["imp"]
        X_shap_imputed = imputer.transform(X_shap)
    else:
        gbt = model
        X_shap_imputed = X_shap

    try:
        explainer = shap.TreeExplainer(gbt)
        shap_values = explainer.shap_values(X_shap_imputed)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
    except Exception as e:
        print(f"[final] SHAP TreeExplainer failed: {e}")
        print(f"[final] falling back to permutation-style explainer")
        try:
            explainer = shap.Explainer(gbt, X_shap_imputed[:100])
            sv_obj = explainer(X_shap_imputed)
            sv = sv_obj.values
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        except Exception as e2:
            print(f"[final] also failed: {e2}")
            return

    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_summary = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap_canonical": np.where(
            (shap_subset["y"] == 1).values[:, None],
            sv, np.nan).mean(axis=0)
            if len(shap_subset[shap_subset["y"] == 1]) > 0 else np.nan,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    canon_mask = (shap_subset["y"] == 1).values
    cryptic_mask = (shap_subset["y"] == 0).values
    if canon_mask.sum() > 0:
        shap_summary["mean_shap_canonical"] = sv[canon_mask, :].mean(axis=0)
    if cryptic_mask.sum() > 0:
        shap_summary["mean_shap_cryptic"] = sv[cryptic_mask, :].mean(axis=0)
    shap_summary["signed_separation"] = (
        shap_summary["mean_shap_canonical"] - shap_summary["mean_shap_cryptic"])
    shap_summary = shap_summary.sort_values("mean_abs_shap", ascending=False)

    print(f"\n[final] Top 15 features by mean |SHAP|:")
    print(f"  {'feature':<25s} {'|SHAP|':>10s} {'canon_avg':>10s} "
           f"{'cryp_avg':>10s} {'separation':>11s}")
    for _, row in shap_summary.head(15).iterrows():
        print(f"  {row['feature']:<25s} {row['mean_abs_shap']:>10.4f} "
               f"{row.get('mean_shap_canonical', np.nan):>10.4f} "
               f"{row.get('mean_shap_cryptic', np.nan):>10.4f} "
               f"{row.get('signed_separation', np.nan):>11.4f}")

    shap_summary.to_csv(os.path.join(args.out_dir, "shap_summary_combiner_subset.tsv"),
                          sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(9, 9))
    top = shap_summary.head(20).iloc[::-1]
    colors = ["#C23B3B" if s > 0 else "#3E8C68" for s in top["signed_separation"]]
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors, alpha=0.8)
    ax.set_xlabel("mean |SHAP value| (canonical-vs-cryptic discrimination)")
    ax.set_title("Top 20 splicing features by SHAP importance\n"
                  f"(val+test labeled sites, n={len(shap_subset):,})",
                  fontweight="bold")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C23B3B", alpha=0.8,
              label="positively associated with canonical"),
        Patch(facecolor="#3E8C68", alpha=0.8,
              label="negatively associated (i.e., higher = more cryptic)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#EEE")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figure_shap_combiner_subset.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig)

    try:
        fig = plt.figure(figsize=(10, 9))
        shap.summary_plot(sv, X_shap_imputed, feature_names=feature_cols,
                           max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "figure_shap_beeswarm.png"),
                     dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[final] beeswarm plot failed: {e}")

    print(f"\n[final] wrote:")
    print(f"  combiner_coefficients_clean.tsv")
    print(f"  shap_summary_combiner_subset.tsv")
    print(f"  figure_shap_combiner_subset.png")
    print(f"  figure_shap_beeswarm.png (if shap available)")


if __name__ == "__main__":
    main()