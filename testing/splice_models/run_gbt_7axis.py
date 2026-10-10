#!/usr/bin/env python3

# fair evaluation + interpretation for the 7-feature mechanistic gbt

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import spearmanr


FEATURES_WG = [
    "upstream_mean_u1__wg",
    "upstream_max_me3__wg",
    "term_3ss__wg",
    "local_me5__wg",
    "pas_prox_u1__wg",
    "dn5ss_has_strong__wg",
    "interv_me5_max__wg",
]
FEATURES_RAW = [f.replace("__wg", "") for f in FEATURES_WG]

_DISPLAY_BY_RAW = {
    "upstream_mean_u1":  "Mean upstream U1 affinity",
    "upstream_max_me3":  "Best upstream 3′SS (MaxEntScan)",
    "term_3ss":          "Terminal 3′SS strength",
    "local_me5":         "Local 5′SS strength (MaxEntScan)",
    "pas_prox_u1":       "PAS-proximal U1 affinity",
    "dn5ss_has_strong":  "Strong downstream 5′SS present",
    "interv_me5_max":    "Intervening 5′SS in terminal exon",
}
FEAT_DISPLAY: Dict[str, str] = {}
for _raw, _disp in _DISPLAY_BY_RAW.items():
    FEAT_DISPLAY[_raw] = _disp
    FEAT_DISPLAY[_raw + "__wg"] = _disp


REAL_COLOR = "#2E8B57"
CRYPTIC_COLOR = "#C23B3B"


def tic(msg: str):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_gbt(seed: int, max_iter: int, max_depth: int, min_samples_leaf: int, learning_rate: float):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )),
    ])


def subsample_idx(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    k = min(int(k), n)
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def per_gene_prediction_metrics(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    dat = pd.DataFrame({"y": y, "pred": pred, "gene": groups})
    rows = []
    for gene, gdf in dat.groupby("gene", sort=False):
        yg = gdf["y"].values.astype(int)
        pg = gdf["pred"].values.astype(float)
        n_real = int(np.sum(yg == 1))
        n_cryptic = int(np.sum(yg == 0))
        if n_real == 0 or n_cryptic == 0:
            continue
        auc_g = float(roc_auc_score(yg, pg))
        real_mean = float(np.mean(pg[yg == 1]))
        cryptic_mean = float(np.mean(pg[yg == 0]))
        delta = real_mean - cryptic_mean
        if delta > 0:
            acc = 1.0
        elif delta < 0:
            acc = 0.0
        else:
            acc = 0.5
        if len(yg) >= 4:
            rho, _ = spearmanr(pg, yg)
            spearman_rho = float(rho) if np.isfinite(rho) else np.nan
        else:
            spearman_rho = np.nan
        top1_is_real = int(yg[np.argmax(pg)] == 1)
        real_scores = pg[yg == 1]
        best_real_score = float(np.max(real_scores))
        cryptic_scores = pg[yg == 0]
        n_above = int(np.sum(cryptic_scores > best_real_score))
        rank_of_real = 1 + n_above
        reciprocal_rank = 1.0 / float(rank_of_real)
        rows.append({
            "gene": str(gene),
            "within_gene_auc": auc_g,
            "real_mean_pred": real_mean,
            "cryptic_mean_pred": cryptic_mean,
            "real_minus_cryptic_pred": delta,
            "within_gene_acc": acc,
            "within_gene_spearman": spearman_rho,
            "top1_is_real": top1_is_real,
            "rank_of_real": rank_of_real,
            "reciprocal_rank": reciprocal_rank,
            "n_real": n_real,
            "n_cryptic": n_cryptic,
            "n_sites": len(yg),
        })
    return pd.DataFrame(rows)


def shap_direction_stats(x: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(x) & np.isfinite(s)
    x = x[ok]; s = s[ok]
    if len(x) == 0:
        return {"mean_abs_shap": np.nan, "mean_shap": np.nan,
                "mean_shap_top10": np.nan, "mean_shap_bottom10": np.nan,
                "delta_top_bottom_shap": np.nan, "signed_importance": np.nan}
    q10 = np.quantile(x, 0.10)
    q90 = np.quantile(x, 0.90)
    bot = s[x <= q10]
    top = s[x >= q90]
    mean_abs = float(np.mean(np.abs(s)))
    mean_s = float(np.mean(s))
    mean_top = float(np.mean(top)) if len(top) else np.nan
    mean_bot = float(np.mean(bot)) if len(bot) else np.nan
    delta = float(mean_top - mean_bot) if np.isfinite(mean_top) and np.isfinite(mean_bot) else np.nan
    if not np.isfinite(delta) or abs(delta) < 1e-12:
        signed_imp = 0.0
    else:
        signed_imp = float(np.sign(delta) * mean_abs)
    return {"mean_abs_shap": mean_abs, "mean_shap": mean_s,
            "mean_shap_top10": mean_top, "mean_shap_bottom10": mean_bot,
            "delta_top_bottom_shap": delta, "signed_importance": signed_imp}


def summarize_feature_direction_rows(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out_rows = []
    for feat, sub in df.groupby("feature", sort=False):
        out_rows.append({
            "feature": feat,
            "display": FEAT_DISPLAY.get(feat, feat),
            "mean_abs_shap_mean": float(sub["mean_abs_shap"].mean()),
            "mean_abs_shap_std": float(sub["mean_abs_shap"].std(ddof=0)),
            "mean_abs_shap_ci_lo": float(sub["mean_abs_shap"].quantile(0.025)),
            "mean_abs_shap_ci_hi": float(sub["mean_abs_shap"].quantile(0.975)),
            "delta_top_bottom_shap_mean": float(sub["delta_top_bottom_shap"].mean()),
            "delta_top_bottom_shap_std": float(sub["delta_top_bottom_shap"].std(ddof=0)),
            "delta_top_bottom_shap_ci_lo": float(sub["delta_top_bottom_shap"].quantile(0.025)),
            "delta_top_bottom_shap_ci_hi": float(sub["delta_top_bottom_shap"].quantile(0.975)),
            "signed_importance_mean": float(sub["signed_importance"].mean()),
            "signed_importance_std": float(sub["signed_importance"].std(ddof=0)),
            "signed_importance_ci_lo": float(sub["signed_importance"].quantile(0.025)),
            "signed_importance_ci_hi": float(sub["signed_importance"].quantile(0.975)),
            "direction_label": (
                "constitutive/active" if float(sub["signed_importance"].mean()) > 0
                else "cryptic" if float(sub["signed_importance"].mean()) < 0
                else "weak/mixed"
            ),
            "n_folds": int(sub["fold"].nunique()),
        })
    out = pd.DataFrame(out_rows).sort_values("mean_abs_shap_mean", ascending=False).reset_index(drop=True)
    return out


def plot_signed_importance(direction_df, out_png, p_at_1, mrr, site_auroc, site_auprc, mode):
    feat_order = direction_df["feature"].tolist()
    labels = direction_df["display"].tolist()
    vals = direction_df["signed_importance_mean"].values
    lo = direction_df["signed_importance_ci_lo"].values
    hi = direction_df["signed_importance_ci_hi"].values
    fig, ax = plt.subplots(figsize=(14, 5.8))
    y = np.arange(len(feat_order))
    for i, v in enumerate(vals):
        color = REAL_COLOR if v > 0 else CRYPTIC_COLOR if v < 0 else "#888888"
        ax.barh(i, v, color=color, alpha=0.88, edgecolor="white", linewidth=0.8, zorder=2)
        ax.errorbar(v, i, xerr=[[v - lo[i]], [hi[i] - v]],
                    fmt="none", ecolor="#333", elinewidth=1.1, capsize=3, zorder=3)
        txt = f"{v:+.3f}"
        if abs(v) >= 0.08:
            ax.text(v / 2, i, txt, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=4)
        else:
            dx = 0.008 if v >= 0 else -0.008
            ha = "left" if v >= 0 else "right"
            ax.text(v + dx, i, txt, ha=ha, va="center",
                    fontsize=8.5, fontweight="bold", color="#333", zorder=4)
    ax.axvline(0, color="#A8A8A8", linewidth=1.0, zorder=1)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_fontweight("bold")
        tick.set_color(REAL_COLOR if vals[i] > 0 else CRYPTIC_COLOR if vals[i] < 0 else "#555555")
    ax.set_xlabel("Signed mean |SHAP|  (positive = high values favor constitutive/active)", fontsize=11)
    ax.set_title(
        f"{mode.upper()} GBT signed feature effects  "
        f"(OOF AUROC={site_auroc:.3f}, AUPRC={site_auprc:.3f}, MRR={mrr:.3f}, P@1={p_at_1:.3f})",
        fontsize=13, fontweight="bold", loc="left"
    )
    ax.text(ax.get_xlim()[0], -0.85, "← high values favor cryptic", color=CRYPTIC_COLOR,
            fontsize=10.5, fontweight="bold", ha="left")
    ax.text(ax.get_xlim()[1], -0.85, "high values favor constitutive/active →", color=REAL_COLOR,
            fontsize=10.5, fontweight="bold", ha="right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_auprc_curve(y_true, y_score, out_png, out_tsv, title_prefix="Grouped OOF precision–recall curve"):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    ok = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[ok]; y_score = y_score[ok]
    if len(np.unique(y_true)) < 2:
        print("WARNING: cannot draw PR curve because only one class is present.", flush=True)
        return
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))
    baseline = float(np.mean(y_true))
    thr_full = np.full(len(precision), np.nan, dtype=float)
    if len(thresholds) > 0:
        thr_full[1:] = thresholds
    pr_df = pd.DataFrame({"recall": recall, "precision": precision, "threshold": thr_full})
    pr_df.to_csv(out_tsv, sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(recall, precision, linewidth=2.2, color="#2E6FBB", label=f"OOF PR curve (AP={ap:.3f})")
    ax.axhline(baseline, color="#999999", linestyle="--", linewidth=1.2,
               label=f"Positive-rate baseline ({baseline:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title_prefix, fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("wg", "raw"), default="wg",
                    help="Which 7-axis panel to train: wg = __wg-centered "
                         "(reproduces interpret_gbt.py); raw = non-centered "
                         "(reproduces 01_run_raw_gbt.py).")
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=180)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--min_samples_leaf", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--no_shap", action="store_true")
    ap.add_argument("--shap_bg", type=int, default=300)
    ap.add_argument("--shap_eval_per_fold", type=int, default=2500)
    ap.add_argument("--shap_summary_n", type=int, default=2000)
    ap.add_argument("--skip_add_minimal", action="store_true",
                    help="Skip minimal_splice_lr.add_minimal_features. Use "
                         "when the input TSV already has the target axes "
                         "(e.g. per-cell features.minimal.tsv from the "
                         "window-aware benchmark, where the window-blind "
                         "add_minimal_features would clobber correct "
                         "sources with bin-fallback). If the columns are "
                         "actually missing, this raises rather than "
                         "silently falling back.")
    ap.add_argument("--features_override", default=None,
                    help="Optional CSV of exactly 7 column names to use "
                         "INSTEAD of the default FEATURES_WG/FEATURES_RAW "
                         "panel for the chosen --mode. Strictly additive: "
                         "if unset, the script behaves byte-equivalent to "
                         "the no-override path. MODEL_NAME gains an "
                         "'_ablate' suffix when this flag is in use so "
                         "downstream tooling can detect ablation runs.")
    ap.add_argument("--display_override", default=None,
                    help="Optional CSV of display names, same length as "
                         "--features_override. If omitted, falls back to "
                         "FEAT_DISPLAY[colname] or the colname itself.")
    args = ap.parse_args()

    FEATURES = FEATURES_WG if args.mode == "wg" else FEATURES_RAW
    MODEL_NAME = "mech_axes7_wg_gbt" if args.mode == "wg" else "mech_axes7_raw_gbt"

    if args.features_override:
        override_cols = [s.strip() for s in args.features_override.split(",") if s.strip()]
        if len(override_cols) != 7:
            raise SystemExit(
                f"--features_override must have exactly 7 entries; "
                f"got {len(override_cols)}: {override_cols}"
            )
        FEATURES = override_cols
        MODEL_NAME = MODEL_NAME + "_ablate"
        if args.display_override:
            override_disps = [s.strip() for s in args.display_override.split(",")]
            if len(override_disps) != 7:
                raise SystemExit(
                    f"--display_override must match --features_override "
                    f"length (7); got {len(override_disps)}"
                )
            for col, disp in zip(override_cols, override_disps):
                FEAT_DISPLAY[col] = disp
        print(f"[override] using --features_override panel ({len(FEATURES)} cols): "
              f"{FEATURES}", flush=True)

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tic("Loading features")
    df = pd.read_csv(args.features_tsv, sep="\t")

    gene_col = "gene" if "gene" in df.columns else "gene_name"
    label_col = "y" if "y" in df.columns else "label"

    if args.features_override:
        for col in FEATURES:
            if col not in df.columns:
                print(f"[override] {col}: not in TSV (will error at the missing-col guard)",
                      flush=True)
                continue
            vals = df[col].astype(float).values
            overall_std = float(np.nanstd(vals))
            try:
                per_gene_std = df.groupby(gene_col, sort=False)[col].std()
                within_std = float(per_gene_std.mean(skipna=True))
            except Exception:
                within_std = float("nan")
            vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else float("nan")
            vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else float("nan")
            print(f"[override] {col}: range=[{vmin:.4g}, {vmax:.4g}] "
                  f"overall_std={overall_std:.4g} within_gene_std={within_std:.4g}",
                  flush=True)
            if overall_std > 0 and np.isfinite(within_std) and within_std / overall_std < 0.01:
                print(f"[override]   WARNING: within-gene std is < 1% of overall std — "
                      f"feature is near-constant per gene; experiment may be uninformative.",
                      flush=True)

    already_have = all(c in df.columns for c in FEATURES)
    if args.skip_add_minimal or already_have:
        if not already_have:
            raise SystemExit(
                f"--skip_add_minimal given but missing columns: "
                f"{[c for c in FEATURES if c not in df.columns]}"
            )
        print(
            f"Using FEATURES from TSV as-is "
            f"({'flag' if args.skip_add_minimal else 'auto-detected'}).",
            flush=True,
        )
    else:
        from minimal_splice_lr import add_minimal_features
        df = add_minimal_features(df)
        if args.mode == "wg":
            grp = df.groupby(gene_col, sort=False)
            for f in FEATURES_WG:
                raw_col = f.replace("__wg", "")
                if f not in df.columns and raw_col in df.columns:
                    df[f] = df[raw_col] - grp[raw_col].transform("mean")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing required {args.mode.upper()} features after "
            f"derivation: {missing}. Refuse to silently fall back."
        )

    features = FEATURES
    X = df[features].values
    y = df[label_col].values.astype(int)
    groups = df[gene_col].astype(str).values

    print(f"Sites: {len(y)} ({y.sum()} constitutive/active, {(y == 0).sum()} cryptic)", flush=True)
    print(f"Genes: {len(np.unique(groups))}", flush=True)
    print(f"Features ({args.mode.upper()}): {features}", flush=True)

    tic("Grouped OOF evaluation")
    gkf = GroupKFold(n_splits=args.n_splits)
    oof = np.full(len(y), np.nan)
    feature_direction_rows = []

    has_shap = False
    if not args.no_shap:
        try:
            import shap
            has_shap = True
        except Exception:
            print("WARNING: shap not installed, skipping SHAP-based direction outputs", flush=True)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"  fold {fold}/{args.n_splits}  train={len(tr)} test={len(te)}", flush=True)
        m = make_gbt(args.seed + fold,
                     max_iter=args.max_iter, max_depth=args.max_depth,
                     min_samples_leaf=args.min_samples_leaf,
                     learning_rate=args.learning_rate)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

        if has_shap:
            import shap
            clf_fold = m.named_steps["clf"]
            imp_fold = m.named_steps["imputer"]
            bg_local = subsample_idx(len(tr), args.shap_bg, rng)
            eval_local = subsample_idx(len(te), args.shap_eval_per_fold, rng)
            bg_idx = tr[bg_local]; ev_idx = te[eval_local]
            X_bg = imp_fold.transform(X[bg_idx])
            X_ev = imp_fold.transform(X[ev_idx])
            explainer = shap.TreeExplainer(clf_fold, data=X_bg, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_ev, check_additivity=False)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
            for j, feat in enumerate(features):
                stats = shap_direction_stats(X_ev[:, j], shap_vals[:, j])
                row = {"fold": fold, "feature": feat}
                row.update(stats)
                feature_direction_rows.append(row)

    tic("Computing fair grouped OOF metrics")
    site_auroc = float(roc_auc_score(y, oof))
    site_auprc = float(average_precision_score(y, oof))
    gene_df = per_gene_prediction_metrics(y, oof, groups)
    within_gene_auc = float(gene_df["within_gene_auc"].mean())
    within_gene_acc = float(gene_df["within_gene_acc"].mean())
    within_gene_spearman = float(gene_df["within_gene_spearman"].dropna().mean())
    top1_acc = float(gene_df["top1_is_real"].mean())
    mrr = float(gene_df["reciprocal_rank"].mean())
    mean_rank = float(gene_df["rank_of_real"].mean())

    summary_df = pd.DataFrame([{
        "model": MODEL_NAME,
        "n_features": len(features),
        "n_sites": len(y),
        "n_genes": len(np.unique(groups)),
        "site_auroc_oof": site_auroc,
        "site_auprc_oof": site_auprc,
        "within_gene_auc": within_gene_auc,
        "within_gene_acc": within_gene_acc,
        "within_gene_spearman": within_gene_spearman,
        "top1_acc": top1_acc,
        "mrr": mrr,
        "mean_rank_of_real": mean_rank,
        "n_paired_genes": len(gene_df),
    }])
    summary_df.to_csv(str(out) + ".summary.tsv", sep="\t", index=False)

    oof_df = df[[label_col, gene_col]].copy()
    if "id" in df.columns:
        oof_df["id"] = df["id"].astype(str)
    else:
        oof_df["id"] = np.arange(len(df)).astype(str)
    oof_df["p"] = oof
    oof_df.to_csv(str(out) + ".oof.tsv", sep="\t", index=False)

    plot_auprc_curve(y_true=y, y_score=oof,
                     out_png=str(out) + ".auprc_curve.png",
                     out_tsv=str(out) + ".auprc_curve.tsv",
                     title_prefix=f"Grouped OOF precision–recall curve  (AP={site_auprc:.3f})")
    gene_df.to_csv(str(out) + ".gene_metrics.tsv", sep="\t", index=False)

    print("\nFair grouped OOF metrics:", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    if has_shap and len(feature_direction_rows) > 0:
        tic("Summarizing fold-wise SHAP directions")
        direction_df = summarize_feature_direction_rows(feature_direction_rows)
        direction_df.to_csv(str(out) + ".feature_directions.tsv", sep="\t", index=False)
        plot_signed_importance(direction_df=direction_df,
                               out_png=str(out) + ".signed_importance.png",
                               p_at_1=top1_acc, mrr=mrr,
                               site_auroc=site_auroc, site_auprc=site_auprc,
                               mode=args.mode)

    if has_shap:
        tic("Fitting full model for SHAP summary/dependence plots")
        import shap
        full_model = make_gbt(args.seed,
                              max_iter=args.max_iter, max_depth=args.max_depth,
                              min_samples_leaf=args.min_samples_leaf,
                              learning_rate=args.learning_rate)
        full_model.fit(X, y)
        X_imp = full_model.named_steps["imputer"].transform(X)
        clf = full_model.named_steps["clf"]
        bg_idx = subsample_idx(len(X_imp), args.shap_bg, rng)
        ex_idx = subsample_idx(len(X_imp), args.shap_summary_n, rng)
        explainer = shap.TreeExplainer(clf, data=X_imp[bg_idx], feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_imp[ex_idx], check_additivity=False)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        tic("Saving SHAP summary plot")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_imp[ex_idx],
                          feature_names=[FEAT_DISPLAY.get(f, f) for f in features],
                          show=False, max_display=len(features))
        plt.title(f"SHAP summary ({args.mode.upper()}): what drives GBT predictions?",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(out) + ".shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close("all")

        tic("Saving SHAP dependence plots")
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        axes = axes.flatten()
        for i, feat in enumerate(features):
            shap.dependence_plot(i, shap_vals, X_imp[ex_idx],
                                 feature_names=[FEAT_DISPLAY.get(f, f) for f in features],
                                 ax=axes[i], show=False)
            axes[i].set_title(FEAT_DISPLAY.get(feat, feat), fontsize=10, fontweight="bold")
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"SHAP dependence ({args.mode.upper()}): feature value → SHAP contribution",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(str(out) + ".shap_dependence.png", dpi=300, bbox_inches="tight")
        plt.close("all")

    print(f"\n[OK] wrote {out}.summary.tsv", flush=True)


if __name__ == "__main__":
    main()