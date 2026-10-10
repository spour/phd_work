#!/usr/bin/env python3
# produces all 5 aim 1 figures + supporting tsvs for the splicing gbt paper
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


EXPECTED_AXES = {
    "term_3ss__wg":           ("+", "strong terminal 3'SS = real terminal exon"),
    "upstream_max_me3__wg":   ("+", "strong upstream 3'SS background = real terminal exon"),
    "upstream_mean_u1__wg":   ("-", "high U1 vs gene mean = intronic context (5'SS-rich)"),
    "pas_prox_u1__wg":        ("-", "strong U1 at PAS = U1 competes with cleavage"),
    "local_me5__wg":          ("?", "PAS-proximal 5'SS — direction ambiguous a priori"),
    "dn5ss_has_strong__wg":   ("-", "strong downstream 5'SS = intron continues past PAS"),
    "interv_me5_max__wg":     ("-", "intervening 5'SS in putative terminal exon"),
}


def expected_dir_bin(bin_idx, motif, stat, n_bins=8, upstream_nt=3000, downstream_nt=1000):
    total = upstream_nt + downstream_nt
    bin_width = total // n_bins
    n_upstream = upstream_nt // bin_width
    is_upstream = bin_idx <= n_upstream

    if motif == "me3":
        if stat == "max" and is_upstream:
            return ("+", "max upstream 3'SS = sharp terminal exon signal")
        return ("?", "")
    elif motif == "me5":
        if not is_upstream:
            return ("-", "downstream 5'SS = intron continues past PAS")
        return ("?", "")
    elif motif == "u1":
        if is_upstream:
            return ("+", "raw upstream U1 = gene-level splicing competence")
        return ("?", "")
    return ("?", "")


def derive_per_gene_metrics(oof_preds, df):
    df = df.copy()
    df["pred"] = oof_preds
    rows = []
    for gene, g in df.groupby("gene", sort=False):
        y = g["y"].values.astype(int)
        p = g["pred"].values.astype(float)
        if not (y == 1).any() or not (y == 0).any():
            continue
        try:
            wgauc = float(roc_auc_score(y, p))
        except ValueError:
            wgauc = np.nan
        top1 = int(y[np.argmax(p)] == 1)
        canonical_score = float(np.max(p[y == 1]))
        rank = 1 + int((p[y == 0] > canonical_score).sum())
        rows.append({
            "gene": gene, "n_sites": int(len(y)),
            "n_cryptic": int((y == 0).sum()),
            "within_gene_auc": wgauc, "top1_is_real": top1,
            "reciprocal_rank": 1.0 / rank, "rank_of_real": rank,
        })
    return pd.DataFrame(rows)


def fit_oof_gbt(df, feature_cols, y, groups):
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
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--model_joblib", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_background", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[finalize] loading model + features")
    bundle = joblib.load(args.model_joblib)
    feature_cols = bundle["feature_cols"]
    bin_cols = bundle["feature_groups"]["per_bin"]
    wg_cols = bundle["feature_groups"]["gene_baselined_axes"]
    oof_combined = bundle["oof_predictions"]

    df = pd.read_csv(args.features_tsv, sep="\t")
    y = df["y"].values.astype(int)
    groups = df["gene"].astype(str).values

    print(f"\n{'='*72}\n[PART 1] GROUP ABLATION\n{'='*72}")
    ablation_rows = [{
        "config": "combined_48_per_bin_+_7_axes_baselined",
        "n_features": len(feature_cols),
        "auroc": float(roc_auc_score(y, oof_combined)),
        "auprc": float(average_precision_score(y, oof_combined)),
    }]

    print(f"  fitting per-bin only (48 raw)")
    oof_pb = fit_oof_gbt(df, bin_cols, y, groups)
    ablation_rows.append({"config": "per_bin_only_48", "n_features": len(bin_cols),
                          "auroc": float(roc_auc_score(y, oof_pb)),
                          "auprc": float(average_precision_score(y, oof_pb))})
    print(f"    AUROC={ablation_rows[-1]['auroc']:.4f} AUPRC={ablation_rows[-1]['auprc']:.4f}")

    print(f"  fitting axes only (7 gene-baselined)")
    oof_ax = fit_oof_gbt(df, wg_cols, y, groups)
    ablation_rows.append({"config": "axes_only_7", "n_features": len(wg_cols),
                          "auroc": float(roc_auc_score(y, oof_ax)),
                          "auprc": float(average_precision_score(y, oof_ax))})
    print(f"    AUROC={ablation_rows[-1]['auroc']:.4f} AUPRC={ablation_rows[-1]['auprc']:.4f}")

    print(f"  leave-one-axis-out from combined")
    for drop in wg_cols:
        kept = bin_cols + [c for c in wg_cols if c != drop]
        oof_k = fit_oof_gbt(df, kept, y, groups)
        ablation_rows.append({
            "config": f"drop_{drop}", "n_features": len(kept),
            "auroc": float(roc_auc_score(y, oof_k)),
            "auprc": float(average_precision_score(y, oof_k)),
        })
        print(f"    drop {drop}: AUROC={ablation_rows[-1]['auroc']:.4f}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(os.path.join(args.out_dir, "group_ablation_results.tsv"),
                        sep="\t", index=False)

    print(f"\n{'='*72}\n[PART 2] SHAP RERUN (corrected expectations)\n{'='*72}")
    shap_df = None; shap_vals = None; X_ev = None; n_yes = 0; n_judged = 0
    try:
        import shap
    except ImportError:
        print(f"  shap not installed, skipping SHAP analysis")
        shap = None

    if shap is not None:
        rng = np.random.default_rng(args.seed)
        bg_idx = rng.choice(len(df), min(args.n_background, len(df)), replace=False)
        ev_idx = rng.choice(len(df), min(args.n_eval, len(df)), replace=False)

        model_pipe = bundle["model"]
        imputer = model_pipe.named_steps["imp"]
        clf = model_pipe.named_steps["gbt"]
        X_bg = imputer.transform(df[feature_cols].values[bg_idx])
        X_ev = imputer.transform(df[feature_cols].values[ev_idx])

        print(f"  fitting TreeExplainer (interventional, n_bg={len(bg_idx)})")
        explainer = shap.TreeExplainer(clf, data=X_bg,
                                         feature_perturbation="interventional")
        print(f"  computing SHAP values on n={len(ev_idx)} eval samples")
        sv = explainer.shap_values(X_ev, check_additivity=False)
        shap_vals = sv[1] if isinstance(sv, list) else sv

        shap_rows = []
        for j, name in enumerate(feature_cols):
            x = X_ev[:, j]; s = shap_vals[:, j]
            ok = np.isfinite(x) & np.isfinite(s)
            if ok.sum() < 100:
                continue
            x = x[ok]; s = s[ok]
            q10, q90 = np.quantile(x, [0.10, 0.90])
            delta = float(np.mean(s[x >= q90]) - np.mean(s[x <= q10]))
            mean_abs = float(np.mean(np.abs(s)))
            obs_dir = "+" if delta > 0 else "-" if delta < 0 else "?"

            if name in EXPECTED_AXES:
                exp_dir, reason = EXPECTED_AXES[name]; kind = "axis"
            elif name.startswith("bin"):
                try:
                    parts = name[3:].split("_")
                    bin_idx = int(parts[0])
                    motif = parts[1]; stat = parts[2]
                    exp_dir, reason = expected_dir_bin(bin_idx, motif, stat)
                    kind = "bin"
                except (ValueError, IndexError):
                    exp_dir, reason, kind = "?", "", "bin"
            else:
                exp_dir, reason, kind = "?", "unmapped", "other"

            agrees = (exp_dir == obs_dir) if exp_dir in ("+", "-") else None
            shap_rows.append({
                "feature": name, "kind": kind, "mean_abs_shap": mean_abs,
                "signed_importance": float(np.sign(delta) * mean_abs),
                "observed_direction": obs_dir, "expected_direction": exp_dir,
                "agrees": agrees, "reason": reason,
            })

        shap_df = pd.DataFrame(shap_rows).sort_values(
            "mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_df.to_csv(os.path.join(args.out_dir,
                                     "shap_feature_directions_corrected.tsv"),
                        sep="\t", index=False)

        judged = shap_df[shap_df["expected_direction"].isin(["+", "-"])]
        n_yes = int((judged["agrees"] == True).sum())
        n_judged = len(judged)
        print(f"  direction agreement: {n_yes}/{n_judged} "
               f"({100*n_yes/max(n_judged,1):.1f}%)")

    print(f"\n{'='*72}\n[PART 3] PER-GENE METRICS + STRATIFICATION\n{'='*72}")
    gm = derive_per_gene_metrics(oof_combined, df)
    gm.to_csv(os.path.join(args.out_dir, "per_gene_metrics_combined.tsv"),
              sep="\t", index=False)
    print(f"  paired genes: {len(gm):,}")
    print(f"  overall: P@1={gm['top1_is_real'].mean():.4f} "
           f"MRR={gm['reciprocal_rank'].mean():.4f} "
           f"WG-AUC={gm['within_gene_auc'].mean():.4f}")

    nc = gm["n_cryptic"].values
    q1, q2, q3 = np.quantile(nc, [0.25, 0.5, 0.75])
    edges = [1, int(np.ceil(q1)), int(np.ceil(q2)), int(np.ceil(q3)), int(nc.max() + 1)]
    edges = sorted(set([e for e in edges if e >= 1]))
    print(f"  cryptic-count quartile edges: {edges}")

    gm["cryptic_bin"] = np.digitize(gm["n_cryptic"], edges[1:], right=False)
    strat_rows = []
    for b in sorted(gm["cryptic_bin"].unique()):
        sub = gm[gm["cryptic_bin"] == b]
        lo = edges[b] if b < len(edges) else edges[-1]
        hi = (edges[b + 1] - 1) if (b + 1) < len(edges) else int(nc.max())
        bin_label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        strat_rows.append({
            "cryptic_bin": int(b), "bin_label": bin_label, "n_genes": len(sub),
            "mean_n_cryptic": float(sub["n_cryptic"].mean()),
            "p_at_1": float(sub["top1_is_real"].mean()),
            "mrr": float(sub["reciprocal_rank"].mean()),
            "within_gene_auc": float(sub["within_gene_auc"].mean()),
        })
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(os.path.join(args.out_dir,
                                   "stratified_by_cryptic_count.tsv"),
                      sep="\t", index=False)
    print(strat_df.to_string(index=False))

    print(f"\n{'='*72}\n[PART 4] RENDERING FIGURES\n{'='*72}")
    plt.rcParams.update({"font.size": 10, "axes.spines.right": False,
                          "axes.spines.top": False})

    approaches = [
        ("LR (best)",              0.706, np.nan,  0.395, 0.604),
        ("GBT 7-axis raw",          0.723, 0.484,  0.420, 0.625),
        ("GBT 48 per-bin raw",      0.760, 0.540,  np.nan, np.nan),
        ("GBT 7-axis gene-base",    0.747, 0.584,  0.431, 0.631),
        ("GBT 55 combined",         0.821, 0.667,  0.523, 0.697),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    metric_names = ["AUROC", "AUPRC", "P@1", "MRR"]
    colors = ["#888888", "#A6BFD9", "#7AAACE", "#3E8C68", "#2E5A88"]
    for ax, met, mc in zip(axes, metric_names, [1, 2, 3, 4]):
        vals = [a[mc] for a in approaches]
        labels = [a[0] for a in approaches]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.9,
                       edgecolor="white", linewidth=1.2)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_title(met, fontweight="bold", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.7)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Splicing GBT — approach comparison "
                  "(u3000_d1000, 5-fold GroupKFold by gene)",
                  fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(args.out_dir, "figure1_approach_comparison.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure1_approach_comparison.png")

    if shap_df is not None:
        top = shap_df.head(20).iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        bar_colors = ["#2E8B57" if v > 0 else "#C23B3B"
                       for v in top["signed_importance"]]
        yp = np.arange(len(top))
        ax.barh(yp, top["signed_importance"], color=bar_colors, alpha=0.9,
                 edgecolor="white", linewidth=1.0)
        ax.axvline(0, color="#333", linewidth=1.0)
        ax.set_yticks(yp)
        labels = []
        for _, r in top.iterrows():
            mark = f"[{r['kind'][0].upper()}]"
            agree_glyph = ("✓" if r["agrees"] is True
                            else "✗" if r["agrees"] is False else "·")
            labels.append(f"{r['feature']}  {mark} {agree_glyph}")
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Signed mean |SHAP|  (+ favors CANONICAL, "
                       "- favors 3-WAY intragenic)", fontsize=11)
        agree_pct = 100 * n_yes / max(n_judged, 1)
        ax.set_title(f"SHAP signed importance — top 20 features\n"
                      f"[A]xis vs [B]in   ✓ = direction matches biological "
                      f"expectation  ({n_yes}/{n_judged} = {agree_pct:.0f}%)",
                      fontweight="bold", fontsize=11, loc="left")
        ax.grid(axis="x", color="#EEEEEE", linewidth=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir,
                                  "figure2_shap_signed_importance.png"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ figure2_shap_signed_importance.png")

        focus = list(shap_df.head(6)["feature"]) + [
            "local_me5__wg", "bin7_me5_mean", "bin8_u1_mean"]
        focus = [f for f in focus if f in feature_cols][:9]
        fig, axg = plt.subplots(3, 3, figsize=(15, 12))
        axg = axg.flatten()
        for i, feat in enumerate(focus):
            j = feature_cols.index(feat)
            shap.dependence_plot(j, shap_vals, X_ev,
                                  feature_names=feature_cols,
                                  ax=axg[i], show=False)
            axg[i].set_title(feat, fontsize=10, fontweight="bold")
        for i in range(len(focus), 9):
            axg[i].axis("off")
        fig.suptitle("SHAP dependence — top features + puzzling cases",
                      fontweight="bold", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(args.out_dir, "figure3_shap_dependence.png"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ figure3_shap_dependence.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    main_cfgs = {"per_bin_only_48": "48 per-bin\n(raw)",
                  "axes_only_7": "7 axes\n(gene-baseline)",
                  "combined_48_per_bin_+_7_axes_baselined": "55 combined"}
    main_comp = ablation_df[ablation_df["config"].isin(main_cfgs.keys())].copy()
    main_comp["label"] = main_comp["config"].map(main_cfgs)
    main_comp = main_comp.sort_values("auroc").reset_index(drop=True)

    width = 0.35
    x = np.arange(len(main_comp))
    bars1 = ax1.bar(x - width/2, main_comp["auroc"], width, label="AUROC",
                     color="#3E8C68", alpha=0.9)
    bars2 = ax1.bar(x + width/2, main_comp["auprc"], width, label="AUPRC",
                     color="#7AAACE", alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(main_comp["label"], fontsize=9)
    for i in range(len(main_comp)):
        ax1.text(i - width/2, main_comp.iloc[i]["auroc"] + 0.012,
                  f"{main_comp.iloc[i]['auroc']:.3f}", ha="center", fontsize=8.5)
        ax1.text(i + width/2, main_comp.iloc[i]["auprc"] + 0.012,
                  f"{main_comp.iloc[i]['auprc']:.3f}", ha="center", fontsize=8.5)
    ax1.set_ylim(0, 1); ax1.set_ylabel("Score")
    ax1.set_title("(a) Feature group comparison", fontweight="bold", loc="left")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(axis="y", color="#EEEEEE", linewidth=0.7)

    base = ablation_df[ablation_df["config"].str.startswith("combined")].iloc[0]["auroc"]
    drops = ablation_df[ablation_df["config"].str.startswith("drop_")].copy()
    drops["axis"] = drops["config"].str.replace("drop_", "", regex=False
                                                  ).str.replace("__wg", "", regex=False)
    drops["delta"] = base - drops["auroc"]
    drops = drops.sort_values("delta", ascending=True).reset_index(drop=True)

    ax2.barh(range(len(drops)), drops["delta"], color="#C23B3B", alpha=0.9,
              edgecolor="white", linewidth=1.0)
    ax2.set_yticks(range(len(drops))); ax2.set_yticklabels(drops["axis"], fontsize=9)
    ax2.axvline(0, color="#333", linewidth=1.0)
    for i, v in enumerate(drops["delta"].values):
        ax2.text(v + max(drops["delta"].max() * 0.02, 0.0005), i,
                  f"{v:+.4f}", va="center", fontsize=8.5)
    ax2.set_xlabel(f"ΔAUROC vs combined ({base:.4f})")
    ax2.set_title("(b) Drop-one-axis from combined",
                   fontweight="bold", loc="left")
    ax2.grid(axis="x", color="#EEEEEE", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figure4_group_ablation.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure4_group_ablation.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs = strat_df["cryptic_bin"].values
    labels = strat_df["bin_label"].values
    ax1.plot(xs, strat_df["p_at_1"], "o-", color="#2E5A88",
              linewidth=2.5, markersize=9, label="P@1")
    ax1.plot(xs, strat_df["mrr"], "s-", color="#3E8C68",
              linewidth=2.5, markersize=9, label="MRR")
    ax1.plot(xs, strat_df["within_gene_auc"], "^-", color="#C68C2F",
              linewidth=2.5, markersize=9, label="Within-gene AUC")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_xlabel("Number of cryptic sites in gene")
    ax1.set_ylabel("Score")
    ax1.set_title("(a) Within-gene ranking by cryptic count",
                   fontweight="bold", loc="left")
    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(color="#EEEEEE", linewidth=0.7); ax1.set_ylim(0, 1)

    ax2.bar(xs, strat_df["n_genes"], color="#7AAACE", alpha=0.9,
             edgecolor="white", linewidth=1.0)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("Number of cryptic sites in gene")
    ax2.set_ylabel("Number of paired genes")
    ax2.set_title("(b) Sample size per stratum",
                   fontweight="bold", loc="left")
    ax2.grid(axis="y", color="#EEEEEE", linewidth=0.7)
    for i, v in enumerate(strat_df["n_genes"].values):
        ax2.text(xs[i], v + max(strat_df["n_genes"].max() * 0.02, 30),
                  f"{int(v):,}", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir,
                              "figure5_within_gene_stratified.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure5_within_gene_stratified.png")

    summary_path = os.path.join(args.out_dir, "headline_numbers.tsv")
    headline = pd.DataFrame([
        {"metric": "site_auroc",       "value": float(roc_auc_score(y, oof_combined))},
        {"metric": "site_auprc",       "value": float(average_precision_score(y, oof_combined))},
        {"metric": "p_at_1",           "value": float(gm["top1_is_real"].mean())},
        {"metric": "mrr",              "value": float(gm["reciprocal_rank"].mean())},
        {"metric": "within_gene_auc",  "value": float(gm["within_gene_auc"].mean())},
        {"metric": "n_paired_genes",   "value": float(len(gm))},
        {"metric": "n_sites",          "value": float(len(df))},
        {"metric": "n_features",       "value": float(len(feature_cols))},
        {"metric": "shap_dir_agree",   "value": float(n_yes / max(n_judged, 1)) if n_judged else np.nan},
    ])
    headline.to_csv(summary_path, sep="\t", index=False)

    print(f"\n{'='*72}\nDONE — Aim 1 finalized\n{'='*72}")
    print(f"\nAll outputs in: {args.out_dir}/")
    for f in ["headline_numbers.tsv", "group_ablation_results.tsv",
               "shap_feature_directions_corrected.tsv",
               "per_gene_metrics_combined.tsv",
               "stratified_by_cryptic_count.tsv",
               "figure1_approach_comparison.png",
               "figure2_shap_signed_importance.png",
               "figure3_shap_dependence.png",
               "figure4_group_ablation.png",
               "figure5_within_gene_stratified.png"]:
        full = os.path.join(args.out_dir, f)
        if os.path.exists(full):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (missing)")


if __name__ == "__main__":
    main()