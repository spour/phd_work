#!/usr/bin/env python3
# shap analysis of the combined 55-feature splicing gbt
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED = {
    "term_3ss__wg":           ("+ (canonical)", "strong terminal 3'SS = real terminal exon"),
    "upstream_max_me3__wg":   ("+ (canonical)", "strong upstream 3'SS background = real terminal exon"),
    "upstream_mean_u1__wg":   ("+ (canonical)", "splicing-competent gene region (telescripting context)"),
    "pas_prox_u1__wg":        ("- (3-way)",     "strong U1 at PAS = U1 binding competes with CPA"),
    "local_me5__wg":          ("- (3-way)",     "strong local 5'SS at PAS = intronic context"),
    "dn5ss_has_strong__wg":   ("- (3-way)",     "strong downstream 5'SS = intron continues"),
    "interv_me5_max__wg":     ("- (3-way)",     "intervening 5'SS in putative terminal exon = exon interrupted"),
}


def parse_bin_feature(name):
    if not name.startswith("bin"):
        return None
    s = name[3:]
    bin_idx_str, rest = s.split("_", 1)
    motif, stat = rest.rsplit("_", 1)
    return int(bin_idx_str), motif, stat


def expected_direction_bin(bin_idx, motif, n_bins, upstream_nt, downstream_nt):
    total = upstream_nt + downstream_nt
    bin_width = total // n_bins
    upstream_bins = upstream_nt // bin_width
    is_upstream = bin_idx <= upstream_bins
    is_pas_proximal = bin_idx == upstream_bins
    is_far_downstream = bin_idx == n_bins

    if motif == "me3":
        if is_pas_proximal:
            return ("+", "PAS-proximal 3'SS = real terminal exon")
        elif is_upstream:
            return ("+", "upstream 3'SS supports terminal-exon identity")
        else:
            return ("?", "downstream 3'SS — no clean expectation")
    elif motif == "me5":
        if is_upstream:
            return ("?", "upstream 5'SS — could indicate intervening intron in 'exon'")
        else:
            return ("-", "downstream 5'SS = intron continues past PAS")
    elif motif == "u1":
        if is_pas_proximal:
            return ("-", "PAS-proximal U1 = U1 competes with cleavage")
        elif is_upstream:
            return ("+", "upstream U1 = splicing-competent context")
        else:
            return ("-", "downstream U1 = telescripting protection")
    return ("?", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--model_joblib", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--n_background", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=3000)
    ap.add_argument("--top_n_summary", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--upstream_nt", type=int, default=3000)
    ap.add_argument("--downstream_nt", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=8)
    args = ap.parse_args()

    try:
        import shap
    except ImportError:
        raise SystemExit("shap required. pip install shap")

    print(f"[shap] loading model {args.model_joblib}")
    bundle = joblib.load(args.model_joblib)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    print(f"[shap]   {len(feature_cols)} features")

    print(f"[shap] loading features {args.features_tsv}")
    df = pd.read_csv(args.features_tsv, sep="\t")
    X = df[feature_cols].values
    y = df["y"].values.astype(int)

    rng = np.random.default_rng(args.seed)
    bg_idx = rng.choice(len(X), size=min(args.n_background, len(X)), replace=False)
    ev_idx = rng.choice(len(X), size=min(args.n_eval, len(X)), replace=False)

    print(f"[shap] preparing data (imputer on bg/eval samples)")
    imputer = model.named_steps["imp"]
    clf = model.named_steps["gbt"]
    X_bg = imputer.transform(X[bg_idx])
    X_ev = imputer.transform(X[ev_idx])
    y_ev = y[ev_idx]

    print(f"[shap] fitting TreeExplainer (interventional, {args.n_background} bg samples)")
    explainer = shap.TreeExplainer(clf, data=X_bg, feature_perturbation="interventional")

    print(f"[shap] computing SHAP values on {args.n_eval} eval samples")
    shap_values = explainer.shap_values(X_ev, check_additivity=False)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    print(f"\n[shap] computing per-feature direction (mean(SHAP@top10%) - mean(SHAP@bottom10%))")
    rows = []
    for j, name in enumerate(feature_cols):
        x = X_ev[:, j]
        s = shap_vals[:, j]
        ok = np.isfinite(x) & np.isfinite(s)
        if ok.sum() < 100:
            continue
        x = x[ok]; s = s[ok]
        q10 = np.quantile(x, 0.10)
        q90 = np.quantile(x, 0.90)
        bot_shap = s[x <= q10]
        top_shap = s[x >= q90]
        mean_abs = float(np.mean(np.abs(s)))
        delta = (float(np.mean(top_shap)) - float(np.mean(bot_shap))
                  if len(top_shap) and len(bot_shap) else np.nan)
        if np.isfinite(delta) and abs(delta) > 1e-12:
            signed_importance = np.sign(delta) * mean_abs
        else:
            signed_importance = 0.0

        observed_dir = "+" if delta > 0 else "-" if delta < 0 else "?"

        if name in EXPECTED:
            exp_dir_full, exp_reason = EXPECTED[name]
            exp_dir = exp_dir_full[0]
            feature_kind = "axis"
        elif name.startswith("bin"):
            parsed = parse_bin_feature(name)
            if parsed:
                bin_idx, motif, _ = parsed
                exp_dir, exp_reason = expected_direction_bin(
                    bin_idx, motif, args.n_bins,
                    args.upstream_nt, args.downstream_nt)
            else:
                exp_dir, exp_reason = ("?", "")
            feature_kind = "bin"
        else:
            exp_dir, exp_reason = ("?", "unmapped")
            feature_kind = "other"

        agrees = (exp_dir == observed_dir) if exp_dir in ("+", "-") else None

        rows.append({
            "feature": name,
            "kind": feature_kind,
            "mean_abs_shap": mean_abs,
            "signed_importance": signed_importance,
            "delta_top10_minus_bottom10_shap": delta,
            "observed_direction": observed_dir,
            "expected_direction": exp_dir,
            "agrees_with_expectation": agrees,
            "expected_reason": exp_reason,
        })

    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out_tsv = f"{args.out_prefix}_feature_directions.tsv"
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"[shap] wrote {out_tsv}")

    print(f"\n{'─'*100}")
    print(f"TOP-{args.top_n_summary} FEATURES BY MEAN |SHAP|")
    print(f"{'─'*100}")
    print(f"{'feature':<30} {'kind':<6} {'|SHAP|':>10} {'obs':>4} {'exp':>4} {'agree':>6}  reason")
    print(f"{'─'*100}")
    for _, r in out.head(args.top_n_summary).iterrows():
        agree_mark = ("YES" if r["agrees_with_expectation"] is True
                       else "NO" if r["agrees_with_expectation"] is False
                       else "?")
        print(f"{r['feature']:<30} {r['kind']:<6} {r['mean_abs_shap']:>10.4f}  "
               f"{r['observed_direction']:>3}  {r['expected_direction']:>3}  "
               f"{agree_mark:>6}  {r['expected_reason']}")

    judged = out[out["expected_direction"].isin(["+", "-"])]
    if len(judged) > 0:
        agree_rate = (judged["agrees_with_expectation"] == True).sum() / len(judged)
        print(f"\nOverall direction agreement (across features with a clear expectation): "
               f"{(judged['agrees_with_expectation']==True).sum()}/{len(judged)} = {agree_rate:.1%}")

    top = out.head(args.top_n_summary).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 0.35 * len(top) + 1))
    colors = ["#2E8B57" if v > 0 else "#C23B3B" for v in top["signed_importance"]]
    yp = np.arange(len(top))
    ax.barh(yp, top["signed_importance"], color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#444", linewidth=1.0)
    ax.set_yticks(yp)
    labels = [f"{f}  [{k[0].upper()}]" for f, k in zip(top["feature"], top["kind"])]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Signed mean |SHAP|  (+ favors CANONICAL, - favors 3-WAY intragenic)",
                   fontsize=11)
    ax.set_title(f"Top {args.top_n_summary} features by SHAP importance (combined model)",
                  fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png = f"{args.out_prefix}_signed_importance.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[shap] wrote {png}")

    print(f"[shap] generating SHAP summary plot")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_ev, feature_names=feature_cols,
                       show=False, max_display=args.top_n_summary)
    plt.title("SHAP summary (combined model)", fontsize=12, fontweight="bold", loc="left")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_summary.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"[shap] wrote {args.out_prefix}_summary.png")

    print(f"[shap] generating dependence plots for top 6 features")
    top6 = out.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(top6):
        j = feature_cols.index(feat)
        shap.dependence_plot(j, shap_vals, X_ev,
                              feature_names=feature_cols,
                              ax=axes[i], show=False)
        axes[i].set_title(feat, fontsize=10, fontweight="bold")
    fig.suptitle("SHAP dependence (top 6 features)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{args.out_prefix}_dependence.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"[shap] wrote {args.out_prefix}_dependence.png")

    print(f"\n[shap] done.")


if __name__ == "__main__":
    main()