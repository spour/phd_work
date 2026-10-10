#!/usr/bin/env python3
# rescue/break analysis comparing splicing_only and the margin-gated combiner against rf
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EPS = 1e-6


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def rf_top2(g):
    pr = np.sort(g["rf_prob"].to_numpy())[::-1]
    return (float(pr[0]), float(pr[1]) if len(pr) > 1 else 0.0)


def is_contested(g, hi, margin):
    top, run = rf_top2(g)
    return (run >= hi) and (top - run <= margin)


def fit_combiner(val_df):
    X = np.column_stack([safe_logit(val_df["rf_prob"]),
                            safe_logit(val_df["splicing_prob"])])
    y = val_df["is_canonical"].values
    clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X, y)
    return float(clf.intercept_[0]), float(clf.coef_[0, 0]), float(clf.coef_[0, 1])


def choose_gate_thresholds(val_df, w_rf, w_splice, max_contested_frac=0.20):
    n_val_genes = val_df["gene"].nunique()
    hi_grid = np.round(np.linspace(0.3, 0.85, 12), 3)
    margin_grid = np.round(np.linspace(0.02, 0.20, 10), 3)
    best = (0.6, 0.05)
    best_obj = -1e9
    best_diag = None
    for hi in hi_grid:
        for margin in margin_grid:
            rescued = broken = contested_n = 0
            for _, g in val_df.groupby("gene", sort=False):
                if not is_contested(g, hi, margin):
                    continue
                if (g["is_canonical"] == 1).sum() == 0:
                    continue
                contested_n += 1
                rf_top_idx = int(np.argmax(g["rf_prob"].to_numpy()))
                base_first = g.iloc[rf_top_idx]["is_canonical"] == 1
                s = (w_rf * safe_logit(g["rf_prob"].to_numpy()) +
                       w_splice * safe_logit(g["splicing_prob"].to_numpy()))
                gated_first = g.iloc[int(np.argmax(s))]["is_canonical"] == 1
                if base_first and not gated_first:
                    broken += 1
                elif (not base_first) and gated_first:
                    rescued += 1
            if (contested_n > max_contested_frac * n_val_genes or
                contested_n == 0):
                continue
            obj = rescued - broken
            if obj > best_obj:
                best_obj = obj
                best = (hi, margin)
                best_diag = (contested_n, rescued, broken)
    return best[0], best[1], best_diag


def gated_score(g, w_rf, w_splice, hi, margin):
    if is_contested(g, hi, margin):
        s = (w_rf * safe_logit(g["rf_prob"].to_numpy()) +
              w_splice * safe_logit(g["splicing_prob"].to_numpy()))
        return s, True
    return g["rf_prob"].to_numpy(), False


def per_gene_top1_table(df, hi, margin, w_rf, w_splice):
    rows = []
    for g_name, sub in df.groupby("gene", sort=False):
        if (sub["is_canonical"] == 1).sum() == 0:
            continue
        top1, runnerup = rf_top2(sub)
        rf_margin = top1 - runnerup
        n_competitors = int((sub["is_canonical"] == 0).sum())
        contested = bool(is_contested(sub, hi, margin))

        def top1_correct(score_col_or_arr):
            if isinstance(score_col_or_arr, str):
                arr = sub[score_col_or_arr].to_numpy()
            else:
                arr = score_col_or_arr
            idx = int(np.argmax(arr))
            return int(sub.iloc[idx]["is_canonical"] == 1)

        c_rf = top1_correct("rf_prob")
        c_sp = top1_correct("splicing_prob")
        ungated = (w_rf * safe_logit(sub["rf_prob"].to_numpy()) +
                    w_splice * safe_logit(sub["splicing_prob"].to_numpy()))
        c_ung = top1_correct(ungated)
        if contested:
            gated_arr = ungated
        else:
            gated_arr = sub["rf_prob"].to_numpy()
        c_gat = top1_correct(gated_arr)

        rows.append({
            "gene": g_name,
            "n_competitors": n_competitors,
            "rf_margin": rf_margin,
            "contested": contested,
            "correct_cpa_only": c_rf,
            "correct_splicing_only": c_sp,
            "correct_ungated_combiner": c_ung,
            "correct_gated_combiner": c_gat,
        })
    return pd.DataFrame(rows)


def contingency(per_gene, base_col, other_col, label=""):
    both_win = int(((per_gene[base_col] == 1) & (per_gene[other_col] == 1)).sum())
    base_only = int(((per_gene[base_col] == 1) & (per_gene[other_col] == 0)).sum())
    other_only = int(((per_gene[base_col] == 0) & (per_gene[other_col] == 1)).sum())
    both_lose = int(((per_gene[base_col] == 0) & (per_gene[other_col] == 0)).sum())
    total = both_win + base_only + other_only + both_lose
    base_right = both_win + base_only
    base_wrong = other_only + both_lose
    other_right = both_win + other_only
    return {
        "scorer": label,
        "total_genes": total,
        "rf_correct": base_right,
        "rf_wrong": base_wrong,
        "joint_correct": other_right,
        "both_win": both_win,
        "rescues": other_only,
        "breaks": base_only,
        "both_lose": both_lose,
        "net_benefit": other_only - base_only,
        "rescue_rate": other_only / base_wrong if base_wrong > 0 else np.nan,
        "break_rate": base_only / base_right if base_right > 0 else np.nan,
        "rf_p_at_1": base_right / total if total > 0 else np.nan,
        "joint_p_at_1": other_right / total if total > 0 else np.nan,
        "p_at_1_lift_pp": ((other_right - base_right) * 100.0 / total
                            if total > 0 else np.nan),
        "rescues_per_break": (other_only / base_only if base_only > 0
                                else float("inf")),
    }


def print_contingency_table(per_gene, scorer_cols, title=""):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    print(f"  {'scorer':<22s} {'rf_OK':>6s} {'rf_X':>5s} {'res':>5s} {'brk':>5s} "
           f"{'net':>5s} {'rescue%':>9s} {'break%':>8s} {'res/brk':>9s} "
           f"{'jnt_P@1':>9s}")
    rows = []
    for col, lbl in scorer_cols:
        c = contingency(per_gene, "correct_cpa_only", col, label=lbl)
        rows.append(c)
        rpb = (f"{c['rescues_per_break']:.2f}"
                if c["rescues_per_break"] != float("inf") else "inf")
        print(f"  {lbl:<22s} {c['rf_correct']:>6d} {c['rf_wrong']:>5d} "
               f"{c['rescues']:>5d} {c['breaks']:>5d} {c['net_benefit']:>+5d} "
               f"{c['rescue_rate']*100:>8.1f}% {c['break_rate']*100:>7.1f}% "
               f"{rpb:>9s} {c['joint_p_at_1']:>9.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_candidates", required=True)
    ap.add_argument("--test_candidates", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    val = pd.read_csv(args.val_candidates, sep="\t")
    test = pd.read_csv(args.test_candidates, sep="\t")
    print(f"[gated] val: {len(val):,} rows from {val['gene'].nunique():,} genes",
           file=sys.stderr)
    print(f"[gated] test: {len(test):,} rows from {test['gene'].nunique():,} genes",
           file=sys.stderr)

    intercept, w_rf, w_splice = fit_combiner(val)
    print(f"\n[gated] combiner fit on val:", file=sys.stderr)
    print(f"  w_logit_rf     = {w_rf:+.4f}", file=sys.stderr)
    print(f"  w_logit_splice = {w_splice:+.4f}", file=sys.stderr)

    hi, margin, diag = choose_gate_thresholds(val, w_rf, w_splice)
    if diag:
        cn, rs, bk = diag
        print(f"\n[gated] gate thresholds chosen on val:", file=sys.stderr)
        print(f"  runner-up >= {hi}, top - runnerup <= {margin}", file=sys.stderr)
        print(f"  val: {cn} contested genes, rescued {rs}, broken {bk}, "
               f"net {rs - bk:+d}", file=sys.stderr)
    else:
        print(f"\n[gated] WARNING: no thresholds met criteria; using defaults",
               file=sys.stderr)

    per_gene = per_gene_top1_table(test, hi, margin, w_rf, w_splice)
    print(f"\n[gated] test per-gene table: {len(per_gene):,} genes", file=sys.stderr)
    n_contested = int(per_gene["contested"].sum())
    print(f"[gated]   contested (gate fires): {n_contested:,} ({100*n_contested/len(per_gene):.1f}%)",
           file=sys.stderr)

    scorers = [
        ("correct_splicing_only", "splicing_only"),
        ("correct_ungated_combiner", "ungated_combiner"),
        ("correct_gated_combiner", "gated_combiner"),
    ]
    headline_rows = print_contingency_table(
        per_gene, scorers,
        title="HEADLINE: rescue/break per scorer on 1,004 CCPCPA test genes")

    print(f"\n{'='*80}")
    print(f"STRATIFICATION 1 (the honest one): by RF margin")
    print(f"{'='*80}")
    margin_bins = [-0.01, 0.01, 0.05, 0.10, 0.20, 0.50, 1.01]
    margin_labels = ["<0.01", "0.01-0.05", "0.05-0.10", "0.10-0.20",
                      "0.20-0.50", ">=0.50"]
    per_gene["margin_bin"] = pd.cut(per_gene["rf_margin"], bins=margin_bins,
                                          labels=margin_labels,
                                          include_lowest=True)
    strat1_rows = []
    print(f"\n  GATED COMBINER, stratified by RF margin:")
    print(f"  {'margin':<14s} {'n':>5s} {'rf_OK':>6s} {'rf_X':>5s} {'res':>5s} "
           f"{'brk':>5s} {'net':>5s} {'rescue%':>9s} {'break%':>8s} "
           f"{'res/brk':>9s}")
    for bin_val in margin_labels:
        sub = per_gene[per_gene["margin_bin"] == bin_val]
        if len(sub) == 0:
            continue
        c = contingency(sub, "correct_cpa_only", "correct_gated_combiner",
                          label=f"gated@{bin_val}")
        strat1_rows.append({"stratum": "rf_margin", "bin": bin_val,
                             "n_genes": len(sub), **c})
        rpb = (f"{c['rescues_per_break']:.2f}"
                if c["rescues_per_break"] != float("inf") else "inf")
        print(f"  {bin_val:<14s} {len(sub):>5d} {c['rf_correct']:>6d} "
               f"{c['rf_wrong']:>5d} {c['rescues']:>5d} {c['breaks']:>5d} "
               f"{c['net_benefit']:>+5d} {c['rescue_rate']*100:>8.1f}% "
               f"{c['break_rate']*100:>7.1f}% {rpb:>9s}")

    print(f"\n{'='*80}")
    print(f"STRATIFICATION 2 (the other honest one): by competitor count")
    print(f"{'='*80}")
    comp_bins = [0, 1, 2, 5, 10, 100]
    comp_labels = ["1", "2", "3-5", "6-10", ">10"]
    per_gene["comp_bin"] = pd.cut(per_gene["n_competitors"], bins=comp_bins,
                                        labels=comp_labels)
    strat2_rows = []
    print(f"\n  GATED COMBINER, stratified by competitor count:")
    print(f"  {'n_compet':<10s} {'n':>5s} {'rf_OK':>6s} {'rf_X':>5s} {'res':>5s} "
           f"{'brk':>5s} {'net':>5s} {'rescue%':>9s} {'break%':>8s} "
           f"{'res/brk':>9s}")
    for bin_val in comp_labels:
        sub = per_gene[per_gene["comp_bin"] == bin_val]
        if len(sub) == 0:
            continue
        c = contingency(sub, "correct_cpa_only", "correct_gated_combiner",
                          label=f"gated@{bin_val}")
        strat2_rows.append({"stratum": "n_competitors", "bin": bin_val,
                             "n_genes": len(sub), **c})
        rpb = (f"{c['rescues_per_break']:.2f}"
                if c["rescues_per_break"] != float("inf") else "inf")
        print(f"  {bin_val:<10s} {len(sub):>5d} {c['rf_correct']:>6d} "
               f"{c['rf_wrong']:>5d} {c['rescues']:>5d} {c['breaks']:>5d} "
               f"{c['net_benefit']:>+5d} {c['rescue_rate']*100:>8.1f}% "
               f"{c['break_rate']*100:>7.1f}% {rpb:>9s}")

    pd.DataFrame(headline_rows).to_csv(
        os.path.join(args.out_dir, "rescue_summary_gated.tsv"),
        sep="\t", index=False)
    pd.DataFrame(strat1_rows + strat2_rows).to_csv(
        os.path.join(args.out_dir, "stratification_gated.tsv"),
        sep="\t", index=False)
    per_gene.to_csv(os.path.join(args.out_dir, "per_gene_status_gated.tsv"),
                      sep="\t", index=False)

    print(f"\n[gated] wrote rescue_summary_gated.tsv, stratification_gated.tsv, "
           f"per_gene_status_gated.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()