#!/usr/bin/env python3
# genome-wide x vs y on the same candidate set that global_rf_canonical_rate.py used
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
VAL_CHROMS = {"chr3", "chr13"}
LEN_BINS = [0, 5000, 20000, 50000, 100000, 5e8]
LEN_LABELS = ["<5kb", "5-20kb", "20-50kb", "50-100kb", ">100kb"]


def norm_chrom(c):
    c = str(c).strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def find_peaks_nms(scores, positions, min_distance_nt, max_peaks):
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
        if len(chosen) >= max_peaks:
            break
    return np.array(chosen, dtype=int)


def rf_top1_margin(g):
    pr = np.sort(g["rf_prob"].to_numpy())[::-1]
    top = float(pr[0])
    run = float(pr[1]) if len(pr) > 1 else 0.0
    return top, top - run


def rf_is_confident(g, hi_conf, min_margin):
    top, margin = rf_top1_margin(g)
    return (top >= hi_conf) and (margin >= min_margin)


def canonical_margin(g):
    if g["in_zone"].sum() == 0:
        return None
    canonical_rows = g[g["in_zone"] == 1]
    internal_rows = g[g["in_zone"] == 0]
    if len(internal_rows) == 0:
        return None
    return float(canonical_rows["rf_prob"].max()) - float(internal_rows["rf_prob"].max())


def score_per_gene(g, scorer, w=None, hi_conf=None, min_margin=None):
    rf = g["rf_prob"].to_numpy()
    sp = g["splicing_prob"].to_numpy()
    if scorer == "cpa_only":
        return rf
    if scorer == "splicing_only":
        return sp
    combiner = w[0] * safe_logit(rf) + w[1] * safe_logit(sp)
    if scorer == "ungated":
        return combiner
    if scorer == "gated":
        return rf if rf_is_confident(g, hi_conf, min_margin) else combiner
    raise ValueError(f"unknown scorer {scorer}")


def per_gene_top1(df, scorer, w=None, hi_conf=None, min_margin=None):
    recs = []
    for gene, g in df.groupby("gene", sort=False):
        glen = int(g["gene_length"].iloc[0])
        if g["in_zone"].sum() == 0:
            recs.append((gene, 0, 0, glen))
            continue
        scores = score_per_gene(g, scorer, w, hi_conf, min_margin)
        hit = int(g.iloc[int(np.argmax(scores))]["in_zone"] == 1)
        recs.append((gene, hit, 1, glen))
    return pd.DataFrame(recs, columns=["gene", "hit", "recall", "gene_length"])


def boot_ci(v, B=4000, seed=0):
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, n, (B, n))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def choose_gate_on_val(val_df, w):
    hi_grid = [0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    margin_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    best = (0.95, 0.10)
    best_p1 = -1.0
    all_results = []
    for hi_conf in hi_grid:
        for min_margin in margin_grid:
            n_total = 0
            n_protected = 0
            n_hits = 0
            for gene, g in val_df.groupby("gene", sort=False):
                if g["in_zone"].sum() == 0:
                    n_total += 1
                    continue
                n_total += 1
                if rf_is_confident(g, hi_conf, min_margin):
                    n_protected += 1
                    scores = g["rf_prob"].to_numpy()
                else:
                    scores = (w[0] * safe_logit(g["rf_prob"].to_numpy()) +
                                w[1] * safe_logit(g["splicing_prob"].to_numpy()))
                n_hits += int(g.iloc[int(np.argmax(scores))]["in_zone"] == 1)
            p1 = n_hits / n_total if n_total > 0 else 0.0
            n_gated = n_total - n_protected
            all_results.append({"hi_conf": hi_conf, "min_margin": min_margin,
                                  "val_p_at_1": p1, "n_protected": n_protected,
                                  "n_gated": n_gated, "n_total": n_total})
            if p1 > best_p1:
                best_p1 = p1
                best = (hi_conf, min_margin)
    return best[0], best[1], all_results, best_p1


def build_table(genes, meta_df, args, split_name=""):
    rows = []
    n_no_cpa = n_no_splice = n_no_canon = n_no_peaks = n_other = 0
    for i, gene in enumerate(genes):
        cpa_path = os.path.join(args.cpa_dir, f"{gene}.parquet")
        sp_path = os.path.join(args.splice_dir, f"{gene}.parquet")
        if not os.path.exists(cpa_path):
            n_no_cpa += 1
            continue
        if not os.path.exists(sp_path):
            n_no_splice += 1
            continue
        try:
            cpa = pd.read_parquet(cpa_path,
                                    columns=["window_start_genomic", "rf_prob",
                                              "is_constitutive_cpa"])
            sp = pd.read_parquet(sp_path,
                                   columns=["window_start_genomic", "splicing_prob"])
        except Exception:
            n_other += 1
            continue
        canonical_positions = cpa.loc[cpa["is_constitutive_cpa"] == True,
                                         "window_start_genomic"].values
        if len(canonical_positions) == 0:
            n_no_canon += 1
            continue
        merged = cpa.merge(sp, on="window_start_genomic", how="inner")
        merged = merged.dropna(subset=["rf_prob", "splicing_prob"])
        if len(merged) == 0:
            n_other += 1
            continue
        positions = merged["window_start_genomic"].values
        rf_scores = merged["rf_prob"].values
        sp_scores = merged["splicing_prob"].values
        peak_idx = find_peaks_nms(rf_scores, positions,
                                     args.min_peak_distance, args.max_peaks)
        if len(peak_idx) == 0:
            n_no_peaks += 1
            continue
        glen = (int(meta_df.loc[gene, "gene_length"])
                 if gene in meta_df.index else 0)
        for j in peak_idx:
            ppos = int(positions[j])
            in_zone = int(np.min(np.abs(canonical_positions - ppos))
                            <= args.tolerance_nt)
            rows.append({
                "gene": gene, "gene_length": glen,
                "position": ppos,
                "rf_prob": float(rf_scores[j]),
                "splicing_prob": float(sp_scores[j]),
                "in_zone": in_zone,
            })
        if (i + 1) % 2000 == 0:
            print(f"[{split_name}]   {i+1:,}/{len(genes):,} "
                   f"({len({r['gene'] for r in rows}):,} scored, "
                   f"no_cpa={n_no_cpa}, no_splice={n_no_splice}, "
                   f"no_canon={n_no_canon}, no_peaks={n_no_peaks}, "
                   f"other={n_other})", file=sys.stderr)
    n_genes = len({r["gene"] for r in rows})
    print(f"[{split_name}] {n_genes:,} genes scored, {len(rows):,} peaks "
           f"(no_cpa={n_no_cpa}, no_splice={n_no_splice}, no_canon={n_no_canon}, "
           f"no_peaks={n_no_peaks}, other={n_other})", file=sys.stderr)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_dir", required=True,
                     help="Dir with CPA per-gene parquets")
    ap.add_argument("--splice_dir", required=True,
                     help="Dir with 55-feature splicing scan parquets")
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tolerance_nt", type=int, default=250)
    ap.add_argument("--min_peak_distance", type=int, default=250)
    ap.add_argument("--max_peaks", type=int, default=20)
    ap.add_argument("--limit_genes", type=int, default=0,
                     help="If >0, limit per split (val AND test each capped). "
                          "Use for smoke tests; 0 = process all.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pc = pd.read_csv(args.pc_genes_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    pc["gene"] = pc["name"].astype(str)
    if pc["gene"].str.contains("_").any():
        pc["gene"] = pc["gene"].str.split("_").str[-1]
    pc["chrom_norm"] = pc["chrom"].apply(norm_chrom)
    pc = pc.drop_duplicates(subset="gene", keep="first")
    pc["gene_length"] = pc["end"] - pc["start"]
    pc["split"] = "train"
    pc.loc[pc["chrom_norm"].isin(VAL_CHROMS), "split"] = "val"
    pc.loc[pc["chrom_norm"].isin(TEST_CHROMS), "split"] = "test"
    meta = pc.set_index("gene")
    print(f"[gw] {len(pc):,} unique genes in BED", file=sys.stderr)
    print(f"[gw] splits: {pc['split'].value_counts().to_dict()}", file=sys.stderr)

    val_genes = sorted(pc.loc[pc["split"] == "val", "gene"].unique().tolist())
    test_genes = sorted(pc.loc[pc["split"] == "test", "gene"].unique().tolist())

    if args.limit_genes > 0:
        rng = np.random.default_rng(42)
        if len(val_genes) > args.limit_genes:
            val_idx = rng.choice(len(val_genes), args.limit_genes, replace=False)
            val_genes = [val_genes[i] for i in sorted(val_idx)]
        if len(test_genes) > args.limit_genes:
            test_idx = rng.choice(len(test_genes), args.limit_genes, replace=False)
            test_genes = [test_genes[i] for i in sorted(test_idx)]
        print(f"[gw] LIMIT_GENES={args.limit_genes}: sampled "
               f"{len(val_genes)} val and {len(test_genes)} test genes randomly",
               file=sys.stderr)

    print(f"[gw] val: {len(val_genes):,} genes; test: {len(test_genes):,} genes",
           file=sys.stderr)

    print(f"\n[gw] Building val peak table...", file=sys.stderr)
    val_df = build_table(val_genes, meta, args, split_name="val")
    print(f"\n[gw] Building test peak table...", file=sys.stderr)
    test_df = build_table(test_genes, meta, args, split_name="test")

    if val_df.empty or test_df.empty:
        sys.exit("[gw] ERROR: no peaks extracted")

    val_df.to_csv(os.path.join(args.out_dir, "val_peaks.tsv"),
                    sep="\t", index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_peaks.tsv"),
                    sep="\t", index=False)

    val_recall = val_df.groupby("gene")["in_zone"].max().mean()
    test_recall = test_df.groupby("gene")["in_zone"].max().mean()
    test_cpa_hits = per_gene_top1(test_df, "cpa_only")
    test_cpa_p1 = float(test_cpa_hits["hit"].mean())
    print(f"\n[gw] SANITY CHECKS:", file=sys.stderr)
    print(f"  val candidate recall:  {val_recall:.4f}", file=sys.stderr)
    print(f"  test candidate recall: {test_recall:.4f}", file=sys.stderr)
    print(f"  test cpa_only P@1:     {test_cpa_p1:.4f}", file=sys.stderr)
    print(f"  (expected from global_rf_canonical_rate.py Def 2: ~0.406)",
           file=sys.stderr)
    if abs(test_cpa_p1 - 0.406) > 0.05:
        print(f"  WARNING: X is {test_cpa_p1:.4f}, expected ~0.406. Check that "
               f"--cpa_dir matches what global_rf_canonical_rate.py used.",
               file=sys.stderr)

    Xv = np.column_stack([safe_logit(val_df["rf_prob"]),
                             safe_logit(val_df["splicing_prob"])])
    clf = LogisticRegression(max_iter=1000).fit(Xv, val_df["in_zone"])
    w = clf.coef_[0]
    print(f"\n[gw] combiner (val fit):", file=sys.stderr)
    print(f"  w_logit_rf     = {w[0]:+.4f}", file=sys.stderr)
    print(f"  w_logit_splice = {w[1]:+.4f}", file=sys.stderr)

    print(f"\n[gw] searching gate thresholds on val...", file=sys.stderr)
    hi_conf, min_margin, gate_results, val_p1 = choose_gate_on_val(val_df, w)
    print(f"  best: hi_conf={hi_conf}, min_margin={min_margin}, "
           f"val_p_at_1={val_p1:.4f}", file=sys.stderr)
    n_test_protected = sum(1 for _, g in test_df.groupby("gene")
                             if rf_is_confident(g, hi_conf, min_margin))
    n_test_total = test_df["gene"].nunique()
    print(f"  test: {n_test_protected}/{n_test_total} protected "
           f"({100*n_test_protected/n_test_total:.1f}%); "
           f"{n_test_total - n_test_protected} gated "
           f"({100*(n_test_total - n_test_protected)/n_test_total:.1f}%)",
           file=sys.stderr)
    pd.DataFrame(gate_results).to_csv(
        os.path.join(args.out_dir, "gate_search.tsv"), sep="\t", index=False)

    print(f"\n[gw] TEST genome-wide gene-end P@1 (n={n_test_total} genes):",
           file=sys.stderr)
    print(f"  {'scorer':<22s}{'P@1':>8s}{'95% CI':>20s}{'lift_vs_X':>11s}",
           file=sys.stderr)
    summary_rows = []
    cpa_p1_value = None
    for label, scorer in [("cpa_only (X)", "cpa_only"),
                            ("splicing_only", "splicing_only"),
                            ("ungated_combiner", "ungated"),
                            ("gated (Y)", "gated")]:
        h = per_gene_top1(test_df, scorer, w, hi_conf, min_margin)
        m, lo, hi_ci = boot_ci(h["hit"])
        if scorer == "cpa_only":
            cpa_p1_value = m
            lift_str = "—"
        else:
            lift_str = f"{m - cpa_p1_value:+.4f}"
        summary_rows.append({"scorer": label, "p_at_1": m,
                              "ci_lo": lo, "ci_hi": hi_ci, "n_genes": len(h)})
        print(f"  {label:<22s}{m:>8.4f}  [{lo:.4f}, {hi_ci:.4f}]{lift_str:>11s}",
               file=sys.stderr)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, "genome_wide_summary.tsv"),
        sep="\t", index=False)

    gene_margins = {}
    for gene, g in test_df.groupby("gene", sort=False):
        gene_margins[gene] = canonical_margin(g)

    hx = per_gene_top1(test_df, "cpa_only", w, hi_conf, min_margin)
    hs = per_gene_top1(test_df, "splicing_only", w, hi_conf, min_margin)
    hu = per_gene_top1(test_df, "ungated", w, hi_conf, min_margin)
    hg = per_gene_top1(test_df, "gated", w, hi_conf, min_margin)
    hx["canonical_margin"] = hx["gene"].map(gene_margins)
    hs["canonical_margin"] = hs["gene"].map(gene_margins)
    hu["canonical_margin"] = hu["gene"].map(gene_margins)
    hg["canonical_margin"] = hg["gene"].map(gene_margins)

    margin_bins = [-1.01, -0.2, -0.05, 0.05, 0.2, 1.01]
    margin_labels = ["< -0.2", "-0.2 to -0.05", "-0.05 to 0.05",
                      "0.05 to 0.2", "> 0.2"]
    margin_descriptions = {
        "< -0.2":         "internal beats canonical clearly",
        "-0.2 to -0.05":  "internal beats canonical narrowly",
        "-0.05 to 0.05":  "essentially tied",
        "0.05 to 0.2":    "canonical beats internal narrowly",
        "> 0.2":          "canonical wins clearly",
    }
    for h in [hx, hs, hu, hg]:
        h["margin_bin"] = pd.cut(h["canonical_margin"], bins=margin_bins,
                                       labels=margin_labels)

    n_no_canon_peak = int(hx["canonical_margin"].isna().sum())

    print(f"\n[gw] PRIMARY: X vs joint scorers by canonical_margin "
           f"(test, n={len(hx)} genes):", file=sys.stderr)
    print(f"  ({n_no_canon_peak} genes with canonical NOT in peaks → ceiling miss)",
           file=sys.stderr)
    print(f"  {'margin_bin':<14s} {'description':<34s} {'n':>5s} "
           f"{'X_cpa':>7s} {'splice':>7s} {'ung':>7s} {'Y_gat':>7s} "
           f"{'lift_Y':>7s}", file=sys.stderr)
    margin_strat_rows = []
    for b in margin_labels:
        gx = hx[hx["margin_bin"] == b]
        gs = hs[hs["margin_bin"] == b]
        gu = hu[hu["margin_bin"] == b]
        gg = hg[hg["margin_bin"] == b]
        if len(gx) == 0:
            continue
        X = float(gx["hit"].mean())
        S = float(gs["hit"].mean())
        U = float(gu["hit"].mean())
        Y = float(gg["hit"].mean())
        desc = margin_descriptions[b]
        print(f"  {b:<14s} {desc:<34s} {len(gx):>5d} {X:>7.4f} "
               f"{S:>7.4f} {U:>7.4f} {Y:>7.4f} {Y - X:>+7.4f}",
               file=sys.stderr)
        margin_strat_rows.append({
            "margin_bin": b, "description": desc, "n_genes": len(gx),
            "X_cpa_only": X, "splicing_only": S,
            "ungated_combiner": U, "Y_gated": Y, "lift": Y - X,
        })
    if n_no_canon_peak > 0:
        gx_miss = hx[hx["canonical_margin"].isna()]
        margin_strat_rows.append({
            "margin_bin": "(canonical not in peaks)",
            "description": "ceiling miss", "n_genes": int(n_no_canon_peak),
            "X_cpa_only": float(gx_miss["hit"].mean()),
            "splicing_only": float(hs[hs["canonical_margin"].isna()]["hit"].mean()),
            "ungated_combiner": float(hu[hu["canonical_margin"].isna()]["hit"].mean()),
            "Y_gated": float(hg[hg["canonical_margin"].isna()]["hit"].mean()),
            "lift": 0.0,
        })
    pd.DataFrame(margin_strat_rows).to_csv(
        os.path.join(args.out_dir, "X_vs_Y_by_canonical_margin.tsv"),
        sep="\t", index=False)

    hx["length_bin"] = pd.cut(hx["gene_length"], LEN_BINS, labels=LEN_LABELS)
    hg["length_bin"] = pd.cut(hg["gene_length"], LEN_BINS, labels=LEN_LABELS)
    print(f"\n[gw] SECONDARY: X vs Y by gene length (test):", file=sys.stderr)
    print(f"  {'length':<10s}{'n':>6s}{'X':>8s}{'Y':>8s}{'lift':>8s}",
           file=sys.stderr)
    strat_rows = []
    for b in LEN_LABELS:
        gx = hx[hx["length_bin"] == b]
        gy = hg[hg["length_bin"] == b]
        if len(gx) == 0:
            continue
        X = float(gx["hit"].mean())
        Y = float(gy["hit"].mean())
        print(f"  {b:<10s}{len(gx):>6d}{X:>8.4f}{Y:>8.4f}{Y - X:>+8.4f}",
               file=sys.stderr)
        strat_rows.append({"length_bin": b, "n_genes": len(gx),
                            "X_cpa_only": X, "Y_gated": Y,
                            "lift": Y - X})
    pd.DataFrame(strat_rows).to_csv(
        os.path.join(args.out_dir, "X_vs_Y_by_length.tsv"),
        sep="\t", index=False)

    per_gene_table = pd.DataFrame({
        "gene": hx["gene"],
        "gene_length": hx["gene_length"],
        "canonical_margin": hx["canonical_margin"],
        "cpa_only_hit": hx["hit"],
        "splicing_only_hit": hs["hit"],
        "ungated_combiner_hit": hu["hit"],
        "gated_hit": hg["hit"],
    })
    per_gene_table.to_csv(
        os.path.join(args.out_dir, "per_gene_results.tsv"),
        sep="\t", index=False)

    print(f"\n[gw] wrote val_peaks.tsv, test_peaks.tsv, gate_search.tsv, "
           f"genome_wide_summary.tsv, X_vs_Y_by_canonical_margin.tsv, "
           f"X_vs_Y_by_length.tsv, per_gene_results.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()