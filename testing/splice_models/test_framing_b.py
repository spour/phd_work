#!/usr/bin/env python3
# test the claim: among sites that are similarly sequence-competent by cpa rf
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


EPS = 1e-6


def normalize_chrom(c):
    c = str(c).strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def build_candidates_for_gene(gene, cpa_df, sp_df, gene_labels, tolerance_nt):
    canonical_centers = gene_labels.loc[gene_labels["y"] == 1, "center"].values
    cryptic_centers = gene_labels.loc[gene_labels["y"] == 0, "center"].values
    if len(canonical_centers) == 0 or len(cryptic_centers) == 0:
        return None

    merged = cpa_df.merge(sp_df, on="window_start_genomic", how="inner")
    merged = merged.dropna(subset=["rf_prob", "splicing_prob"])
    if len(merged) == 0:
        return None

    positions = merged["window_start_genomic"].values

    rows = []
    for cc in canonical_centers:
        mask = np.abs(positions - cc) <= tolerance_nt
        if mask.any():
            sub = merged[mask]
            idx = sub["rf_prob"].idxmax()
            rows.append({
                "gene": gene,
                "position": int(merged.loc[idx, "window_start_genomic"]),
                "rf_prob": float(merged.loc[idx, "rf_prob"]),
                "splicing_prob": float(merged.loc[idx, "splicing_prob"]),
                "is_canonical": 1,
                "label_center": int(cc),
            })

    for cc in cryptic_centers:
        mask = np.abs(positions - cc) <= tolerance_nt
        if mask.any():
            sub = merged[mask]
            idx = sub["rf_prob"].idxmax()
            rows.append({
                "gene": gene,
                "position": int(merged.loc[idx, "window_start_genomic"]),
                "rf_prob": float(merged.loc[idx, "rf_prob"]),
                "splicing_prob": float(merged.loc[idx, "splicing_prob"]),
                "is_canonical": 0,
                "label_center": int(cc),
            })

    return pd.DataFrame(rows)


def per_gene_top1(df, score_col):
    hits = 0
    n = 0
    rows = []
    for g, sub in df.groupby("gene", sort=False):
        if (sub["is_canonical"] == 1).sum() == 0:
            continue
        n += 1
        top_idx = sub[score_col].idxmax()
        hit = int(sub.loc[top_idx, "is_canonical"] == 1)
        hits += hit
        rows.append({"gene": g, "hit": hit,
                      "n_competitors": len(sub) - 1,
                      "top_score": float(sub.loc[top_idx, score_col])})
    return hits, n, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_parquet_dir", required=True)
    ap.add_argument("--splicing_parquet_dir", required=True)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--test_chroms", default="chr18,chr19,chr20,chr21,chr22,chrX")
    ap.add_argument("--val_chroms", default="chr3,chr13")
    ap.add_argument("--rf_threshold", type=float, default=0.5,
                     help="A site is 'sequence-competent' if its rf_prob exceeds "
                          "this. Restrict to genes where canonical AND at least "
                          "one CCPCPA both meet this threshold.")
    ap.add_argument("--tolerance_nt", type=int, default=250)
    ap.add_argument("--cpa_model", default="rf_prob")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    val_chroms = {normalize_chrom(c) for c in args.val_chroms.split(",")}
    test_chroms = {normalize_chrom(c) for c in args.test_chroms.split(",")}
    print(f"[framing_b] val={sorted(val_chroms)} test={sorted(test_chroms)}")
    print(f"[framing_b] rf threshold: {args.rf_threshold} "
           f"(sites must exceed this to be 'sequence-competent')")

    feat = pd.read_csv(args.features_tsv, sep="\t",
                        usecols=["gene", "chrom", "center", "y"])
    feat["chrom_norm"] = feat["chrom"].apply(normalize_chrom)

    candidates_by_split = {"val": [], "test": []}
    n_skipped = {"val": 0, "test": 0}
    for split, chroms in [("val", val_chroms), ("test", test_chroms)]:
        split_feat = feat[feat["chrom_norm"].isin(chroms)]
        label_counts = split_feat.groupby("gene")["y"].agg(["sum", "count"])
        label_counts["n_canon"] = label_counts["sum"]
        label_counts["n_crypt"] = label_counts["count"] - label_counts["sum"]
        bothclass_genes = label_counts[(label_counts["n_canon"] >= 1) &
                                          (label_counts["n_crypt"] >= 1)].index
        print(f"[framing_b] {split}: {len(bothclass_genes):,} bothclass genes")

        for gene in bothclass_genes:
            cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
            sp_path = os.path.join(args.splicing_parquet_dir, f"{gene}.parquet")
            if not os.path.exists(cpa_path) or not os.path.exists(sp_path):
                n_skipped[split] += 1
                continue
            try:
                cpa = pd.read_parquet(cpa_path,
                                        columns=["window_start_genomic", args.cpa_model])
                cpa = cpa.rename(columns={args.cpa_model: "rf_prob"})
                sp = pd.read_parquet(sp_path,
                                       columns=["window_start_genomic", "splicing_prob"])
            except Exception:
                n_skipped[split] += 1
                continue
            gene_labels = split_feat[split_feat["gene"] == gene]
            cand_df = build_candidates_for_gene(gene, cpa, sp, gene_labels,
                                                   args.tolerance_nt)
            if cand_df is None or cand_df["is_canonical"].sum() == 0 or \
                (cand_df["is_canonical"] == 0).sum() == 0:
                n_skipped[split] += 1
                continue
            candidates_by_split[split].append(cand_df)

    val_df = pd.concat(candidates_by_split["val"], ignore_index=True) \
        if candidates_by_split["val"] else pd.DataFrame()
    test_df = pd.concat(candidates_by_split["test"], ignore_index=True) \
        if candidates_by_split["test"] else pd.DataFrame()
    print(f"\n[framing_b] val candidates: {len(val_df):,} rows from "
           f"{val_df['gene'].nunique() if len(val_df) > 0 else 0:,} genes "
           f"(skipped {n_skipped['val']:,})")
    print(f"[framing_b] test candidates: {len(test_df):,} rows from "
           f"{test_df['gene'].nunique() if len(test_df) > 0 else 0:,} genes "
           f"(skipped {n_skipped['test']:,})")

    val_df.to_csv(os.path.join(args.out_dir, "val_ccpcpa_candidates.tsv"),
                   sep="\t", index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_ccpcpa_candidates.tsv"),
                    sep="\t", index=False)

    if len(val_df) < 50 or len(test_df) < 50:
        print("[framing_b] ERROR: not enough candidates")
        return

    X_val = np.column_stack([safe_logit(val_df["rf_prob"]),
                                safe_logit(val_df["splicing_prob"])])
    y_val = val_df["is_canonical"].values
    clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X_val, y_val)
    w_intercept = float(clf.intercept_[0])
    w_rf = float(clf.coef_[0, 0])
    w_spl = float(clf.coef_[0, 1])
    print(f"\n[framing_b] combiner fit on val (n={len(val_df):,}):")
    print(f"    w_logit_rf     = {w_rf:+.4f}")
    print(f"    w_logit_splice = {w_spl:+.4f}")
    if abs(w_rf) > 1e-6:
        print(f"    ratio splice/rf = {w_spl/w_rf:+.4f}")

    test_df = test_df.copy()
    test_df["score_cpa_only"] = test_df["rf_prob"]
    test_df["score_splicing_only"] = test_df["splicing_prob"]
    test_df["score_product"] = test_df["rf_prob"] * test_df["splicing_prob"]
    X_test = np.column_stack([safe_logit(test_df["rf_prob"]),
                                 safe_logit(test_df["splicing_prob"])])
    test_df["score_learned"] = clf.predict_proba(X_test)[:, 1]

    T = args.rf_threshold
    competent_subsets = []
    n_dropped_no_competent_competitor = 0
    n_dropped_canonical_below_threshold = 0
    for g, sub in test_df.groupby("gene", sort=False):
        canon_rows = sub[sub["is_canonical"] == 1]
        crypt_rows = sub[sub["is_canonical"] == 0]
        if len(canon_rows) == 0:
            continue
        canon_competent = canon_rows[canon_rows["rf_prob"] >= T]
        if len(canon_competent) == 0:
            n_dropped_canonical_below_threshold += 1
            continue
        crypt_competent = crypt_rows[crypt_rows["rf_prob"] >= T]
        if len(crypt_competent) == 0:
            n_dropped_no_competent_competitor += 1
            continue
        keep_rows = pd.concat([canon_competent, crypt_competent])
        competent_subsets.append(keep_rows)
    similar_df = pd.concat(competent_subsets, ignore_index=True) \
        if competent_subsets else pd.DataFrame()
    print(f"\n[framing_b] sequence-competent subset (rf_prob >= {T}):")
    print(f"    {len(similar_df):,} candidates from "
           f"{similar_df['gene'].nunique() if len(similar_df) > 0 else 0:,} genes")
    print(f"    dropped: {n_dropped_canonical_below_threshold} genes "
           f"(canonical itself below threshold), "
           f"{n_dropped_no_competent_competitor} genes (no CCPCPA above threshold)")

    print(f"\n{'='*78}")
    print(f"FRAMING B RESULTS: per-gene P@1 by candidate set")
    print(f"{'='*78}")

    def report(name, df, score_cols):
        print(f"\n[{name}] {df['gene'].nunique():,} genes, "
               f"{len(df):,} candidate rows")
        comp_counts = df.groupby("gene").size() - 1
        print(f"  competitors per gene: median={comp_counts.median():.0f} "
               f"mean={comp_counts.mean():.1f} "
               f"min={comp_counts.min()} max={comp_counts.max()}")
        print(f"  {'scorer':<22s} {'hits':>6s} {'n_genes':>9s} {'P@1':>7s}")
        out = []
        for label, col in score_cols:
            hits, n, _ = per_gene_top1(df, col)
            p1 = hits / n if n > 0 else np.nan
            print(f"  {label:<22s} {hits:>6d} {n:>9d} {p1:>7.4f}")
            out.append({"candidate_set": name, "scorer": label,
                         "hits": hits, "n_genes": n, "p_at_1": p1})
        return out

    score_cols = [
        ("cpa_only", "score_cpa_only"),
        ("splicing_only", "score_splicing_only"),
        ("product", "score_product"),
        ("learned_combiner", "score_learned"),
    ]

    all_metrics = []
    all_metrics.extend(report("ALL CCPCPAs (canonical vs all cryptics)",
                                 test_df, score_cols))
    all_metrics.extend(report(f"SEQUENCE-COMPETENT CCPCPAs "
                                  f"(rf_prob >= {args.rf_threshold})",
                                 similar_df, score_cols))

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(args.out_dir, "framing_b_metrics.tsv"),
        sep="\t", index=False)

    print(f"\n{'='*78}")
    print(f"HEADLINE for framing B:")
    print(f"{'='*78}")
    if len(similar_df) > 0:
        n_genes_similar = similar_df["gene"].nunique()
        cpa_hits, _, _ = per_gene_top1(similar_df, "score_cpa_only")
        learned_hits, _, _ = per_gene_top1(similar_df, "score_learned")
        cpa_p1 = cpa_hits / n_genes_similar
        learned_p1 = learned_hits / n_genes_similar
        lift_pp = (learned_p1 - cpa_p1) * 100
        lift_rel = (learned_p1 / cpa_p1 - 1) * 100 if cpa_p1 > 0 else np.nan
        print(f"\n  Among {n_genes_similar:,} test genes with sequence-competent "
               f"CCPCPA competitors (rf_prob >= {args.rf_threshold}):")
        print(f"    CPA-only P@1:        {cpa_p1:.4f}  ({cpa_hits}/{n_genes_similar})")
        print(f"    Joint (learned) P@1: {learned_p1:.4f}  ({learned_hits}/{n_genes_similar})")
        print(f"    Absolute lift: {lift_pp:+.1f} pp")
        print(f"    Relative lift: {lift_rel:+.1f}%")

    print(f"\n{'='*78}")
    print(f"SENSITIVITY: P@1 at different rf_prob thresholds")
    print(f"{'='*78}")
    sens_rows = []
    print(f"  {'rf>=T':<8s} {'n_genes':>9s} {'cpa_P@1':>9s} {'joint_P@1':>11s} "
           f"{'lift_pp':>9s}")
    for T_alt in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        subsets = []
        for g, sub in test_df.groupby("gene", sort=False):
            canon_rows = sub[(sub["is_canonical"] == 1) & (sub["rf_prob"] >= T_alt)]
            crypt_rows = sub[(sub["is_canonical"] == 0) & (sub["rf_prob"] >= T_alt)]
            if len(canon_rows) == 0 or len(crypt_rows) == 0:
                continue
            subsets.append(pd.concat([canon_rows, crypt_rows]))
        if not subsets:
            continue
        sdf = pd.concat(subsets, ignore_index=True)
        n_g = sdf["gene"].nunique()
        if n_g < 5:
            continue
        cpa_h, _, _ = per_gene_top1(sdf, "score_cpa_only")
        learn_h, _, _ = per_gene_top1(sdf, "score_learned")
        cpa_p = cpa_h / n_g
        learn_p = learn_h / n_g
        lift = (learn_p - cpa_p) * 100
        print(f"  {T_alt:<8.2f} {n_g:>9d} {cpa_p:>9.4f} {learn_p:>11.4f} "
               f"{lift:>+9.2f}")
        sens_rows.append({"rf_threshold": T_alt, "n_genes": int(n_g),
                           "cpa_p_at_1": float(cpa_p),
                           "learned_p_at_1": float(learn_p),
                           "lift_pp": float(lift)})
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(
            os.path.join(args.out_dir, "threshold_sensitivity.tsv"),
            sep="\t", index=False)

    test_df.to_csv(os.path.join(args.out_dir, "test_scored.tsv"),
                    sep="\t", index=False)
    similar_df.to_csv(os.path.join(args.out_dir, "test_scored_similar_only.tsv"),
                       sep="\t", index=False)
    print(f"\n[framing_b] wrote:")
    print(f"  framing_b_metrics.tsv  test_scored.tsv  test_scored_similar_only.tsv")


if __name__ == "__main__":
    main()