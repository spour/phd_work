#!/usr/bin/env python3
# variant of test_framing_b.py where competitors are rf-only peaks (not 3-way ccpcpas)
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
EPS = 1e-6


def normalize_chrom(c):
    c = str(c).strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def find_peaks_nms(scores, positions, min_distance_nt, max_peaks, rf_threshold=None):
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)
    finite = np.isfinite(scores)
    if rf_threshold is not None:
        finite &= scores >= rf_threshold
    if not finite.any():
        return np.array([], dtype=int)
    order = np.argsort(scores)[::-1]
    chosen = []
    chosen_positions = []
    for idx in order:
        if not finite[idx]:
            continue
        p = positions[idx]
        if any(abs(p - cp) <= min_distance_nt for cp in chosen_positions):
            continue
        chosen.append(int(idx))
        chosen_positions.append(int(p))
        if len(chosen) >= max_peaks:
            break
    return np.array(chosen, dtype=int)


def build_rf_peak_candidates(gene, cpa_df, sp_df, canonical_positions,
                               rf_threshold, max_peaks, min_peak_distance,
                               tolerance_nt, cpa_model):
    merged = cpa_df.merge(sp_df, on="window_start_genomic", how="inner")
    merged = merged.dropna(subset=[cpa_model, "splicing_prob"])
    if len(merged) == 0:
        return None
    positions = merged["window_start_genomic"].values
    rf_scores = merged[cpa_model].values
    splice_scores = merged["splicing_prob"].values

    peak_idx = find_peaks_nms(rf_scores, positions, min_peak_distance,
                                 max_peaks, rf_threshold=rf_threshold)
    if len(peak_idx) == 0:
        return None
    peak_positions = positions[peak_idx]
    peak_rf = rf_scores[peak_idx]
    peak_splice = splice_scores[peak_idx]

    in_zone = np.zeros(len(peak_idx), dtype=bool)
    for cp in canonical_positions:
        in_zone |= np.abs(peak_positions - cp) <= tolerance_nt

    return pd.DataFrame({
        "gene": gene,
        "position": peak_positions,
        "rf_prob": peak_rf,
        "splicing_prob": peak_splice,
        "is_canonical": in_zone.astype(int),
    })


def per_gene_top1(df, score_col):
    hits = 0
    n = 0
    for g, sub in df.groupby("gene", sort=False):
        if (sub["is_canonical"] == 1).sum() == 0:
            continue
        n += 1
        top_idx = sub[score_col].idxmax()
        hits += int(sub.loc[top_idx, "is_canonical"] == 1)
    return hits, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_parquet_dir", required=True)
    ap.add_argument("--splicing_parquet_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--test_chroms", default="chr18,chr19,chr20,chr21,chr22,chrX")
    ap.add_argument("--val_chroms", default="chr3,chr13")
    ap.add_argument("--rf_threshold", type=float, default=0.5,
                     help="Minimum rf_prob for a peak to be 'sequence-competent'")
    ap.add_argument("--max_peaks", type=int, default=50,
                     help="Maximum RF peaks per gene")
    ap.add_argument("--min_peak_distance", type=int, default=250)
    ap.add_argument("--tolerance_nt", type=int, default=250)
    ap.add_argument("--cpa_model", default="rf_prob")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    val_chroms = {normalize_chrom(c) for c in args.val_chroms.split(",")}
    test_chroms = {normalize_chrom(c) for c in args.test_chroms.split(",")}
    print(f"[rf_peaks] val={sorted(val_chroms)} test={sorted(test_chroms)}")
    print(f"[rf_peaks] rf_threshold={args.rf_threshold}, max_peaks={args.max_peaks}, "
           f"min_peak_distance={args.min_peak_distance}nt, tolerance=±{args.tolerance_nt}nt")

    pc = pd.read_csv(args.pc_genes_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    pc["gene"] = pc["name"].astype(str)
    if pc["gene"].str.contains("_").any():
        pc["gene"] = pc["gene"].str.split("_").str[-1]
    pc["chrom_norm"] = pc["chrom"].apply(normalize_chrom)
    pc = pc.drop_duplicates(subset="gene", keep="first")
    pc["split"] = "train"
    pc.loc[pc["chrom_norm"].isin(val_chroms), "split"] = "val"
    pc.loc[pc["chrom_norm"].isin(test_chroms), "split"] = "test"

    candidates_by_split = {"val": [], "test": []}
    n_skipped = {"val": 0, "test": 0}
    n_no_canon = {"val": 0, "test": 0}
    n_no_peaks = {"val": 0, "test": 0}

    for i, row in pc.iterrows():
        split = row["split"]
        if split not in candidates_by_split:
            continue
        gene = row["gene"]
        cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
        sp_path = os.path.join(args.splicing_parquet_dir, f"{gene}.parquet")
        if not os.path.exists(cpa_path) or not os.path.exists(sp_path):
            n_skipped[split] += 1
            continue
        try:
            cpa = pd.read_parquet(cpa_path, columns=["window_start_genomic",
                                                       args.cpa_model,
                                                       "is_constitutive_cpa"])
            sp = pd.read_parquet(sp_path, columns=["window_start_genomic",
                                                       "splicing_prob"])
        except Exception:
            n_skipped[split] += 1
            continue

        canonical_positions = cpa.loc[cpa["is_constitutive_cpa"] == True,
                                         "window_start_genomic"].values
        if len(canonical_positions) == 0:
            n_no_canon[split] += 1
            continue

        cand_df = build_rf_peak_candidates(
            gene, cpa, sp, canonical_positions,
            args.rf_threshold, args.max_peaks, args.min_peak_distance,
            args.tolerance_nt, args.cpa_model)
        if cand_df is None or len(cand_df) == 0:
            n_no_peaks[split] += 1
            continue
        if (cand_df["is_canonical"] == 1).sum() == 0 or \
            (cand_df["is_canonical"] == 0).sum() == 0:
            n_skipped[split] += 1
            continue
        candidates_by_split[split].append(cand_df)

        if (i + 1) % 2000 == 0:
            print(f"[rf_peaks]   processed {i+1:,} genes "
                   f"(val: {len(candidates_by_split['val']):,}, "
                   f"test: {len(candidates_by_split['test']):,})")

    val_df = pd.concat(candidates_by_split["val"], ignore_index=True) \
        if candidates_by_split["val"] else pd.DataFrame()
    test_df = pd.concat(candidates_by_split["test"], ignore_index=True) \
        if candidates_by_split["test"] else pd.DataFrame()
    print(f"\n[rf_peaks] val candidates: {len(val_df):,} rows from "
           f"{val_df['gene'].nunique() if len(val_df) > 0 else 0:,} genes")
    print(f"[rf_peaks] test candidates: {len(test_df):,} rows from "
           f"{test_df['gene'].nunique() if len(test_df) > 0 else 0:,} genes")
    for split in ["val", "test"]:
        print(f"  {split} skipped: {n_skipped[split]:,}  "
               f"no_canon: {n_no_canon[split]:,}  no_peaks: {n_no_peaks[split]:,}")

    val_df.to_csv(os.path.join(args.out_dir, "val_rf_peak_candidates.tsv"),
                   sep="\t", index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_rf_peak_candidates.tsv"),
                    sep="\t", index=False)

    if len(val_df) < 50 or len(test_df) < 50:
        print("[rf_peaks] ERROR: not enough candidates")
        return

    X_val = np.column_stack([safe_logit(val_df["rf_prob"]),
                                safe_logit(val_df["splicing_prob"])])
    y_val = val_df["is_canonical"].values
    clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X_val, y_val)
    w_int = float(clf.intercept_[0])
    w_rf = float(clf.coef_[0, 0])
    w_spl = float(clf.coef_[0, 1])
    print(f"\n[rf_peaks] combiner fit on val (n={len(val_df):,}):")
    print(f"    intercept      = {w_int:+.4f}")
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

    print(f"\n{'='*78}")
    print(f"RESULTS: RF-PEAK candidates, rf_prob >= {args.rf_threshold}")
    print(f"{'='*78}")
    print(f"\n[TEST] {test_df['gene'].nunique():,} genes, {len(test_df):,} peaks")
    comp_counts = test_df.groupby("gene").size() - 1
    print(f"  competitors per gene: median={comp_counts.median():.0f} "
           f"mean={comp_counts.mean():.1f} min={comp_counts.min()} max={comp_counts.max()}")
    print(f"  {'scorer':<22s} {'hits':>6s} {'n_genes':>9s} {'P@1':>7s}")
    score_cols = [("cpa_only", "score_cpa_only"),
                   ("splicing_only", "score_splicing_only"),
                   ("product", "score_product"),
                   ("learned_combiner", "score_learned")]
    out_rows = []
    for label, col in score_cols:
        hits, n = per_gene_top1(test_df, col)
        p1 = hits / n if n > 0 else np.nan
        print(f"  {label:<22s} {hits:>6d} {n:>9d} {p1:>7.4f}")
        out_rows.append({"scorer": label, "hits": hits, "n_genes": n, "p_at_1": p1})

    pd.DataFrame(out_rows).to_csv(os.path.join(args.out_dir,
                                                   "framing_b_rf_peaks_metrics.tsv"),
                                     sep="\t", index=False)

    print(f"\n{'='*78}")
    print(f"SENSITIVITY: P@1 at different rf_prob thresholds (test only)")
    print(f"{'='*78}")
    print(f"  {'rf>=T':<8s} {'n_genes':>9s} {'cpa_P@1':>9s} {'joint_P@1':>11s} "
           f"{'lift_pp':>9s}")
    sens_rows = []
    for T in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sub = test_df[test_df["rf_prob"] >= T]
        gene_has_canon = sub.groupby("gene")["is_canonical"].sum() > 0
        gene_has_crypt = sub.groupby("gene").size() - sub.groupby("gene")["is_canonical"].sum() > 0
        valid_genes = gene_has_canon[gene_has_canon & gene_has_crypt].index
        sub = sub[sub["gene"].isin(valid_genes)]
        if sub["gene"].nunique() < 5:
            continue
        cpa_h, _ = per_gene_top1(sub, "score_cpa_only")
        learn_h, ng = per_gene_top1(sub, "score_learned")
        cpa_p = cpa_h / ng
        learn_p = learn_h / ng
        lift = (learn_p - cpa_p) * 100
        print(f"  {T:<8.2f} {ng:>9d} {cpa_p:>9.4f} {learn_p:>11.4f} {lift:>+9.2f}")
        sens_rows.append({"rf_threshold": T, "n_genes": int(ng),
                           "cpa_p_at_1": float(cpa_p),
                           "learned_p_at_1": float(learn_p),
                           "lift_pp": float(lift)})
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(
            os.path.join(args.out_dir, "threshold_sensitivity.tsv"),
            sep="\t", index=False)

    print(f"\n[rf_peaks] wrote test_rf_peak_candidates.tsv, "
           f"framing_b_rf_peaks_metrics.tsv, threshold_sensitivity.tsv")


if __name__ == "__main__":
    main()