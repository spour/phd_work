#!/usr/bin/env python3
# quantify how often rf alone correctly identifies the canonical cpa across the human
import argparse
import os
import numpy as np
import pandas as pd

TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
VAL_CHROMS = {"chr3", "chr13"}


def normalize_chrom(c):
    c = str(c).strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def find_peaks_nms(scores, positions, min_distance_nt, max_peaks):
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)
    finite = np.isfinite(scores)
    if not finite.any():
        return np.array([], dtype=int)
    order = np.argsort(scores)[::-1]
    chosen = []
    chosen_pos = []
    for idx in order:
        if not finite[idx]:
            continue
        p = positions[idx]
        if any(abs(p - cp) <= min_distance_nt for cp in chosen_pos):
            continue
        chosen.append(int(idx))
        chosen_pos.append(int(p))
        if len(chosen) >= max_peaks:
            break
    return np.array(chosen, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_parquet_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--max_peaks", type=int, default=20)
    ap.add_argument("--min_peak_distance", type=int, default=250)
    ap.add_argument("--tolerance_nt", type=int, default=250)
    ap.add_argument("--rf_threshold", type=float, default=0.5)
    ap.add_argument("--cpa_model", default="rf_prob")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[global] tolerance=±{args.tolerance_nt}nt, max_peaks={args.max_peaks}, "
           f"rf_threshold={args.rf_threshold}")

    pc = pd.read_csv(args.pc_genes_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    pc["gene"] = pc["name"].astype(str)
    if pc["gene"].str.contains("_").any():
        pc["gene"] = pc["gene"].str.split("_").str[-1]
    pc["chrom_norm"] = pc["chrom"].apply(normalize_chrom)
    pc = pc.drop_duplicates(subset="gene", keep="first")
    pc["split"] = "train"
    pc.loc[pc["chrom_norm"].isin(VAL_CHROMS), "split"] = "val"
    pc.loc[pc["chrom_norm"].isin(TEST_CHROMS), "split"] = "test"
    pc["gene_length"] = pc["end"] - pc["start"]
    print(f"[global] {len(pc):,} genes in BED")

    rows = []
    n_no_canon = 0
    n_skipped = 0
    all_genes = sorted(pc["gene"].unique())
    for i, gene in enumerate(all_genes):
        cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
        if not os.path.exists(cpa_path):
            n_skipped += 1
            continue
        try:
            cpa = pd.read_parquet(cpa_path,
                                    columns=["window_start_genomic",
                                              args.cpa_model,
                                              "is_constitutive_cpa"])
        except Exception:
            n_skipped += 1
            continue
        canonical_positions = cpa.loc[cpa["is_constitutive_cpa"] == True,
                                         "window_start_genomic"].values
        if len(canonical_positions) == 0:
            n_no_canon += 1
            continue
        positions = cpa["window_start_genomic"].values
        rf_scores = cpa[args.cpa_model].values
        finite_mask = np.isfinite(rf_scores)
        if not finite_mask.any():
            n_skipped += 1
            continue

        gi = pc[pc["gene"] == gene].iloc[0]

        rf_finite = rf_scores.copy()
        rf_finite[~finite_mask] = -np.inf
        s1_idx = int(np.argmax(rf_finite))
        s1_pos = int(positions[s1_idx])
        s1_dist = int(np.min(np.abs(canonical_positions - s1_pos)))
        s1_hit = int(s1_dist <= args.tolerance_nt)
        s1_rf = float(rf_scores[s1_idx])

        peak_idx = find_peaks_nms(rf_scores, positions,
                                     args.min_peak_distance, args.max_peaks)
        n_peaks = len(peak_idx)
        if n_peaks > 0:
            peak_top_idx = int(peak_idx[np.argmax(rf_scores[peak_idx])])
            p2_pos = int(positions[peak_top_idx])
            p2_dist = int(np.min(np.abs(canonical_positions - p2_pos)))
            p2_hit = int(p2_dist <= args.tolerance_nt)
            p2_rf = float(rf_scores[peak_top_idx])
        else:
            p2_pos = -1; p2_dist = -1; p2_hit = 0; p2_rf = np.nan

        if n_peaks > 0:
            peak_rf_vals = rf_scores[peak_idx]
            above = peak_idx[peak_rf_vals >= args.rf_threshold]
            n_peaks_above = len(above)
            if n_peaks_above > 0:
                above_top_idx = int(above[np.argmax(rf_scores[above])])
                p3_pos = int(positions[above_top_idx])
                p3_dist = int(np.min(np.abs(canonical_positions - p3_pos)))
                p3_hit = int(p3_dist <= args.tolerance_nt)
                p3_rf = float(rf_scores[above_top_idx])
            else:
                p3_pos = -1; p3_dist = -1; p3_hit = 0; p3_rf = np.nan
        else:
            n_peaks_above = 0
            p3_pos = -1; p3_dist = -1; p3_hit = 0; p3_rf = np.nan

        rows.append({
            "gene": gene, "chrom": gi["chrom"], "split": gi["split"],
            "gene_length": int(gi["gene_length"]),
            "n_canonical_annotated": len(canonical_positions),
            "n_rf_peaks": n_peaks,
            "n_rf_peaks_above_threshold": n_peaks_above,
            "stride1_top1_rf": s1_rf,
            "stride1_top1_dist": s1_dist,
            "stride1_top1_hit": s1_hit,
            "peaks_top1_rf": p2_rf,
            "peaks_top1_dist": p2_dist,
            "peaks_top1_hit": p2_hit,
            "peaks_thresh_top1_rf": p3_rf,
            "peaks_thresh_top1_dist": p3_dist,
            "peaks_thresh_top1_hit": p3_hit,
        })

        if (i + 1) % 2000 == 0:
            print(f"[global]   processed {i+1:,}/{len(all_genes):,} "
                   f"(scored: {len(rows):,}, no_canon: {n_no_canon:,}, "
                   f"skipped: {n_skipped:,})")

    df = pd.DataFrame(rows)
    print(f"\n[global] genes scored: {len(df):,}")
    print(f"[global]   splits: {df['split'].value_counts().to_dict()}")
    df.to_csv(os.path.join(args.out_dir, "per_gene_rf_canonical.tsv"),
                sep="\t", index=False)

    def report(name, sub):
        if len(sub) == 0:
            return None
        n = len(sub)
        s1 = float(sub["stride1_top1_hit"].mean())
        p2 = float(sub["peaks_top1_hit"].mean())
        p3 = float(sub["peaks_thresh_top1_hit"].mean())
        p3_subset = sub[sub["n_rf_peaks_above_threshold"] > 0]
        p3_applicable = (float(p3_subset["peaks_thresh_top1_hit"].mean())
                          if len(p3_subset) > 0 else np.nan)
        print(f"\n[{name}] n={n:,}")
        print(f"  Definition 1: stride-1 argmax of rf_prob")
        print(f"    P@1 = {s1:.4f}  ({int(sub['stride1_top1_hit'].sum()):,}/{n:,})")
        print(f"  Definition 2: RF peaks (top {args.max_peaks}, NMS {args.min_peak_distance}nt)")
        print(f"    P@1 = {p2:.4f}  ({int(sub['peaks_top1_hit'].sum()):,}/{n:,})")
        print(f"  Definition 3: RF peaks above rf_prob >= {args.rf_threshold}")
        print(f"    P@1 = {p3:.4f}  ({int(sub['peaks_thresh_top1_hit'].sum()):,}/{n:,}) "
               f"(includes genes with no above-threshold peak as misses)")
        print(f"    P@1 among genes with >=1 above-threshold peak = "
               f"{p3_applicable:.4f}  "
               f"({int(p3_subset['peaks_thresh_top1_hit'].sum()):,}/{len(p3_subset):,})"
               if len(p3_subset) > 0 else "    no applicable genes")
        return {
            "subset": name, "n_genes": n,
            "def1_stride1_p_at_1": s1,
            "def2_peaks_p_at_1": p2,
            "def3_peaks_thresh_p_at_1": p3,
            "def3_peaks_thresh_p_at_1_applicable": p3_applicable,
            "def3_n_applicable_genes": len(p3_subset),
        }

    print(f"\n{'='*78}")
    print(f"GLOBAL RF-ALONE CANONICAL P@1")
    print(f"{'='*78}")

    reports = []
    for name, sub in [
        ("WHOLE GENOME (all genes)", df),
        ("TRAIN chroms (model fit on these)", df[df["split"] == "train"]),
        ("VAL chroms", df[df["split"] == "val"]),
        ("TEST chroms (held-out, chr18-22, chrX)", df[df["split"] == "test"]),
    ]:
        r = report(name, sub)
        if r:
            reports.append(r)

    pd.DataFrame(reports).to_csv(
        os.path.join(args.out_dir, "global_rf_summary.tsv"),
        sep="\t", index=False)

    print(f"\n{'='*78}")
    print(f"STRATIFICATION by gene length (test chroms only)")
    print(f"{'='*78}")
    test_df = df[df["split"] == "test"].copy()
    length_bins = [0, 5000, 20000, 50000, 100000, 500000, 1e10]
    length_labels = ["<5kb", "5-20kb", "20-50kb", "50-100kb", "100-500kb",
                      ">500kb"]
    test_df["length_bin"] = pd.cut(test_df["gene_length"],
                                        bins=length_bins, labels=length_labels)

    print(f"  {'length':<10s} {'n_genes':>8s} {'def1_P@1':>9s} {'def2_P@1':>9s} "
           f"{'def3_P@1':>9s}")
    strat_rows = []
    for bin_val in length_labels:
        sub = test_df[test_df["length_bin"] == bin_val]
        if len(sub) == 0:
            continue
        s1 = float(sub["stride1_top1_hit"].mean())
        p2 = float(sub["peaks_top1_hit"].mean())
        p3 = float(sub["peaks_thresh_top1_hit"].mean())
        print(f"  {bin_val:<10s} {len(sub):>8d} {s1:>9.4f} {p2:>9.4f} "
               f"{p3:>9.4f}")
        strat_rows.append({"length_bin": bin_val, "n_genes": len(sub),
                            "def1_p_at_1": s1, "def2_p_at_1": p2,
                            "def3_p_at_1": p3})
    pd.DataFrame(strat_rows).to_csv(
        os.path.join(args.out_dir, "stratified_by_length.tsv"),
        sep="\t", index=False)

    print(f"\n{'='*78}")
    print(f"STRATIFICATION by number of RF peaks (test chroms only)")
    print(f"{'='*78}")
    peak_bins = [-0.5, 0, 1, 3, 5, 10, 100]
    peak_labels = ["0", "1", "2-3", "4-5", "6-10", ">10"]
    test_df["peak_bin"] = pd.cut(test_df["n_rf_peaks"], bins=peak_bins,
                                      labels=peak_labels)

    print(f"  {'n_peaks':<10s} {'n_genes':>8s} {'def2_P@1':>9s} {'def3_P@1':>9s}")
    for bin_val in peak_labels:
        sub = test_df[test_df["peak_bin"] == bin_val]
        if len(sub) == 0:
            continue
        p2 = float(sub["peaks_top1_hit"].mean())
        p3 = float(sub["peaks_thresh_top1_hit"].mean())
        print(f"  {bin_val:<10s} {len(sub):>8d} {p2:>9.4f} {p3:>9.4f}")

    print(f"\n[global] wrote per_gene_rf_canonical.tsv, global_rf_summary.tsv, "
           f"stratified_by_length.tsv")


if __name__ == "__main__":
    main()