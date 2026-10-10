#!/usr/bin/env python3
# extend the framing b claim to all genes with an annotated canonical cpa
import argparse
import os
import numpy as np
import pandas as pd


TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
VAL_CHROMS = {"chr3", "chr13"}
EPS = 1e-6


def normalize_chrom(c):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_parquet_dir", required=True)
    ap.add_argument("--splicing_parquet_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--combiner_w_rf", type=float, required=True,
                     help="Learned weight on logit(rf_prob) from framing B")
    ap.add_argument("--combiner_w_splice", type=float, required=True,
                     help="Learned weight on logit(splicing_prob)")
    ap.add_argument("--combiner_intercept", type=float, default=0.0)
    ap.add_argument("--max_peaks", type=int, default=20,
                     help="Max RF peaks per gene")
    ap.add_argument("--min_peak_distance", type=int, default=250,
                     help="Min distance between RF peaks (nt)")
    ap.add_argument("--tolerance_nt", type=int, default=250,
                     help="Hit if top-1 within this distance of canonical")
    ap.add_argument("--cpa_model", default="rf_prob")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[wg] max_peaks={args.max_peaks}, min_peak_dist={args.min_peak_distance}nt, "
           f"tolerance=±{args.tolerance_nt}nt")
    print(f"[wg] combiner: {args.combiner_w_rf:+.4f}*logit_rf + "
           f"{args.combiner_w_splice:+.4f}*logit_splice + {args.combiner_intercept:+.4f}")

    pc = pd.read_csv(args.pc_genes_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    pc["gene"] = pc["name"].astype(str)
    if pc["gene"].str.contains("_").any():
        pc["gene"] = pc["gene"].str.split("_").str[-1]
    pc["chrom_norm"] = pc["chrom"].apply(normalize_chrom)
    pc = pc.drop_duplicates(subset="gene", keep="first")
    pc["is_test"] = pc["chrom_norm"].isin(TEST_CHROMS)
    pc["is_val"] = pc["chrom_norm"].isin(VAL_CHROMS)
    pc["split"] = "train"
    pc.loc[pc["is_test"], "split"] = "test"
    pc.loc[pc["is_val"], "split"] = "val"
    print(f"[wg] {len(pc):,} genes in BED")
    print(f"[wg] splits: {pc['split'].value_counts().to_dict()}")

    rows = []
    n_skipped = 0
    n_no_canon = 0
    n_no_splice = 0
    n_no_peaks = 0
    all_genes = sorted(pc["gene"].unique())
    for i, gene in enumerate(all_genes):
        cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
        splice_path = os.path.join(args.splicing_parquet_dir, f"{gene}.parquet")
        if not os.path.exists(cpa_path):
            n_skipped += 1
            continue
        try:
            cpa = pd.read_parquet(cpa_path, columns=["window_start_genomic",
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

        if not os.path.exists(splice_path):
            n_no_splice += 1
            merged = cpa.dropna(subset=[args.cpa_model])
            has_splice = False
        else:
            try:
                sp = pd.read_parquet(splice_path,
                                       columns=["window_start_genomic", "splicing_prob"])
                merged = cpa.merge(sp, on="window_start_genomic", how="inner")
                merged = merged.dropna(subset=[args.cpa_model, "splicing_prob"])
                has_splice = True
            except Exception:
                n_skipped += 1
                continue
        if len(merged) == 0:
            n_skipped += 1
            continue

        positions = merged["window_start_genomic"].values
        rf_scores = merged[args.cpa_model].values

        peak_idx = find_peaks_nms(rf_scores, positions, args.min_peak_distance,
                                     args.max_peaks)
        if len(peak_idx) == 0:
            n_no_peaks += 1
            continue

        peak_positions = positions[peak_idx]
        peak_rf = rf_scores[peak_idx]

        gene_info = pc[pc["gene"] == gene].iloc[0]
        row = {"gene": gene, "chrom": gene_info["chrom"],
                "split": gene_info["split"],
                "n_canonical_annotated": len(canonical_positions),
                "n_rf_peaks": len(peak_idx),
                "has_splice": has_splice}

        top1_idx = int(np.argmax(peak_rf))
        top1_pos = int(peak_positions[top1_idx])
        dist = int(np.min(np.abs(canonical_positions - top1_pos)))
        row["cpa_only_top1_pos"] = top1_pos
        row["cpa_only_top1_dist"] = dist
        row["cpa_only_top1_in_zone"] = int(dist <= args.tolerance_nt)

        if has_splice:
            peak_splice = merged["splicing_prob"].values[peak_idx]

            top1_idx = int(np.argmax(peak_splice))
            top1_pos = int(peak_positions[top1_idx])
            dist = int(np.min(np.abs(canonical_positions - top1_pos)))
            row["splicing_only_top1_pos"] = top1_pos
            row["splicing_only_top1_dist"] = dist
            row["splicing_only_top1_in_zone"] = int(dist <= args.tolerance_nt)

            prod = peak_rf * peak_splice
            top1_idx = int(np.argmax(prod))
            top1_pos = int(peak_positions[top1_idx])
            dist = int(np.min(np.abs(canonical_positions - top1_pos)))
            row["product_top1_pos"] = top1_pos
            row["product_top1_dist"] = dist
            row["product_top1_in_zone"] = int(dist <= args.tolerance_nt)

            combined = (args.combiner_intercept +
                         args.combiner_w_rf * safe_logit(peak_rf) +
                         args.combiner_w_splice * safe_logit(peak_splice))
            top1_idx = int(np.argmax(combined))
            top1_pos = int(peak_positions[top1_idx])
            dist = int(np.min(np.abs(canonical_positions - top1_pos)))
            row["learned_combiner_top1_pos"] = top1_pos
            row["learned_combiner_top1_dist"] = dist
            row["learned_combiner_top1_in_zone"] = int(dist <= args.tolerance_nt)
        else:
            for sc in ["splicing_only", "product", "learned_combiner"]:
                row[f"{sc}_top1_in_zone"] = np.nan
                row[f"{sc}_top1_dist"] = np.nan

        rows.append(row)

        if (i + 1) % 2000 == 0:
            print(f"[wg]   processed {i+1:,}/{len(all_genes):,} "
                   f"(scored: {len(rows):,}, no_canon: {n_no_canon:,}, "
                   f"no_splice: {n_no_splice:,}, no_peaks: {n_no_peaks:,})")

    df = pd.DataFrame(rows)
    print(f"\n[wg] genes scored: {len(df):,}")
    print(f"[wg]   with splice available: {int(df['has_splice'].sum()):,}")
    print(f"[wg]   splits: {df['split'].value_counts().to_dict()}")
    df.to_csv(os.path.join(args.out_dir, "per_gene_top1.tsv"), sep="\t", index=False)

    def report(name, sub):
        if len(sub) == 0:
            return []
        print(f"\n[{name}] {len(sub):,} genes")
        print(f"  {'scorer':<22s} {'n_valid':>9s} {'P@1':>7s} "
               f"{'median_dist':>12s} {'mean_dist':>11s}")
        out = []
        for scorer in ["cpa_only", "splicing_only", "product", "learned_combiner"]:
            col_zone = f"{scorer}_top1_in_zone"
            col_dist = f"{scorer}_top1_dist"
            valid = sub[col_zone].dropna()
            if len(valid) == 0:
                continue
            p1 = float(valid.mean())
            med_dist = float(sub[col_dist].median())
            mean_dist = float(sub[col_dist].mean())
            print(f"  {scorer:<22s} {int(len(valid)):>9d} {p1:>7.4f} "
                   f"{med_dist:>12,.0f} {mean_dist:>11,.0f}")
            out.append({"subset": name, "scorer": scorer,
                         "n_genes": int(len(valid)), "p_at_1": p1,
                         "median_dist_nt": med_dist,
                         "mean_dist_nt": mean_dist})
        return out

    all_metrics = []
    all_metrics.extend(report("WHOLE GENOME (all genes)", df))
    all_metrics.extend(report("WHOLE GENOME (splice available)",
                                 df[df["has_splice"] == True]))
    all_metrics.extend(report("TEST chroms (chr18-22, chrX)",
                                 df[df["split"] == "test"]))
    all_metrics.extend(report("TEST chroms (splice available)",
                                 df[(df["split"] == "test") & (df["has_splice"])]))
    all_metrics.extend(report("TRAIN+VAL chroms (splice available)",
                                 df[(df["split"] != "test") & (df["has_splice"])]))

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(args.out_dir, "framing_b_genome_metrics.tsv"),
        sep="\t", index=False)
    print(f"\n[wg] wrote per_gene_top1.tsv and framing_b_genome_metrics.tsv")


if __name__ == "__main__":
    main()