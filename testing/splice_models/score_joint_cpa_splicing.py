#!/usr/bin/env python3
# joint cpa × splicing scoring with ±tolerance_nt tolerance for the canonical position
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_chrom(c):
    if not isinstance(c, str):
        return c
    c = c.strip()
    if c.lower().startswith("chr"):
        return "chr" + c[3:].upper()
    return "chr" + c.upper()


def read_pc_genes_bed(path):
    df = pd.read_csv(path, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    df["gene"] = df["name"].astype(str)
    if df["gene"].str.contains("_").any():
        df["gene"] = df["gene"].str.split("_").str[-1]
    df["chrom_norm"] = df["chrom"].apply(normalize_chrom)
    return df.drop_duplicates(subset="gene", keep="first")


def expand_canonical_zones(per_gene_df, tolerance_nt):
    df = per_gene_df.copy()
    canonical_positions = df.loc[df["is_constitutive_cpa"] == True,
                                    "window_start_genomic"].values
    if len(canonical_positions) == 0:
        df["in_canonical_zone"] = False
        return df

    positions = df["window_start_genomic"].values
    in_zone = np.zeros(len(df), dtype=bool)
    for cp in canonical_positions:
        in_zone |= (np.abs(positions - cp) <= tolerance_nt)
    df["in_canonical_zone"] = in_zone
    return df


def per_gene_max_canonical_score(gene_df, score_col, tolerance_nt):
    canonical_positions = gene_df.loc[gene_df["is_constitutive_cpa"] == True,
                                         "window_start_genomic"].values
    if len(canonical_positions) == 0:
        return None

    positions = gene_df["window_start_genomic"].values
    scores = gene_df[score_col].values

    in_zone = np.zeros(len(gene_df), dtype=bool)
    for cp in canonical_positions:
        in_zone |= (np.abs(positions - cp) <= tolerance_nt)

    if not in_zone.any() or in_zone.all():
        return None

    canonical_scores = scores[in_zone]
    outside_scores = scores[~in_zone]

    canonical_scores = canonical_scores[np.isfinite(canonical_scores)]
    outside_scores = outside_scores[np.isfinite(outside_scores)]

    if len(canonical_scores) == 0 or len(outside_scores) == 0:
        return None

    return {
        "max_canonical": float(np.max(canonical_scores)),
        "max_outside": float(np.max(outside_scores)),
        "n_zone": int(in_zone.sum()),
        "n_outside": int(len(scores) - in_zone.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_parquet_dir", required=True)
    ap.add_argument("--splicing_parquet_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--features_tsv", required=True,
                     help="features_genebaselines.tsv — used to identify "
                          "genes that have labeled cryptic sites for the "
                          "within-gene canonical-vs-cryptic comparison")
    ap.add_argument("--test_chroms", default="chr18,chr19,chr20,chr21,chr22,chrX")
    ap.add_argument("--tolerance_nt", type=int, default=250)
    ap.add_argument("--cpa_model", default="rf_prob")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_chroms = {normalize_chrom(c.strip()) for c in args.test_chroms.split(",")}
    print(f"[joint] test chroms: {sorted(test_chroms)}")
    print(f"[joint] tolerance: ±{args.tolerance_nt} nt around canonical positions")

    pc = read_pc_genes_bed(args.pc_genes_bed)
    test_genes = sorted(pc.loc[pc["chrom_norm"].isin(test_chroms), "gene"].unique())
    print(f"[joint] {len(test_genes):,} genes on test chroms")

    feat_df = pd.read_csv(args.features_tsv, sep="\t",
                            usecols=["gene", "chrom", "center", "y"])
    cryptic_positions_per_gene = (feat_df[feat_df["y"] == 0]
                                    .groupby("gene")["center"]
                                    .apply(lambda s: s.values.tolist())
                                    .to_dict())
    print(f"[joint] cryptic labels: {len(cryptic_positions_per_gene):,} genes "
          f"have at least one cryptic site")

    all_pooled_rows = []
    per_gene_summary = []
    n_missing = 0
    for i, gene in enumerate(test_genes):
        cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
        splice_path = os.path.join(args.splicing_parquet_dir, f"{gene}.parquet")
        if not os.path.exists(cpa_path) or not os.path.exists(splice_path):
            n_missing += 1
            continue
        try:
            cpa = pd.read_parquet(cpa_path, columns=["window_start_genomic",
                                                       args.cpa_model,
                                                       "is_constitutive_cpa"])
            sp = pd.read_parquet(splice_path,
                                   columns=["window_start_genomic", "splicing_prob"])
            merged = cpa.merge(sp, on="window_start_genomic", how="inner")
            merged["gene"] = gene
            merged["joint_score"] = merged["splicing_prob"] * merged[args.cpa_model]

            merged = expand_canonical_zones(merged, args.tolerance_nt)

            merged_clean = merged.dropna(subset=[args.cpa_model, "splicing_prob"])
            all_pooled_rows.append(merged_clean)

            row = {"gene": gene, "n_sites_total": len(merged_clean)}
            for scorer_name, scorer_col in [("cpa", args.cpa_model),
                                              ("splicing", "splicing_prob"),
                                              ("joint", "joint_score")]:
                result = per_gene_max_canonical_score(merged_clean, scorer_col,
                                                       args.tolerance_nt)
                if result is None:
                    row[f"{scorer_name}_max_canonical"] = np.nan
                    row[f"{scorer_name}_max_outside"] = np.nan
                    row[f"{scorer_name}_canonical_wins"] = np.nan
                else:
                    row[f"{scorer_name}_max_canonical"] = result["max_canonical"]
                    row[f"{scorer_name}_max_outside"] = result["max_outside"]
                    row[f"{scorer_name}_canonical_wins"] = int(
                        result["max_canonical"] > result["max_outside"])
                    row["n_zone_positions"] = result["n_zone"]
                    row["n_outside_positions"] = result["n_outside"]
            row["has_labeled_cryptic"] = gene in cryptic_positions_per_gene
            per_gene_summary.append(row)
        except Exception as e:
            print(f"[joint] ERROR on {gene}: {e}")

        if (i + 1) % 500 == 0:
            print(f"[joint]   processed {i+1:,}/{len(test_genes):,} genes")

    if n_missing > 0:
        print(f"[joint] WARNING: {n_missing} genes missing parquet files")

    pooled_df = pd.concat(all_pooled_rows, ignore_index=True)
    summary_df = pd.DataFrame(per_gene_summary)
    print(f"\n[joint] pooled rows: {len(pooled_df):,}")
    print(f"[joint] per-gene summary: {len(summary_df):,} genes")
    print(f"[joint]   in canonical zone (positives): "
           f"{int(pooled_df['in_canonical_zone'].sum()):,}")
    print(f"[joint]   outside canonical zone (negatives): "
           f"{int((~pooled_df['in_canonical_zone']).sum()):,}")
    print(f"[joint]   positive class rate: "
           f"{float(pooled_df['in_canonical_zone'].mean()):.4f}")

    print(f"\n[joint] POOLED METRICS (canonical zone ± {args.tolerance_nt} nt)")
    print(f"  {'scorer':<14s} {'AUROC':>7s} {'AUPRC':>7s}")
    y_pool = pooled_df["in_canonical_zone"].values
    pooled_rows = []
    for scorer_name, scorer_col in [("cpa_only", args.cpa_model),
                                       ("splicing_only", "splicing_prob"),
                                       ("joint_product", "joint_score")]:
        s = pooled_df[scorer_col].values
        try:
            auc = roc_auc_score(y_pool, s)
            ap = average_precision_score(y_pool, s)
        except ValueError:
            auc, ap = np.nan, np.nan
        print(f"  {scorer_name:<14s} {auc:>7.4f} {ap:>7.4f}")
        pooled_rows.append({"scorer": scorer_name, "metric_type": "pooled",
                             "auroc": float(auc), "auprc": float(ap)})

    print(f"\n[joint] PER-GENE METRICS (max-in-zone vs max-outside-zone)")
    print(f"  {'scorer':<14s} {'P@1':>7s} {'n_genes':>9s}")
    per_gene_rows = []
    for scorer_name in ["cpa", "splicing", "joint"]:
        col = f"{scorer_name}_canonical_wins"
        valid = summary_df[col].dropna()
        p_at_1 = float(valid.mean()) if len(valid) > 0 else np.nan
        scorer_label = {"cpa": "cpa_only", "splicing": "splicing_only",
                          "joint": "joint_product"}[scorer_name]
        print(f"  {scorer_label:<14s} {p_at_1:>7.4f} {len(valid):>9d}")
        per_gene_rows.append({"scorer": scorer_label, "metric_type": "per_gene_all",
                               "p_at_1": p_at_1, "n_genes": int(len(valid))})

    crypt_subset = summary_df[summary_df["has_labeled_cryptic"] == True]
    print(f"\n[joint] PER-GENE METRICS — subset with labeled cryptic sites "
           f"({len(crypt_subset):,} genes)")
    print(f"  {'scorer':<14s} {'P@1':>7s} {'n_genes':>9s}")
    for scorer_name in ["cpa", "splicing", "joint"]:
        col = f"{scorer_name}_canonical_wins"
        valid = crypt_subset[col].dropna()
        p_at_1 = float(valid.mean()) if len(valid) > 0 else np.nan
        scorer_label = {"cpa": "cpa_only", "splicing": "splicing_only",
                          "joint": "joint_product"}[scorer_name]
        print(f"  {scorer_label:<14s} {p_at_1:>7.4f} {len(valid):>9d}")
        per_gene_rows.append({"scorer": scorer_label, "metric_type": "per_gene_bothclass",
                               "p_at_1": p_at_1, "n_genes": int(len(valid))})

    all_metrics = pd.DataFrame(pooled_rows + per_gene_rows)
    all_metrics.to_csv(os.path.join(args.out_dir, "joint_metrics_tolerance.tsv"),
                        sep="\t", index=False)
    summary_df.to_csv(os.path.join(args.out_dir, "per_gene_summary.tsv"),
                       sep="\t", index=False)
    print(f"\n[joint] wrote joint_metrics_tolerance.tsv and per_gene_summary.tsv")

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"cpa_only": "#888", "splicing_only": "#3E8C68", "joint_product": "#C23B3B"}
    for scorer_name, scorer_col in [("cpa_only", args.cpa_model),
                                       ("splicing_only", "splicing_prob"),
                                       ("joint_product", "joint_score")]:
        s = pooled_df[scorer_col].values
        try:
            fpr, tpr, _ = roc_curve(y_pool, s)
            prec, rec, _ = precision_recall_curve(y_pool, s)
            auc = roc_auc_score(y_pool, s)
            ap = average_precision_score(y_pool, s)
        except ValueError:
            continue
        ax_roc.plot(fpr, tpr, color=colors[scorer_name], lw=2,
                     label=f"{scorer_name} (AUC={auc:.3f})")
        ax_pr.plot(rec, prec, color=colors[scorer_name], lw=2,
                    label=f"{scorer_name} (AP={ap:.3f})")
    base_rate = float(y_pool.mean())
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax_pr.axhline(base_rate, color="k", linestyle="--", lw=1, alpha=0.5,
                   label=f"chance ({base_rate:.4f})")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC (±{} nt tolerance)".format(args.tolerance_nt),
                      fontweight="bold")
    ax_roc.legend(loc="lower right", frameon=False); ax_roc.grid(color="#EEE")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR (±{} nt tolerance)".format(args.tolerance_nt),
                     fontweight="bold")
    ax_pr.legend(loc="upper right", frameon=False); ax_pr.grid(color="#EEE")
    fig.suptitle(f"Joint CPA × splicing on test chroms — pooled "
                  f"({len(pooled_df):,} sites)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.out_dir, "figure_joint_roc_pr_tolerance.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[joint] wrote figure_joint_roc_pr_tolerance.png")


if __name__ == "__main__":
    main()