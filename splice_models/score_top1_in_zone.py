#!/usr/bin/env python3
# logit-scale joint scoring
import argparse
import os
import numpy as np
import pandas as pd


TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}


def normalize_chrom(c):
    if not isinstance(c, str):
        return c
    c = c.strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def check_top1_in_zone(scores, positions, canonical_positions, tolerance_nt):
    if not np.isfinite(scores).any():
        return np.nan, np.nan
    top1_idx = int(np.nanargmax(scores))
    top1_pos = int(positions[top1_idx])
    dist = int(np.min(np.abs(canonical_positions - top1_pos)))
    return int(dist <= tolerance_nt), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_parquet_dir", required=True)
    ap.add_argument("--splicing_parquet_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--tolerance_nt", type=int, default=250)
    ap.add_argument("--cpa_model", default="rf_prob")
    ap.add_argument("--n_genes", type=int, default=1000,
                     help="Sample N random genes (stratified test/non-test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[logit] tolerance = ±{args.tolerance_nt} nt, n_genes = {args.n_genes}")

    pc = pd.read_csv(args.pc_genes_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    pc["gene"] = pc["name"].astype(str)
    if pc["gene"].str.contains("_").any():
        pc["gene"] = pc["gene"].str.split("_").str[-1]
    pc["chrom_norm"] = pc["chrom"].apply(normalize_chrom)
    pc = pc.drop_duplicates(subset="gene", keep="first")
    pc["is_test"] = pc["chrom_norm"].isin(TEST_CHROMS)

    print(f"[logit] scanning for genes with both CPA and splicing parquets...")
    available_genes = []
    for gene in pc["gene"].values:
        cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
        splice_path = os.path.join(args.splicing_parquet_dir, f"{gene}.parquet")
        if os.path.exists(cpa_path) and os.path.exists(splice_path):
            available_genes.append(gene)
    print(f"[logit]   {len(available_genes):,} genes have both parquets")

    rng = np.random.default_rng(args.seed)
    pc_avail = pc[pc["gene"].isin(available_genes)]
    test_avail = pc_avail[pc_avail["is_test"] == True]["gene"].values
    nontest_avail = pc_avail[pc_avail["is_test"] == False]["gene"].values
    n_each = args.n_genes // 2
    n_test = min(n_each, len(test_avail))
    n_nontest = min(args.n_genes - n_test, len(nontest_avail))
    test_sample = rng.choice(test_avail, size=n_test, replace=False)
    nontest_sample = rng.choice(nontest_avail, size=n_nontest, replace=False)
    sampled_genes = sorted(np.concatenate([test_sample, nontest_sample]))
    print(f"[logit] sampled {n_test:,} test-chrom genes and "
           f"{n_nontest:,} non-test genes = {len(sampled_genes):,} total")

    combiner_names = [
        "cpa_only", "splicing_only",
        "product",
        "logit_sum",
        "logit_cpa_07",
        "logit_cpa_03",
        "sum",
        "min",
        "max",
    ]

    rows = []
    for i, gene in enumerate(sampled_genes):
        cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
        splice_path = os.path.join(args.splicing_parquet_dir, f"{gene}.parquet")
        try:
            cpa = pd.read_parquet(cpa_path, columns=["window_start_genomic",
                                                       args.cpa_model,
                                                       "is_constitutive_cpa"])
            sp = pd.read_parquet(splice_path,
                                   columns=["window_start_genomic", "splicing_prob"])
        except Exception:
            continue
        canonical_positions = cpa.loc[cpa["is_constitutive_cpa"] == True,
                                         "window_start_genomic"].values
        if len(canonical_positions) == 0:
            continue
        merged = cpa.merge(sp, on="window_start_genomic", how="inner")
        merged = merged.dropna(subset=[args.cpa_model, "splicing_prob"])
        if len(merged) == 0:
            continue

        positions = merged["window_start_genomic"].values
        p_cpa = merged[args.cpa_model].values.astype(np.float64)
        p_spl = merged["splicing_prob"].values.astype(np.float64)

        l_cpa = safe_logit(p_cpa)
        l_spl = safe_logit(p_spl)

        scores_by_combiner = {
            "cpa_only": p_cpa,
            "splicing_only": p_spl,
            "product": p_cpa * p_spl,
            "logit_sum": l_cpa + l_spl,
            "logit_cpa_07": 0.7 * l_cpa + 0.3 * l_spl,
            "logit_cpa_03": 0.3 * l_cpa + 0.7 * l_spl,
            "sum": p_cpa + p_spl,
            "min": np.minimum(p_cpa, p_spl),
            "max": np.maximum(p_cpa, p_spl),
        }

        gene_info = pc[pc["gene"] == gene].iloc[0]
        row = {"gene": gene, "chrom": gene_info["chrom"],
                "is_test": bool(gene_info["is_test"]),
                "n_scan_positions": len(merged),
                "n_canonical": len(canonical_positions)}

        for cname, scores in scores_by_combiner.items():
            win, dist = check_top1_in_zone(scores, positions, canonical_positions,
                                              args.tolerance_nt)
            row[f"{cname}_in_zone"] = win
            row[f"{cname}_dist"] = dist
        rows.append(row)

        if (i + 1) % 200 == 0:
            print(f"[logit]   processed {i+1:,}/{len(sampled_genes):,} "
                   f"(scored: {len(rows):,})")

    df = pd.DataFrame(rows)
    print(f"\n[logit] {len(df):,} genes scored "
           f"({int(df['is_test'].sum()):,} test, "
           f"{int((~df['is_test']).sum()):,} non-test)")
    df.to_csv(os.path.join(args.out_dir, "per_gene_logit.tsv"),
                sep="\t", index=False)

    def report(subset_name, sub):
        if len(sub) == 0:
            return []
        print(f"\n[{subset_name}] {len(sub):,} genes")
        print(f"  {'combiner':<16s} {'P@1':>7s} {'median dist':>13s} {'mean dist':>13s}")
        out = []
        for cname in combiner_names:
            in_zone = sub[f"{cname}_in_zone"].dropna()
            if len(in_zone) == 0:
                continue
            p1 = float(in_zone.mean())
            med_dist = float(sub[f"{cname}_dist"].median())
            mean_dist = float(sub[f"{cname}_dist"].mean())
            print(f"  {cname:<16s} {p1:>7.4f} {med_dist:>13,.0f} {mean_dist:>13,.0f}")
            out.append({"subset": subset_name, "combiner": cname,
                         "n_genes": int(len(in_zone)), "p_at_1": p1,
                         "median_dist_nt": med_dist, "mean_dist_nt": mean_dist})
        return out

    all_metrics = []
    all_metrics.extend(report("test chroms", df[df["is_test"] == True]))
    all_metrics.extend(report("non-test chroms", df[df["is_test"] == False]))
    all_metrics.extend(report("all sampled genes", df))

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(args.out_dir, "logit_metrics.tsv"),
        sep="\t", index=False)
    print(f"\n[logit] wrote per_gene_logit.tsv and logit_metrics.tsv")


if __name__ == "__main__":
    main()