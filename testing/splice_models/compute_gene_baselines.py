#!/usr/bin/env python3
# per-gene splicing-context baselines for label-free within-gene centering
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from extract_splicing_features import (
    GenomeFasta, load_u1_lookup, load_me5_lookup,
    init_maxentscan, extract_one_site,
)

AXES = [
    "upstream_mean_u1",
    "upstream_max_me3",
    "term_3ss",
    "term_ppt",
    "local_me5",
    "pas_prox_u1",
    "dn5ss_has_strong",
    "interv_me5_max",
]


def compute_term_ppt(seq, upstream_nt, search_start=500, search_end=5,
                       window_size=30):
    if seq is None or len(seq) < upstream_nt:
        return np.nan
    region_start = max(0, upstream_nt - search_start)
    region_end = max(0, upstream_nt - search_end)
    if region_end <= region_start + window_size:
        return np.nan
    region = seq[region_start:region_end].upper()
    is_py = np.fromiter((1.0 if b in "CTU" else 0.0 for b in region),
                          dtype=np.float32, count=len(region))
    if len(is_py) < window_size:
        return np.nan
    window_sums = np.convolve(is_py,
                                 np.ones(window_size, dtype=np.float32),
                                 mode="valid")
    if window_sums.size == 0:
        return np.nan
    return float(window_sums.max() / window_size)


def derive_axes(feats, seq, upstream_nt):
    bin_u1_means = [feats.get(f"bin{i}_u1_mean", np.nan) for i in range(1, 7)]
    bin_me3_maxs = [feats.get(f"bin{i}_me3_max", np.nan) for i in range(1, 7)]
    return {
        "upstream_mean_u1": float(np.nanmean(bin_u1_means))
            if np.isfinite(bin_u1_means).any() else np.nan,
        "upstream_max_me3": float(np.nanmax(bin_me3_maxs))
            if np.isfinite(bin_me3_maxs).any() else np.nan,
        "term_3ss": feats.get("term_me3_top3mean_500_25", np.nan),
        "term_ppt": compute_term_ppt(seq, upstream_nt),
        "local_me5": feats.get("bin6_me5_mean", np.nan),
        "pas_prox_u1": feats.get("bin6_u1_mean", np.nan),
        "dn5ss_has_strong": 1.0 if np.isfinite(
            feats.get("dn5ss_nearest_strong_dist", np.nan)) else 0.0,
        "interv_me5_max": feats.get("interv_me5_max_500",
                                       feats.get("interv_me5_max_1000", np.nan)),
    }


def compute_one_gene_baseline(gene_row, reference_fasta, u1_lookup_path,
                                me5_lookup_path, upstream_nt, downstream_nt,
                                n_bins, n_samples, seed):
    chrom = gene_row["chrom"]
    start = int(gene_row["start"])
    end = int(gene_row["end"])
    strand = gene_row["strand"]
    gene_name = gene_row["gene"]

    fa = GenomeFasta(reference_fasta)
    u1_lookup = load_u1_lookup(u1_lookup_path)
    me5_lookup = load_me5_lookup(me5_lookup_path)
    _, score3, _, m3 = init_maxentscan()

    rng = np.random.default_rng(seed)

    margin = max(upstream_nt, downstream_nt)
    sample_lo = start + margin
    sample_hi = end - margin
    if sample_hi <= sample_lo:
        return {"gene": gene_name, "n_samples_used": 0,
                **{f"baseline_{ax}": np.nan for ax in AXES}}

    positions = rng.integers(sample_lo, sample_hi, size=n_samples)
    per_sample_rows = []
    for pos in positions:
        if strand == "+":
            win_start = pos - upstream_nt
            win_end = pos + downstream_nt
        else:
            win_start = pos - downstream_nt
            win_end = pos + upstream_nt
        win_start = max(0, win_start)
        seq = fa.fetch(chrom, win_start, win_end, strand=strand)
        if seq is None or len(seq) < (upstream_nt + downstream_nt - 100):
            continue
        feats = extract_one_site(seq, upstream_nt, downstream_nt, n_bins,
                                    me5_lookup, score3, m3, u1_lookup)
        axes = derive_axes(feats, seq, upstream_nt)
        per_sample_rows.append(axes)

    if len(per_sample_rows) == 0:
        return {"gene": gene_name, "n_samples_used": 0,
                **{f"baseline_{ax}": np.nan for ax in AXES}}

    df = pd.DataFrame(per_sample_rows)
    return {
        "gene": gene_name,
        "n_samples_used": len(per_sample_rows),
        **{f"baseline_{ax}": float(np.nanmean(df[ax])) for ax in AXES},
    }


def read_pc_genes_bed(path):
    df = pd.read_csv(path, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    df["gene"] = df["name"].astype(str)
    if df["gene"].str.contains("_").any():
        df["gene"] = df["gene"].str.split("_").str[-1]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_genes_bed", required=True,
                     help="BED6 of protein-coding gene bodies (one row per "
                          "gene with full gene-body coordinates).")
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--upstream_nt", type=int, default=3000)
    ap.add_argument("--downstream_nt", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--gene_limit", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    print(f"[baselines] reading genes from {args.pc_genes_bed}")
    genes_df = read_pc_genes_bed(args.pc_genes_bed)
    print(f"[baselines]   {len(genes_df):,} gene rows loaded")

    if args.gene_limit > 0:
        genes_df = genes_df.head(args.gene_limit).copy()
        print(f"[baselines]   limited to first {len(genes_df):,} genes")

    print(f"[baselines] window: -{args.upstream_nt} / +{args.downstream_nt} nt, "
           f"n_samples={args.n_samples} per gene, n_workers={args.n_workers}")
    print(f"[baselines] 8 axes (includes NEW term_ppt)")

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**31 - 1, size=len(genes_df))

    t0 = time.time()
    rows = []
    if args.n_workers <= 1:
        for i, ((_, gene_row), seed) in enumerate(zip(genes_df.iterrows(), seeds)):
            try:
                rows.append(compute_one_gene_baseline(
                    gene_row, args.reference_fasta, args.u1_lookup,
                    args.me5_lookup, args.upstream_nt, args.downstream_nt,
                    args.n_bins, args.n_samples, int(seed)))
            except Exception as e:
                print(f"[baselines]   ERROR on gene {gene_row['gene']}: {e}",
                       flush=True)
                rows.append({"gene": gene_row["gene"], "n_samples_used": 0,
                              **{f"baseline_{ax}": np.nan for ax in AXES}})
            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(genes_df) - i - 1) / max(rate, 1e-9)
                print(f"[baselines]   {i + 1:>6,}/{len(genes_df):>6,}  "
                       f"rate={rate:.1f} genes/s  eta={eta/60:.1f}m",
                       flush=True)
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise SystemExit("joblib required for parallel mode. "
                              "pip install joblib")
        print(f"[baselines] using {args.n_workers} joblib workers (loky)")
        args_list = [
            (gene_row, args.reference_fasta, args.u1_lookup,
             args.me5_lookup, args.upstream_nt, args.downstream_nt,
             args.n_bins, args.n_samples, int(seed))
            for (_, gene_row), seed in zip(genes_df.iterrows(), seeds)
        ]
        rows = Parallel(n_jobs=args.n_workers, backend="loky", verbose=10)(
            delayed(compute_one_gene_baseline)(*a) for a in args_list
        )

    out_dir = os.path.dirname(os.path.abspath(args.out_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    elapsed = time.time() - t0
    print(f"[baselines] done in {elapsed/60:.1f} min")
    print(f"[baselines] wrote {args.out_tsv}  "
           f"({len(out_df):,} rows, "
           f"{(out_df['n_samples_used'] > 0).sum():,} with usable samples)")

    ppt_col = "baseline_term_ppt"
    if ppt_col in out_df.columns:
        ppt_valid = out_df[ppt_col].dropna()
        if len(ppt_valid) > 0:
            print(f"[baselines] term_ppt baseline distribution across genes:")
            print(f"  mean={ppt_valid.mean():.3f}  median={ppt_valid.median():.3f}  "
                   f"std={ppt_valid.std():.3f}  "
                   f"range=[{ppt_valid.min():.3f}, {ppt_valid.max():.3f}]")
            print(f"  (random gene-body PPT max in 30nt windows; biologically "
                   f"reasonable values: ~0.55-0.75)")


if __name__ == "__main__":
    main()