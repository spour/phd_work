#!/usr/bin/env python3
# augmented baseline computer that outputs both 7-axis baselines and 48 per-bin baselines
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
    "upstream_mean_u1", "upstream_max_me3", "term_3ss", "local_me5",
    "pas_prox_u1", "dn5ss_has_strong", "interv_me5_max",
]


def derive_7_axes(feats):
    bin_u1_means = [feats.get(f"bin{i}_u1_mean", np.nan) for i in range(1, 7)]
    bin_me3_maxs = [feats.get(f"bin{i}_me3_max", np.nan) for i in range(1, 7)]
    return {
        "upstream_mean_u1": float(np.nanmean(bin_u1_means))
            if np.isfinite(bin_u1_means).any() else np.nan,
        "upstream_max_me3": float(np.nanmax(bin_me3_maxs))
            if np.isfinite(bin_me3_maxs).any() else np.nan,
        "term_3ss": feats.get("term_me3_top3mean_500_25", np.nan),
        "local_me5": feats.get("bin6_me5_mean", np.nan),
        "pas_prox_u1": feats.get("bin6_u1_mean", np.nan),
        "dn5ss_has_strong": 1.0 if np.isfinite(
            feats.get("dn5ss_nearest_strong_dist", np.nan)) else 0.0,
        "interv_me5_max": feats.get("interv_me5_max_500",
                                     feats.get("interv_me5_max_1000", np.nan)),
    }


def derive_per_bin(feats, n_bins=8):
    out = {}
    for i in range(1, n_bins + 1):
        for motif in ("u1", "me3", "me5"):
            for stat in ("mean", "max"):
                col = f"bin{i}_{motif}_{stat}"
                out[col] = feats.get(col, np.nan)
    return out


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

    bin_cols = [f"bin{i}_{m}_{s}" for i in range(1, n_bins + 1)
                  for m in ("u1", "me3", "me5") for s in ("mean", "max")]
    empty_baseline = {f"baseline_{ax}": np.nan for ax in AXES}
    empty_baseline.update({f"baseline_{c}": np.nan for c in bin_cols})

    margin = max(upstream_nt, downstream_nt)
    sample_lo = start + margin
    sample_hi = end - margin
    if sample_hi <= sample_lo:
        return {"gene": gene_name, "n_samples_used": 0, **empty_baseline}

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
        per_sample_rows.append({**derive_7_axes(feats), **derive_per_bin(feats, n_bins)})

    if len(per_sample_rows) == 0:
        return {"gene": gene_name, "n_samples_used": 0, **empty_baseline}

    df_s = pd.DataFrame(per_sample_rows)
    return {
        "gene": gene_name,
        "n_samples_used": len(per_sample_rows),
        **{f"baseline_{ax}": float(np.nanmean(df_s[ax])) for ax in AXES},
        **{f"baseline_{c}": float(np.nanmean(df_s[c])) for c in bin_cols
            if c in df_s.columns},
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
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--upstream_nt", type=int, default=3000)
    ap.add_argument("--downstream_nt", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=30,
                     help="Random positions per gene. 30 is fine for the test; "
                          "50 if you want maximum precision.")
    ap.add_argument("--gene_limit", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    print(f"[baselines-v2] reading genes from {args.pc_genes_bed}")
    genes_df = read_pc_genes_bed(args.pc_genes_bed)
    print(f"[baselines-v2]   {len(genes_df):,} gene rows")

    if args.gene_limit > 0:
        genes_df = genes_df.head(args.gene_limit).copy()
        print(f"[baselines-v2]   limited to {len(genes_df):,} genes")

    print(f"[baselines-v2] window: -{args.upstream_nt}/+{args.downstream_nt}, "
           f"n_samples={args.n_samples}, n_workers={args.n_workers}")

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
                print(f"[baselines-v2] ERROR on {gene_row['gene']}: {e}", flush=True)
            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(genes_df) - i - 1) / max(rate, 1e-9)
                print(f"[baselines-v2] {i + 1:>6,}/{len(genes_df):>6,}  "
                       f"rate={rate:.1f} g/s  eta={eta/60:.1f}m", flush=True)
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise SystemExit("joblib required for parallel mode")
        print(f"[baselines-v2] using {args.n_workers} joblib workers (loky)")
        args_list = [
            (gene_row, args.reference_fasta, args.u1_lookup, args.me5_lookup,
             args.upstream_nt, args.downstream_nt, args.n_bins,
             args.n_samples, int(seed))
            for (_, gene_row), seed in zip(genes_df.iterrows(), seeds)
        ]
        rows = Parallel(n_jobs=args.n_workers, backend="loky", verbose=10)(
            delayed(compute_one_gene_baseline)(*a) for a in args_list)

    out_dir = os.path.dirname(os.path.abspath(args.out_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    elapsed = time.time() - t0
    n_axis_cols = sum(1 for c in out_df.columns if c.startswith("baseline_")
                       and any(ax in c for ax in AXES))
    n_bin_cols = sum(1 for c in out_df.columns if c.startswith("baseline_bin"))
    print(f"[baselines-v2] done in {elapsed/60:.1f} min")
    print(f"[baselines-v2] wrote {args.out_tsv}: {len(out_df):,} rows, "
           f"{n_axis_cols} axis baselines + {n_bin_cols} per-bin baselines")
    print(f"[baselines-v2]   genes with usable samples: "
           f"{(out_df['n_samples_used'] > 0).sum():,}")


if __name__ == "__main__":
    main()