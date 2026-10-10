#!/usr/bin/env python3
# for one labeled gene, run the scanner across all its labeled sites and compare
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from extract_splicing_features import (
    GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan,
    extract_one_site,
)
from scan_splicing_genomewide import (
    GeneScoreTracks, extract_features_from_tracks, apply_gene_baselines_to_axes,
    UPSTREAM_NT, DOWNSTREAM_NT,
)


def read_pc_genes_bed(path):
    df = pd.read_csv(path, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    df["gene"] = df["name"].astype(str)
    if df["gene"].str.contains("_").any():
        df["gene"] = df["gene"].str.split("_").str[-1]
    return df.drop_duplicates(subset="gene", keep="first")


def derive_7_axes(feats):
    out = dict(feats)
    bin_u1m = [feats.get(f"bin{i}_u1_mean", np.nan) for i in range(1, 7)]
    bin_me3m = [feats.get(f"bin{i}_me3_max", np.nan) for i in range(1, 7)]
    out["upstream_mean_u1"] = float(np.nanmean(bin_u1m)) \
        if np.isfinite(bin_u1m).any() else np.nan
    out["upstream_max_me3"] = float(np.nanmax(bin_me3m)) \
        if np.isfinite(bin_me3m).any() else np.nan
    out["term_3ss"] = feats.get("term_me3_top3mean_500_25", np.nan)
    out["local_me5"] = feats.get("bin6_me5_mean", np.nan)
    out["pas_prox_u1"] = feats.get("bin6_u1_mean", np.nan)
    out["dn5ss_has_strong"] = 1.0 if np.isfinite(
        feats.get("dn5ss_nearest_strong_dist", np.nan)) else 0.0
    out["interv_me5_max"] = feats.get("interv_me5_max_500",
                                        feats.get("interv_me5_max_1000", np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--gene_baselines", required=True)
    ap.add_argument("--model_joblib", required=True)
    args = ap.parse_args()

    print(f"\nValidating scanner on all labeled sites of {args.gene}")
    print("=" * 70)

    feat_df = pd.read_csv(args.features_tsv, sep="\t")
    gene_sites = feat_df[feat_df["gene"] == args.gene].copy().reset_index(drop=True)
    print(f"sites for {args.gene}: {len(gene_sites)} "
           f"({(gene_sites['y']==1).sum()} canonical, "
           f"{(gene_sites['y']==0).sum()} cryptic)")

    pc = read_pc_genes_bed(args.pc_genes_bed)
    gi = pc[pc["gene"] == args.gene].iloc[0]
    chrom = gi["chrom"]
    gene_start = int(gi["start"])
    gene_end = int(gi["end"])
    strand = gi["strand"]
    print(f"gene: {chrom}:{gene_start}-{gene_end} strand={strand}")

    base_df = pd.read_csv(args.gene_baselines, sep="\t")
    base_df = base_df.drop_duplicates(subset="gene", keep="first")
    gene_baseline = base_df[base_df["gene"] == args.gene].iloc[0].to_dict()

    bundle = joblib.load(args.model_joblib)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    fa = GenomeFasta(args.reference_fasta)
    u1_lookup = load_u1_lookup(args.u1_lookup)
    me5_lookup = load_me5_lookup(args.me5_lookup)
    _, score3, _, m3 = init_maxentscan()

    print(f"\nbuilding gene-wide tracks for scanner...")
    import time
    t0 = time.time()
    tracks = GeneScoreTracks(chrom, gene_start, gene_end, strand,
                               fa, u1_lookup, me5_lookup, score3)
    print(f"  track build time: {time.time()-t0:.1f}s, length: {tracks.length}")

    rows = []
    for i, site in gene_sites.iterrows():
        canon_pos = int(site["center"])
        y = int(site["y"])

        if strand == "+":
            win_start = canon_pos - UPSTREAM_NT
            win_end = canon_pos + DOWNSTREAM_NT
        else:
            win_start = canon_pos - DOWNSTREAM_NT
            win_end = canon_pos + UPSTREAM_NT
        seq_a = fa.fetch(chrom, win_start, win_end, strand=strand)
        if seq_a is None or len(seq_a) < 3500:
            p_a = np.nan
        else:
            feats_a = extract_one_site(seq_a, UPSTREAM_NT, DOWNSTREAM_NT, 8,
                                          me5_lookup, score3, m3, u1_lookup)
            feats_a = derive_7_axes(feats_a)
            feats_a = apply_gene_baselines_to_axes(feats_a, gene_baseline)
            x = np.array([[feats_a.get(c, np.nan) for c in feature_cols]],
                          dtype=np.float32)
            p_a = float(model.predict_proba(x)[0, 1])

        if strand == "+":
            gene_local = canon_pos - tracks.track_start
        else:
            gene_local = tracks.track_end - canon_pos
        window = tracks.slice_window(gene_local)
        if window is None:
            p_b = np.nan
        else:
            feats_b = extract_features_from_tracks(window)
            feats_b = apply_gene_baselines_to_axes(feats_b, gene_baseline)
            x = np.array([[feats_b.get(c, np.nan) for c in feature_cols]],
                          dtype=np.float32)
            p_b = float(model.predict_proba(x)[0, 1])

        x = np.array([[site.get(c, np.nan) for c in feature_cols]],
                      dtype=np.float32)
        p_c = float(model.predict_proba(x)[0, 1])

        rows.append({"pos": canon_pos, "y": y, "p_train": p_a, "p_scan": p_b,
                      "p_tsv": p_c, "delta_scan_vs_train": p_b - p_a,
                      "delta_scan_vs_tsv": p_b - p_c})

    rdf = pd.DataFrame(rows)
    print(f"\nper-site predictions:")
    print(rdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nsummary:")
    print(f"  mean |p_scan - p_train|: {rdf['delta_scan_vs_train'].abs().mean():.4f}")
    print(f"  mean |p_scan - p_tsv|:   {rdf['delta_scan_vs_tsv'].abs().mean():.4f}")
    print(f"  mean |p_train - p_tsv|:  {(rdf['p_train'] - rdf['p_tsv']).abs().mean():.4f}")

    print(f"\nwithin-gene ranking of canonical site:")
    for col in ["p_train", "p_scan", "p_tsv"]:
        ranked = rdf.sort_values(col, ascending=False).reset_index(drop=True)
        canon_idx = ranked[ranked["y"] == 1].index[0] + 1
        print(f"  by {col}: canonical at rank {canon_idx}/{len(ranked)} "
               f"(score={rdf[rdf['y']==1][col].iloc[0]:.4f})")


if __name__ == "__main__":
    main()