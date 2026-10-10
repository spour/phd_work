#!/usr/bin/env python3
# for one gene's canonical site, compute features using both the training pipeline
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


def main():
    print(f"\n{'='*70}")
    print(f"DEBUG: scanner features vs training features")
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

    print(f"\n{'='*70}")
    print(f"DEBUG: {args.gene} — scanner features vs training features")
    print(f"{'='*70}\n")

    print(f"[debug] loading training features TSV")
    feat_df = pd.read_csv(args.features_tsv, sep="\t")
    gene_sites = feat_df[feat_df["gene"] == args.gene]
    print(f"[debug]   {args.gene} has {len(gene_sites)} labeled sites in TSV")
    if len(gene_sites) == 0:
        print(f"[debug] ERROR: gene not in features TSV")
        return
    print(f"[debug]   labels: {gene_sites['y'].value_counts().to_dict()}")

    canon = gene_sites[gene_sites["y"] == 1]
    if len(canon) == 0:
        print(f"[debug] ERROR: no canonical site for {args.gene}")
        return
    canon_row = canon.iloc[0]
    canon_pos = int(canon_row["center"])
    chrom = canon_row["chrom"]
    print(f"[debug] canonical site: {chrom}:{canon_pos}")
    print(f"[debug]   strand from TSV: {canon_row.get('strand', '?')}")

    pc = read_pc_genes_bed(args.pc_genes_bed)
    gene_info = pc[pc["gene"] == args.gene]
    if len(gene_info) == 0:
        print(f"[debug] ERROR: gene not in pc_genes.bed")
        return
    gi = gene_info.iloc[0]
    gene_start = int(gi["start"])
    gene_end = int(gi["end"])
    strand = gi["strand"]
    print(f"[debug] gene info from BED: {gi['chrom']}:{gene_start}-{gene_end} strand={strand}")

    if str(gi["chrom"]).strip() != str(chrom).strip():
        print(f"[debug] CHROM MISMATCH: BED={gi['chrom']} vs TSV={chrom}")
    if canon_pos < gene_start or canon_pos > gene_end:
        print(f"[debug] WARNING: canonical pos {canon_pos} outside gene body "
               f"{gene_start}-{gene_end}")

    print(f"\n[debug] loading gene baselines")
    base_df = pd.read_csv(args.gene_baselines, sep="\t")
    base_df = base_df.drop_duplicates(subset="gene", keep="first")
    if args.gene not in set(base_df["gene"]):
        print(f"[debug] ERROR: no baseline for {args.gene}")
        return
    gene_baseline = base_df[base_df["gene"] == args.gene].iloc[0].to_dict()
    print(f"[debug]   baseline n_samples_used: {gene_baseline.get('n_samples_used')}")

    print(f"\n[debug] loading FASTA + score lookups + MaxEntScan")
    fa = GenomeFasta(args.reference_fasta)
    u1_lookup = load_u1_lookup(args.u1_lookup)
    me5_lookup = load_me5_lookup(args.me5_lookup)
    _, score3, _, m3 = init_maxentscan()

    print(f"\n{'─'*70}")
    print(f"METHOD A: training pipeline (extract_one_site at canonical pos)")
    print(f"{'─'*70}")

    if strand == "+":
        win_start = canon_pos - UPSTREAM_NT
        win_end = canon_pos + DOWNSTREAM_NT
    else:
        win_start = canon_pos - DOWNSTREAM_NT
        win_end = canon_pos + UPSTREAM_NT
    print(f"[A] window genomic coords: {win_start}-{win_end} (length {win_end-win_start})")
    seq_a = fa.fetch(chrom, win_start, win_end, strand=strand)
    print(f"[A] fetched seq length: {len(seq_a) if seq_a else 'None'}")
    if seq_a is None or len(seq_a) < 3500:
        print(f"[A] ERROR: bad sequence")
        return
    print(f"[A] first 30 chars: {seq_a[:30]}")
    print(f"[A] last 30 chars:  {seq_a[-30:]}")
    print(f"[A] PAS-position-30 to PAS-position+30: {seq_a[UPSTREAM_NT-30:UPSTREAM_NT+30]}")

    feats_a = extract_one_site(seq_a, UPSTREAM_NT, DOWNSTREAM_NT, 8,
                                  me5_lookup, score3, m3, u1_lookup)
    print(f"[A] number of features computed: {len(feats_a)}")

    print(f"\n{'─'*70}")
    print(f"METHOD B: scanner (precomputed tracks at canonical pos)")
    print(f"{'─'*70}")

    tracks = GeneScoreTracks(chrom, gene_start, gene_end, strand,
                               fa, u1_lookup, me5_lookup, score3)
    print(f"[B] track length: {tracks.length}")
    print(f"[B] track_start_genomic: {tracks.track_start}")
    print(f"[B] track_end_genomic:   {tracks.track_end}")

    if strand == "+":
        gene_local = canon_pos - tracks.track_start
    else:
        gene_local = tracks.track_end - canon_pos - 1
    print(f"[B] gene_local position of canonical: {gene_local}")

    seq_at_gene_local = tracks.seq[gene_local - 30: gene_local + 30]
    print(f"[B] track seq at gene_local ±30: {seq_at_gene_local}")
    print(f"[A] same region (PAS ±30):     {seq_a[UPSTREAM_NT-30:UPSTREAM_NT+30]}")
    if seq_at_gene_local == seq_a[UPSTREAM_NT-30:UPSTREAM_NT+30]:
        print(f"[B] ✓ sequence matches METHOD A at canonical position")
    else:
        print(f"[B] ✗ SEQUENCE MISMATCH — coordinate math is wrong")
        for delta in [-1, 1, -2, 2, -3, 3]:
            alt = gene_local + delta
            alt_seq = tracks.seq[alt - 30: alt + 30]
            if alt_seq == seq_a[UPSTREAM_NT-30:UPSTREAM_NT+30]:
                print(f"[B]   match found at gene_local + {delta} ({alt})")
                gene_local = alt
                break

    window = tracks.slice_window(gene_local)
    if window is None:
        print(f"[B] ERROR: window out of bounds")
        return

    feats_b = extract_features_from_tracks(window)
    print(f"[B] number of features computed: {len(feats_b)}")

    print(f"\n{'─'*70}")
    print(f"FEATURE COMPARISON (per-bin features that exist in both methods)")
    print(f"{'─'*70}")
    print(f"{'feature':<25} {'training (A)':>15} {'scanner (B)':>15} {'delta':>10} {'tsv':>15}")

    common_features = sorted(set(feats_a.keys()) & set(feats_b.keys()))
    n_match = n_close = n_diff = 0
    for f in common_features:
        va = feats_a.get(f, np.nan)
        vb = feats_b.get(f, np.nan)
        vc = canon_row.get(f, np.nan)
        try:
            va = float(va); vb = float(vb); vc = float(vc)
        except (ValueError, TypeError):
            continue
        if np.isnan(va) and np.isnan(vb):
            continue
        delta = vb - va
        flag = ""
        if np.isnan(va) or np.isnan(vb):
            flag = "  ← NaN mismatch"
            n_diff += 1
        elif abs(delta) < 1e-4:
            flag = "  ✓"
            n_match += 1
        elif abs(delta) / (abs(va) + 1e-10) < 0.05:
            flag = "  close"
            n_close += 1
        else:
            flag = "  ← DIFFERS"
            n_diff += 1
        print(f"  {f:<25} {va:>15.4f} {vb:>15.4f} {delta:>+10.4f} {vc:>15.4f}{flag}")
    print(f"\n  Summary: {n_match} match, {n_close} close, {n_diff} differ")

    print(f"\n{'─'*70}")
    print(f"MODEL PREDICTIONS at canonical site")
    print(f"{'─'*70}")
    bundle = joblib.load(args.model_joblib)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    def derive_7_axes_from_bins(feats):
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

    def build_vec(feats_dict, source_label):
        feats_with_axes = derive_7_axes_from_bins(feats_dict)
        feats_with_wg = apply_gene_baselines_to_axes(feats_with_axes, gene_baseline)
        x = np.array([[feats_with_wg.get(c, np.nan) for c in feature_cols]],
                       dtype=np.float32)
        prob = float(model.predict_proba(x)[0, 1])
        n_nan = int(np.isnan(x).sum())
        print(f"  [{source_label}] splicing_prob = {prob:.4f}  "
               f"(NaN features: {n_nan}/{len(feature_cols)})")
        return prob

    p_a = build_vec(feats_a, "training pipeline A")
    p_b = build_vec(feats_b, "scanner B")

    x_tsv = np.array([[canon_row.get(c, np.nan) for c in feature_cols]],
                      dtype=np.float32)
    p_tsv = float(model.predict_proba(x_tsv)[0, 1])
    n_nan_tsv = int(np.isnan(x_tsv).sum())
    print(f"  [TSV row (ground truth features)] splicing_prob = {p_tsv:.4f}  "
           f"(NaN features: {n_nan_tsv}/{len(feature_cols)})")

    print(f"\n  Reference: training OOF prediction in chrsplit/test_predictions.tsv "
           f"(if A1BG is in test split)")


if __name__ == "__main__":
    main()