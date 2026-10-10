#!/usr/bin/env python3
# per-gene stride-1 genome-wide splicing scanner
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from extract_splicing_features import (
    GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan,
)

UPSTREAM_NT = 3000
DOWNSTREAM_NT = 1000
N_BINS = 8
BIN_WIDTH = (UPSTREAM_NT + DOWNSTREAM_NT) // N_BINS


AXES = ["upstream_mean_u1", "upstream_max_me3", "term_3ss", "local_me5",
        "pas_prox_u1", "dn5ss_has_strong", "interv_me5_max"]


class GeneScoreTracks:
    def __init__(self, chrom, gene_start, gene_end, strand,
                  fa, u1_lookup, me5_lookup, score3):
        self.chrom = chrom
        self.strand = strand
        margin = max(UPSTREAM_NT, DOWNSTREAM_NT) + 50
        self.track_start = max(0, gene_start - margin)
        self.track_end = gene_end + margin
        seq = fa.fetch(chrom, self.track_start, self.track_end, strand=strand)
        if seq is None:
            raise RuntimeError(f"Could not fetch sequence for {chrom}")
        self.seq = seq.upper()
        self.length = len(self.seq)

        self.u1 = self._compute_u1_track(u1_lookup)
        self.me3 = self._compute_me3_track(score3)
        self.me5 = self._compute_me5_track(me5_lookup)

    def _compute_u1_track(self, u1_lookup):
        out = np.full(self.length, np.nan, dtype=np.float32)
        L = self.length
        for i in range(L - 9 + 1):
            kmer = self.seq[i:i+9]
            v = u1_lookup.get(kmer)
            if v is not None:
                out[i] = -float(v)
        return out

    def _compute_me3_track(self, score3):
        out = np.full(self.length, np.nan, dtype=np.float32)
        L = self.length
        for i in range(L - 23 + 1):
            kmer = self.seq[i:i+23]
            if kmer[18:20] == "AG":
                try:
                    out[i] = score3(kmer)
                except Exception:
                    pass
        return out

    def _compute_me5_track(self, me5_lookup):
        out = np.full(self.length, np.nan, dtype=np.float32)
        L = self.length
        for i in range(L - 9 + 1):
            kmer = self.seq[i:i+9]
            if kmer[3:5] == "GT":
                v = me5_lookup.get(kmer)
                if v is not None:
                    out[i] = v
        return out

    def slice_window(self, gene_local_pos):
        win_start = gene_local_pos - UPSTREAM_NT
        win_end = gene_local_pos + DOWNSTREAM_NT
        if win_start < 0 or win_end > self.length:
            return None
        return {
            "u1": self.u1[win_start:win_end],
            "me3": self.me3[win_start:win_end],
            "me5": self.me5[win_start:win_end],
            "pas_local": gene_local_pos - win_start,
        }


def nanmean(a):
    if a.size == 0:
        return np.nan
    m = ~np.isnan(a)
    if not m.any():
        return np.nan
    return float(a[m].mean())


def nanmax(a):
    if a.size == 0:
        return np.nan
    m = ~np.isnan(a)
    if not m.any():
        return np.nan
    return float(a[m].max())


def extract_features_from_tracks(window):
    out = {}
    bins_split = np.array_split(np.arange(UPSTREAM_NT + DOWNSTREAM_NT), N_BINS)
    for i, idxs in enumerate(bins_split, start=1):
        for motif, arr in (("u1", window["u1"]), ("me3", window["me3"]),
                           ("me5", window["me5"])):
            slc = arr[idxs]
            out[f"bin{i}_{motif}_mean"] = nanmean(slc)
            out[f"bin{i}_{motif}_max"] = nanmax(slc)

    bin_u1_means = [out[f"bin{i}_u1_mean"] for i in range(1, 7)]
    bin_me3_maxs = [out[f"bin{i}_me3_max"] for i in range(1, 7)]
    out["upstream_mean_u1"] = float(np.nanmean(bin_u1_means)) \
        if np.isfinite(bin_u1_means).any() else np.nan
    out["upstream_max_me3"] = float(np.nanmax(bin_me3_maxs)) \
        if np.isfinite(bin_me3_maxs).any() else np.nan

    pas_local = window["pas_local"]
    me3_slice = window["me3"][pas_local - 500: pas_local - 25]
    finite = me3_slice[~np.isnan(me3_slice)]
    if len(finite) >= 3:
        out["term_3ss"] = float(np.sort(finite)[-3:].mean())
    elif len(finite) > 0:
        out["term_3ss"] = float(finite.mean())
    else:
        out["term_3ss"] = np.nan

    out["local_me5"] = out["bin6_me5_mean"]
    out["pas_prox_u1"] = out["bin6_u1_mean"]

    dn5ss_slice = window["me5"][pas_local: pas_local + 500]
    strong_threshold = 6.5
    out["dn5ss_has_strong"] = float(
        np.any((~np.isnan(dn5ss_slice)) & (dn5ss_slice > strong_threshold)))

    out["interv_me5_max"] = out["bin6_me5_max"]

    return out


def apply_gene_baselines_to_axes(features, gene_baselines):
    out = dict(features)
    for ax in AXES:
        baseline = gene_baselines.get(f"baseline_{ax}", np.nan)
        if np.isfinite(baseline):
            out[f"{ax}__wg"] = features[ax] - baseline
        else:
            out[f"{ax}__wg"] = np.nan
    return out


def scan_one_gene(gene, gene_row, cpa_parquet_path, fa, u1_lookup, me5_lookup,
                   score3, gene_baselines_lookup, model_pipe, feature_cols,
                   out_path):
    chrom = gene_row["chrom"]
    gene_start = int(gene_row["start"])
    gene_end = int(gene_row["end"])
    strand = gene_row["strand"]

    cpa_df = pd.read_parquet(cpa_parquet_path)
    n_candidates = len(cpa_df)
    if n_candidates == 0:
        return None

    t0 = time.time()
    try:
        tracks = GeneScoreTracks(chrom, gene_start, gene_end, strand,
                                   fa, u1_lookup, me5_lookup, score3)
    except Exception as e:
        return {"gene": gene, "status": f"track_error: {e}", "n_candidates": n_candidates}
    track_time = time.time() - t0

    gene_baselines = gene_baselines_lookup.get(gene, {})

    t1 = time.time()
    n_cand = len(cpa_df)
    cand_positions = cpa_df["window_start_genomic"].values
    feature_matrix = np.full((n_cand, len(feature_cols)), np.nan,
                              dtype=np.float32)
    statuses = ["out_of_bounds"] * n_cand
    n_skip = 0

    for i, win_start_gen in enumerate(cand_positions):
        if strand == "+":
            gene_local = int(win_start_gen) - tracks.track_start
        else:
            gene_local = tracks.track_end - int(win_start_gen)

        window = tracks.slice_window(gene_local)
        if window is None:
            n_skip += 1
            continue

        feats = extract_features_from_tracks(window)
        feats = apply_gene_baselines_to_axes(feats, gene_baselines)
        for j, c in enumerate(feature_cols):
            v = feats.get(c, np.nan)
            if v is not None and np.isfinite(v):
                feature_matrix[i, j] = v
        statuses[i] = "ok"

    feat_time = time.time() - t1

    t2 = time.time()
    ok_mask = np.array([s == "ok" for s in statuses])
    probs = np.full(n_cand, np.nan, dtype=np.float32)
    if ok_mask.any():
        try:
            probs[ok_mask] = model_pipe.predict_proba(
                feature_matrix[ok_mask])[:, 1].astype(np.float32)
        except Exception as e:
            return {"gene": gene, "status": f"predict_error: {e}",
                    "n_candidates": n_cand, "n_out_of_bounds": n_skip}
    predict_time = time.time() - t2
    scan_time = feat_time + predict_time

    out_df = pd.DataFrame({
        "window_start_genomic": cand_positions.astype(np.int64),
        "splicing_prob": probs,
        "status": statuses,
    })
    out_df.to_parquet(out_path, index=False)

    return {"gene": gene, "status": "ok", "n_candidates": n_cand,
            "n_out_of_bounds": n_skip, "track_time_s": track_time,
            "scan_time_s": scan_time,
            "feat_time_s": feat_time, "predict_time_s": predict_time,
            "track_length": tracks.length}


def read_pc_genes_bed(path):
    df = pd.read_csv(path, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    df["gene"] = df["name"].astype(str)
    if df["gene"].str.contains("_").any():
        df["gene"] = df["gene"].str.split("_").str[-1]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpa_parquet_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--gene_baselines", required=True)
    ap.add_argument("--model_joblib", required=True)
    ap.add_argument("--out_dir", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--gene", help="Single gene to scan (for testing)")
    group.add_argument("--gene_list_file", help="One gene name per line")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--skip_if_exists", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.gene:
        genes_to_run = [args.gene]
    else:
        with open(args.gene_list_file) as f:
            genes_to_run = [line.strip() for line in f if line.strip()]
    print(f"[scan] {len(genes_to_run)} genes to scan")

    print(f"[scan] loading gene metadata")
    pc_genes = read_pc_genes_bed(args.pc_genes_bed)
    pc_genes_lookup = pc_genes.drop_duplicates(subset="gene", keep="first"
                                                  ).set_index("gene").to_dict("index")

    print(f"[scan] loading gene baselines")
    base_df = pd.read_csv(args.gene_baselines, sep="\t")
    base_df = base_df.drop_duplicates(subset="gene", keep="first")
    baseline_cols = [c for c in base_df.columns if c.startswith("baseline_")]
    gene_baselines_lookup = base_df.set_index("gene")[baseline_cols].to_dict("index")
    print(f"[scan]   {len(gene_baselines_lookup):,} gene baselines loaded")

    print(f"[scan] loading model")
    bundle = joblib.load(args.model_joblib)
    model_pipe = bundle["model"]
    feature_cols = bundle["feature_cols"]
    print(f"[scan]   {len(feature_cols)} features required")

    print(f"[scan] loading FASTA + score lookups")
    fa = GenomeFasta(args.reference_fasta)
    u1_lookup = load_u1_lookup(args.u1_lookup)
    me5_lookup = load_me5_lookup(args.me5_lookup)
    _, score3, _, m3 = init_maxentscan()

    t0 = time.time()
    status_rows = []
    for i, gene in enumerate(genes_to_run):
        if gene not in pc_genes_lookup:
            status_rows.append({"gene": gene, "status": "not_in_pc_genes_bed",
                                 "n_candidates": 0})
            continue
        cpa_path = os.path.join(args.cpa_parquet_dir, f"{gene}.parquet")
        if not os.path.exists(cpa_path):
            status_rows.append({"gene": gene, "status": "cpa_parquet_missing",
                                 "n_candidates": 0})
            continue
        out_path = os.path.join(args.out_dir, f"{gene}.parquet")
        if args.skip_if_exists and os.path.exists(out_path):
            continue
        try:
            result = scan_one_gene(gene, pc_genes_lookup[gene], cpa_path,
                                     fa, u1_lookup, me5_lookup, score3,
                                     gene_baselines_lookup, model_pipe,
                                     feature_cols, out_path)
            if result is not None:
                status_rows.append(result)
        except Exception as e:
            status_rows.append({"gene": gene, "status": f"error: {e}",
                                 "n_candidates": 0})

        if (i + 1) % args.log_every == 0 or (i + 1) == len(genes_to_run):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(genes_to_run) - i - 1) / max(rate, 1e-9)
            print(f"[scan] {i + 1:>6,}/{len(genes_to_run):>6,}  "
                   f"rate={rate:.2f} g/s  eta={eta/60:.1f}m  "
                   f"last={genes_to_run[i]}", flush=True)

    status_path = os.path.join(args.out_dir,
                                  f"_status_{os.getpid()}.tsv")
    pd.DataFrame(status_rows).to_csv(status_path, sep="\t", index=False)
    print(f"[scan] wrote status {status_path}")
    print(f"[scan] total elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()