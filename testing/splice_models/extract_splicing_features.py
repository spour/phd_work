#!/usr/bin/env python3
# splicing-environment feature extractor
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd


ME5_STRONG_THRESHOLD = 7.0
ME3_STRONG_THRESHOLD = 6.5

DECAY_TAUS = (50, 150, 400)


class GenomeFasta:

    def __init__(self, path):
        self.path = path
        try:
            import pysam
            self.fa = pysam.FastaFile(path)
            self.engine = "pysam"
            self.refs = set(self.fa.references)
        except Exception:
            from pyfaidx import Fasta
            self.fa = Fasta(path, sequence_always_upper=True)
            self.engine = "pyfaidx"
            self.refs = set(self.fa.keys())

    def fetch(self, chrom, start, end, strand="+"):
        c = chrom if chrom in self.refs else chrom.lstrip("chr")
        if c not in self.refs:
            return None
        if self.engine == "pysam":
            seq = self.fa.fetch(c, start, end).upper()
        else:
            seq = str(self.fa[c][start:end]).upper()
        if strand == "-":
            seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        return seq


def read_bed_with_labels(path, default_label=None):
    sites = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            chrom, start, end, name = p[0], int(p[1]), int(p[2]), p[3]
            strand = p[5]
            tokens = name.split("_")
            gene_guess = tokens[-1] if len(tokens) > 1 else name
            sites.append({
                "chrom": chrom, "start": start, "end": end, "name": name,
                "strand": strand, "gene": gene_guess,
                "y": default_label,
            })
    return sites


def auto_label_from_name(name):
    n = name.lower()
    if n.startswith("3way_") or n.startswith("ccpcpa_") or \
       n.startswith("intragenic_") or n.startswith("tier"):
        return 0
    if n.startswith("canonical_") or n.startswith("constitutive_") or \
       n.startswith("real_"):
        return 1
    return None


def init_maxentscan():
    try:
        from maxentpy import maxent
        from maxentpy.maxent import load_matrix5, load_matrix3
        m5 = load_matrix5()
        m3 = load_matrix3()
        def score5(seq):
            if "N" in seq or len(seq) != 9:
                return -99.0
            try:
                return float(maxent.score5(seq, matrix=m5))
            except Exception:
                return -99.0
        def score3(seq):
            if "N" in seq or len(seq) != 23:
                return -99.0
            try:
                return float(maxent.score3(seq, matrix=m3))
            except Exception:
                return -99.0
        return score5, score3, m5, m3
    except ImportError:
        raise SystemExit(
            "maxentpy is required. Install via:\n"
            "    bash install_maxentpy_noc.sh"
        )


def load_me5_lookup(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"MaxEntScan 5'SS lookup file not found: {path}\n"
            f"Run: python build_me5_lookup.py --out {path}"
        )
    arr = np.load(path, allow_pickle=True)
    lookup = arr.item() if arr.dtype == object else dict(arr)
    return lookup


def scan_me5(seq, me5_lookup):
    n = len(seq) - 9 + 1
    if n <= 0:
        return np.array([], dtype=np.float32)
    scores = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if seq[i + 3] != "G" or seq[i + 4] != "T":
            continue
        kmer = seq[i:i + 9]
        if "N" in kmer:
            continue
        v = me5_lookup.get(kmer)
        if v is not None:
            scores[i] = v
    return scores


def scan_me3(seq, score3, m3):
    n = len(seq) - 23 + 1
    if n <= 0:
        return np.array([], dtype=np.float32)
    scores = np.full(n, np.nan, dtype=np.float32)
    try:
        from maxentpy import maxent
    except ImportError:
        return scores
    for i in range(n):
        if seq[i + 18] != "A" or seq[i + 19] != "G":
            continue
        kmer = seq[i:i + 23]
        if "N" in kmer:
            continue
        try:
            scores[i] = maxent.score3(kmer, matrix=m3)
        except Exception:
            pass
    return scores


def load_u1_lookup(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"U1 lookup file not found: {path}\n"
            f"Run: python build_u1_lookup.py --out {path}"
        )
    arr = np.load(path, allow_pickle=True)
    lookup = arr.item() if arr.dtype == object else dict(arr)
    return lookup


def scan_u1(seq, u1_lookup):
    n = len(seq) - 9 + 1
    if n <= 0:
        return np.array([], dtype=np.float32)
    scores = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        kmer = seq[i:i + 9]
        if "N" in kmer:
            continue
        v = u1_lookup.get(kmer)
        if v is not None:
            scores[i] = -float(v)
    return scores


def bin_position_in_window(pos, n_bins, total_width):
    if pos < 0 or pos >= total_width:
        return None
    return min(n_bins, 1 + (pos * n_bins) // total_width)


def aggregate_per_bin(positions, scores, n_bins, total_width):
    out = {(b, "mean"): np.nan for b in range(1, n_bins + 1)}
    out.update({(b, "max"): np.nan for b in range(1, n_bins + 1)})
    if len(positions) == 0:
        return out
    bins = np.array([bin_position_in_window(p, n_bins, total_width)
                      for p in positions])
    valid = (bins != None) & ~np.isnan(scores)
    if not valid.any():
        return out
    bins_v = bins[valid].astype(int)
    scores_v = scores[valid]
    for b in range(1, n_bins + 1):
        m = bins_v == b
        if m.any():
            out[(b, "mean")] = float(np.nanmean(scores_v[m]))
            out[(b, "max")] = float(np.nanmax(scores_v[m]))
    return out


def decay_kernel(positions, scores, center, tau, direction):
    if len(positions) == 0:
        return np.nan
    positions = np.asarray(positions, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if direction == "up":
        m = positions < center
    else:
        m = positions > center
    valid = m & np.isfinite(scores)
    if not valid.any():
        return np.nan
    d = np.abs(positions[valid] - center)
    w = np.exp(-d / float(tau))
    s = scores[valid] * w
    return float(np.nansum(s) / max(np.nansum(w), 1e-12))


def extract_one_site(seq, upstream_nt, downstream_nt, n_bins,
                       me5_lookup, score3, m3, u1_lookup,
                       me5_strong=ME5_STRONG_THRESHOLD,
                       me3_strong=ME3_STRONG_THRESHOLD):
    total_width = upstream_nt + downstream_nt
    if len(seq) < total_width:
        seq = seq + "N" * (total_width - len(seq))
    elif len(seq) > total_width:
        seq = seq[:total_width]

    center = upstream_nt

    u1_scores = scan_u1(seq, u1_lookup)
    me3_scores = scan_me3(seq, score3, m3)
    me5_scores = scan_me5(seq, me5_lookup)

    u1_pos = np.arange(len(u1_scores))
    me5_pos = np.arange(len(me5_scores))
    me3_pos = np.arange(len(me3_scores))

    out = {}

    u1_bins = aggregate_per_bin(u1_pos, u1_scores, n_bins, total_width)
    me3_bins = aggregate_per_bin(me3_pos, me3_scores, n_bins, total_width)
    me5_bins = aggregate_per_bin(me5_pos, me5_scores, n_bins, total_width)
    for b in range(1, n_bins + 1):
        out[f"bin{b}_u1_mean"] = u1_bins[(b, "mean")]
        out[f"bin{b}_u1_max"] = u1_bins[(b, "max")]
        out[f"bin{b}_me3_mean"] = me3_bins[(b, "mean")]
        out[f"bin{b}_me3_max"] = me3_bins[(b, "max")]
        out[f"bin{b}_me5_mean"] = me5_bins[(b, "mean")]
        out[f"bin{b}_me5_max"] = me5_bins[(b, "max")]

    out["all_u1_max"] = (float(np.nanmax(u1_scores))
                          if np.isfinite(u1_scores).any() else np.nan)

    for tau in DECAY_TAUS:
        out[f"all_u1_decay_up_tau{tau}"] = decay_kernel(
            u1_pos, u1_scores, center, tau, "up")
        out[f"all_u1_decay_dn_tau{tau}"] = decay_kernel(
            u1_pos, u1_scores, center, tau, "dn")
        out[f"all_me3_decay_up_tau{tau}"] = decay_kernel(
            me3_pos, me3_scores, center, tau, "up")
        out[f"all_me3_decay_dn_tau{tau}"] = decay_kernel(
            me3_pos, me3_scores, center, tau, "dn")

    inner_margin = 25
    for W in (250, 500):
        lo = max(0, center - W)
        hi = max(0, center - inner_margin)
        if hi > lo:
            mask = (me3_pos >= lo) & (me3_pos < hi)
            sub = me3_scores[mask]
            sub_valid = sub[np.isfinite(sub)]
            if len(sub_valid) >= 3:
                top3 = np.sort(sub_valid)[-3:]
                out[f"term_me3_top3mean_{W}_{inner_margin}"] = float(np.mean(top3))
            elif len(sub_valid) > 0:
                out[f"term_me3_top3mean_{W}_{inner_margin}"] = float(np.mean(sub_valid))
            else:
                out[f"term_me3_top3mean_{W}_{inner_margin}"] = np.nan
            out[f"term_me3_max_{W}_{inner_margin}"] = (float(np.max(sub_valid))
                if len(sub_valid) else np.nan)
            out[f"term_me3_mean_{W}_{inner_margin}"] = (float(np.mean(sub_valid))
                if len(sub_valid) else np.nan)
        else:
            out[f"term_me3_top3mean_{W}_{inner_margin}"] = np.nan
            out[f"term_me3_max_{W}_{inner_margin}"] = np.nan
            out[f"term_me3_mean_{W}_{inner_margin}"] = np.nan

    dn_mask = me5_pos > center
    dn_pos = me5_pos[dn_mask]
    dn_scores = me5_scores[dn_mask]
    dn_valid = np.isfinite(dn_scores)
    dn_pos = dn_pos[dn_valid]
    dn_scores = dn_scores[dn_valid]
    strong_mask = dn_scores >= me5_strong
    if len(dn_scores):
        out["dn5ss_best_score"] = float(np.max(dn_scores))
        out["dn5ss_top3mean"] = float(np.mean(np.sort(dn_scores)[-3:]))
    else:
        out["dn5ss_best_score"] = np.nan
        out["dn5ss_top3mean"] = np.nan
    out["dn5ss_n_strong"] = int(strong_mask.sum())
    if strong_mask.any():
        nearest_strong = int(np.min(dn_pos[strong_mask]) - center)
        out["dn5ss_nearest_strong_dist"] = nearest_strong
    else:
        out["dn5ss_nearest_strong_dist"] = np.nan
    dn250 = (dn_pos >= center) & (dn_pos < center + 250)
    out["dn5ss_mean_dn250"] = (float(np.mean(dn_scores[dn250]))
        if dn250.any() else np.nan)

    for W in (500, 1000):
        lo = max(0, center - W)
        hi = max(0, center - inner_margin)
        if hi > lo:
            sub_mask = (me3_pos >= lo) & (me3_pos < hi)
            sub_me3 = me3_scores[sub_mask]
            sub_pos = me3_pos[sub_mask]
            valid = np.isfinite(sub_me3)
            if valid.any():
                best_3ss_local_pos = sub_pos[valid][np.argmax(sub_me3[valid])]
                ex_lo, ex_hi = int(best_3ss_local_pos), int(center)
                interv_me5_mask = (me5_pos >= ex_lo) & (me5_pos < ex_hi)
                interv_me5_v = me5_scores[interv_me5_mask]
                interv_valid = np.isfinite(interv_me5_v)
                if interv_valid.any():
                    out[f"interv_me5_max_{W}"] = float(np.max(interv_me5_v[interv_valid]))
                    out[f"interv_me5_mean_{W}"] = float(np.mean(interv_me5_v[interv_valid]))
                    out[f"interv_me5_n_strong_{W}"] = int(
                        (interv_me5_v[interv_valid] >= me5_strong).sum())
                else:
                    out[f"interv_me5_max_{W}"] = np.nan
                    out[f"interv_me5_mean_{W}"] = np.nan
                    out[f"interv_me5_n_strong_{W}"] = 0
                exon_len = ex_hi - ex_lo
                out[f"interv_len_{W}"] = int(exon_len)
                out[f"interv_exon_geometry_{W}"] = int(
                    50 <= exon_len <= 400)
            else:
                out[f"interv_me5_max_{W}"] = np.nan
                out[f"interv_me5_mean_{W}"] = np.nan
                out[f"interv_me5_n_strong_{W}"] = 0
                out[f"interv_len_{W}"] = 0
                out[f"interv_exon_geometry_{W}"] = 0
        else:
            out[f"interv_me5_max_{W}"] = np.nan
            out[f"interv_me5_mean_{W}"] = np.nan
            out[f"interv_me5_n_strong_{W}"] = 0
            out[f"interv_len_{W}"] = 0
            out[f"interv_exon_geometry_{W}"] = 0

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_bed", default=None,
                     help="Single BED6 with both classes, labels inferred "
                          "from the name prefix (canonical_*=1, 3way_*=0, "
                          "etc).")
    ap.add_argument("--pos_bed", default=None,
                     help="BED6 of positive (canonical, y=1) sites.")
    ap.add_argument("--neg_bed", default=None,
                     help="BED6 of negative (3-way intragenic, y=0) sites.")
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True,
                     help="Path to U1 hybridization lookup .npy "
                          "(see build_u1_lookup.py)")
    ap.add_argument("--me5_lookup", required=True,
                     help="Path to MaxEntScan 5'SS lookup .npy "
                          "(see build_me5_lookup.py)")
    ap.add_argument("--upstream_nt", type=int, default=3000,
                     help="Window upstream of cleavage site, in transcribed "
                          "orientation. Default 3000.")
    ap.add_argument("--downstream_nt", type=int, default=1000,
                     help="Window downstream of cleavage site, in transcribed "
                          "orientation. Default 1000.")
    ap.add_argument("--n_bins", type=int, default=8,
                     help="Number of positional bins to summarize per site. "
                          "Default 8.")
    ap.add_argument("--out_tsv", required=True,
                     help="Output features TSV. A sibling .config.json is "
                          "also written documenting the exact parameters.")
    ap.add_argument("--limit", type=int, default=0,
                     help="Stop after N sites (0 = all). Useful for testing.")
    ap.add_argument("--log_every", type=int, default=500)
    args = ap.parse_args()

    if not (args.combined_bed or (args.pos_bed and args.neg_bed)):
        raise SystemExit("Provide either --combined_bed OR both --pos_bed and --neg_bed")

    print(f"[extract] window: -{args.upstream_nt} / +{args.downstream_nt} nt, "
          f"n_bins={args.n_bins}")

    print(f"[extract] loading U1 lookup from {args.u1_lookup}")
    u1_lookup = load_u1_lookup(args.u1_lookup)
    print(f"[extract]   {len(u1_lookup):,} 9-mers loaded")

    print(f"[extract] loading MaxEntScan 5'SS lookup from {args.me5_lookup}")
    me5_lookup = load_me5_lookup(args.me5_lookup)
    print(f"[extract]   {len(me5_lookup):,} 9-mers loaded")

    print(f"[extract] initializing MaxEntScan (for 3'SS only; 5'SS uses lookup)")
    score5, score3, m5, m3 = init_maxentscan()

    print(f"[extract] opening FASTA {args.reference_fasta}")
    fa = GenomeFasta(args.reference_fasta)

    sites = []
    if args.combined_bed:
        sites_raw = read_bed_with_labels(args.combined_bed, default_label=None)
        for s in sites_raw:
            y = auto_label_from_name(s["name"])
            if y is None:
                continue
            s["y"] = y
            sites.append(s)
    else:
        sites.extend(read_bed_with_labels(args.pos_bed, default_label=1))
        sites.extend(read_bed_with_labels(args.neg_bed, default_label=0))

    if args.limit:
        sites = sites[:args.limit]
    print(f"[extract] total sites: {len(sites):,} "
          f"({sum(1 for s in sites if s['y']==1):,} positive, "
          f"{sum(1 for s in sites if s['y']==0):,} negative)")

    t0 = time.time()
    rows = []
    n_bad = 0
    for i, s in enumerate(sites):
        center_genomic = (s["start"] + s["end"]) // 2
        if s["strand"] == "+":
            win_start = center_genomic - args.upstream_nt
            win_end = center_genomic + args.downstream_nt
        else:
            win_start = center_genomic - args.downstream_nt
            win_end = center_genomic + args.upstream_nt
        win_start = max(0, win_start)
        seq = fa.fetch(s["chrom"], win_start, win_end, strand=s["strand"])
        if seq is None or len(seq) < (args.upstream_nt + args.downstream_nt - 100):
            n_bad += 1
            continue
        feats = extract_one_site(
            seq, args.upstream_nt, args.downstream_nt, args.n_bins,
            me5_lookup, score3, m3, u1_lookup,
        )
        feats["id"] = f"{s['chrom']}:{center_genomic}:{s['strand']}"
        feats["gene"] = s["gene"]
        feats["y"] = int(s["y"])
        feats["chrom"] = s["chrom"]
        feats["center"] = center_genomic
        feats["strand"] = s["strand"]
        rows.append(feats)
        if (i + 1) % args.log_every == 0 or i < 5:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (len(sites) - i - 1) / max(rate, 1e-9)
            print(f"[extract]   {i + 1:>6,}/{len(sites):>6,}  "
                   f"rate={rate:.1f}/s  eta={eta/60:.1f}m  "
                   f"bad={n_bad}")

    print(f"[extract] done in {(time.time() - t0)/60:.1f} min  "
           f"({n_bad:,} sites with bad sequence fetch)")
    out_dir = os.path.dirname(os.path.abspath(args.out_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[extract] wrote {args.out_tsv}  ({len(df):,} rows, "
           f"{len(df.columns)} columns)")

    cfg = {
        "upstream_nt": args.upstream_nt,
        "downstream_nt": args.downstream_nt,
        "n_bins": args.n_bins,
        "n_sites_input": len(sites),
        "n_sites_output": len(df),
        "n_bad_fetch": n_bad,
        "u1_lookup": args.u1_lookup,
        "me5_lookup": args.me5_lookup,
        "reference_fasta": args.reference_fasta,
        "me5_strong_threshold": ME5_STRONG_THRESHOLD,
        "me3_strong_threshold": ME3_STRONG_THRESHOLD,
        "decay_taus": list(DECAY_TAUS),
    }
    config_path = args.out_tsv + ".config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[extract] wrote {config_path}")


if __name__ == "__main__":
    main()