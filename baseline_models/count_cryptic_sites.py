#!/usr/bin/env python3
# count cryptic sites per shkurin et al
import argparse
import glob
import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd


def read_bed(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            rows.append({
                "chrom":  p[0], "start": int(p[1]), "end": int(p[2]),
                "name":   p[3] if len(p) > 3 else f"{p[0]}:{p[1]}-{p[2]}",
                "strand": p[5] if len(p) > 5 else "+",
            })
    return rows


def get_mean_thresholds(scan_dir, canonical_bed, test_chroms_path):
    test_chroms = set(open(test_chroms_path).read().split())
    canon = pd.DataFrame(read_bed(canonical_bed))
    canon["gene"] = canon["name"].str.split("::").str[0]
    canon = canon[canon["chrom"].isin(test_chroms)]

    parquets = {os.path.basename(p).replace(".parquet", ""): p
                for p in glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet"))}

    probs = {"rf": [], "lr": [], "cnn": []}
    for _, row in canon.iterrows():
        pq = parquets.get(row["gene"])
        if pq is None:
            continue
        df = pd.read_parquet(pq, columns=[
            "rf_prob", "lr_prob", "cnn_prob", "is_constitutive_cpa"])
        canon_idx = df.index[df["is_constitutive_cpa"]]
        if len(canon_idx) == 0:
            continue
        ci = canon_idx[0]
        for m in ["rf", "lr", "cnn"]:
            probs[m].append(float(df.iloc[ci][f"{m}_prob"]))
    return {m: float(np.mean(v)) for m, v in probs.items()}


def cache_path(out_dir, model, threshold, merge_dist):
    return os.path.join(
        out_dir,
        f"raw_peaks_{model}_thr{threshold:.4f}_merge{merge_dist}.parquet")


def collect_peaks_by_center(scan_dir, pc_genes_bed, thresholds, out_dir,
                            merge_dist=30, peak_width=500, force=False):
    if not force:
        caches = {m: cache_path(out_dir, m, thresholds[m], merge_dist)
                  for m in ["rf", "lr", "cnn"]}
        if all(os.path.exists(p) for p in caches.values()):
            print(f"  [cache] all three model caches exist; loading...")
            result = {}
            for m, p in caches.items():
                df = pd.read_parquet(p)
                result[m] = df
                print(f"    {m.upper()}: {len(df):,} peaks  ({p})")
            return result

    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    half = peak_width // 2
    out = defaultdict(list)

    for i, pq in enumerate(parquets):
        gene = os.path.basename(pq).replace(".parquet", "")
        gi_rows = pc_df[pc_df["gene"] == gene]
        if len(gi_rows) == 0:
            continue
        gi = gi_rows.iloc[0]

        df = pd.read_parquet(pq, columns=[
            "rf_prob", "lr_prob", "cnn_prob",
            "is_constitutive_cpa", "window_offset"])
        offset = df["window_offset"].values
        if gi["strand"] == "+":
            gs = gi["start"] + offset
        else:
            gs = gi["end"] - offset - peak_width
        center = gs + half
        canon_mask = df["is_constitutive_cpa"].values

        for m in ["rf", "lr", "cnn"]:
            probs_m = df[f"{m}_prob"].values
            high = probs_m >= thresholds[m]
            if not high.any():
                continue
            idx = np.where(high)[0]
            order = np.argsort(center[idx])
            c_sorted = center[idx][order]
            p_sorted = probs_m[idx][order]
            can_sorted = canon_mask[idx][order]

            cur_lo = int(c_sorted[0])
            cur_hi = int(c_sorted[0])
            cur_max = float(p_sorted[0])
            cur_can = bool(can_sorted[0])
            for j in range(1, len(c_sorted)):
                c = int(c_sorted[j])
                if c - cur_hi < merge_dist:
                    cur_hi = c
                    if p_sorted[j] > cur_max:
                        cur_max = float(p_sorted[j])
                    cur_can = cur_can or bool(can_sorted[j])
                else:
                    out[m].append((gi["chrom"],
                                   cur_lo - half, cur_hi + half,
                                   gene, gi["strand"], cur_max, cur_can))
                    cur_lo, cur_hi = c, c
                    cur_max = float(p_sorted[j])
                    cur_can = bool(can_sorted[j])
            out[m].append((gi["chrom"], cur_lo - half, cur_hi + half,
                           gene, gi["strand"], cur_max, cur_can))

        if (i + 1) % 2000 == 0:
            print(f"    scanned {i+1:,}/{len(parquets):,}")

    cols = ["chrom", "start", "end", "gene", "strand", "max_prob", "is_can"]
    result = {}
    for m in ["rf", "lr", "cnn"]:
        df = pd.DataFrame(out[m], columns=cols)
        result[m] = df
        path = cache_path(out_dir, m, thresholds[m], merge_dist)
        df.to_parquet(path, index=False, compression="snappy")
        print(f"    cached {m.upper()}: {len(df):,} peaks -> {path}")
    return result


def flank_bed(in_path, out_path, flank):
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            p[1] = str(max(0, int(p[1]) - flank))
            p[2] = str(int(p[2]) + flank)
            fout.write("\t".join(p) + "\n")


def bedtools_filter_out_multi(peaks_df, filter_beds, sample_name,
                              flank=0, strand_aware=True):
    if len(peaks_df) == 0:
        return peaks_df.copy()

    in_bed  = f"/tmp/__cryptic_in_{sample_name}.bed"
    out_bed = f"/tmp/__cryptic_out_{sample_name}.bed"

    flanked_beds = []
    temp_to_clean = []
    if flank > 0:
        for k, bp in enumerate(filter_beds):
            t = f"/tmp/__cryptic_flt_{sample_name}_{k}.bed"
            flank_bed(bp, t, flank)
            flanked_beds.append(t)
            temp_to_clean.append(t)
    else:
        flanked_beds = list(filter_beds)

    peaks_df.assign(score=peaks_df["max_prob"])[[
        "chrom", "start", "end", "gene", "score", "strand", "is_can"
    ]].to_csv(in_bed, sep="\t", header=False, index=False)
    temp_to_clean.append(in_bed)

    cmd = ["bedtools", "intersect", "-v"]
    if strand_aware:
        cmd.append("-s")
    cmd += ["-a", in_bed, "-b"] + flanked_beds
    subprocess.run(cmd, stdout=open(out_bed, "w"), check=True)
    temp_to_clean.append(out_bed)

    if os.path.getsize(out_bed) == 0:
        for p in temp_to_clean:
            if os.path.exists(p): os.remove(p)
        return peaks_df.iloc[0:0].copy()
    result = pd.read_csv(out_bed, sep="\t", header=None, names=[
        "chrom", "start", "end", "gene", "score", "strand", "is_can"])
    result = result.rename(columns={"score": "max_prob"})
    for p in temp_to_clean:
        if os.path.exists(p): os.remove(p)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir",      required=True)
    ap.add_argument("--pc_genes_bed",  required=True)
    ap.add_argument("--canonical_bed", required=True)
    ap.add_argument("--atlas_bed",     required=True, nargs="+",
                    help="One or more atlas BEDs. Filtering excludes peaks "
                         "overlapping ANY of these. Pass both PolyASite atlas "
                         "and polyA_DB-3 APA atlas here.")
    ap.add_argument("--test_chroms",   required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--merge_dist",    type=int, default=30,
                    help="Merge centers within this many nt (paper: 30)")
    ap.add_argument("--atlas_flank",   type=int, default=250,
                    help="±flank applied to ALL atlas BEDs (set 0 if your "
                         "atlas BEDs are already pre-flanked)")
    ap.add_argument("--canonical_flank", type=int, default=0,
                    help="Optional flank for canonical filter")
    ap.add_argument("--force", action="store_true",
                    help="Recompute raw peaks even if cache exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Compute thresholds (mean prob at test-set constitutive sites)")
    print("=" * 70)
    thresholds = get_mean_thresholds(args.scan_dir, args.canonical_bed,
                                     args.test_chroms)
    for m in ["rf", "lr", "cnn"]:
        print(f"  {m.upper()}: threshold = {thresholds[m]:.4f}  "
              f"cache: {cache_path(args.out_dir, m, thresholds[m], args.merge_dist)}")

    params = {
        "thresholds":      thresholds,
        "merge_dist":      args.merge_dist,
        "atlas_flank":     args.atlas_flank,
        "canonical_flank": args.canonical_flank,
        "scan_dir":        args.scan_dir,
        "canonical_bed":   args.canonical_bed,
        "atlas_beds":      list(args.atlas_bed),
    }
    with open(os.path.join(args.out_dir, "run_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(f"\n  atlas BEDs ({len(args.atlas_bed)}):")
    for bp in args.atlas_bed:
        n = sum(1 for _ in open(bp))
        print(f"    {bp} ({n:,} lines)")

    print()
    print("=" * 70)
    print(f"STEP 2-3: Threshold + merge window CENTERS within {args.merge_dist} nt")
    print("=" * 70)
    peaks = collect_peaks_by_center(args.scan_dir, args.pc_genes_bed,
                                    thresholds, args.out_dir,
                                    merge_dist=args.merge_dist,
                                    force=args.force)
    raw_counts = {m: len(peaks[m]) for m in ["rf", "lr", "cnn"]}
    for m in ["rf", "lr", "cnn"]:
        print(f"  {m.upper()}: {raw_counts[m]:,} peaks after merging")

    print()
    print("=" * 70)
    print(f"STEP 4: Drop peaks overlapping LABELED constitutive CPA "
          f"(flank=±{args.canonical_flank} nt)")
    print("=" * 70)
    after_canon = {}
    for m in ["rf", "lr", "cnn"]:
        before = len(peaks[m])
        df_a = peaks[m][~peaks[m]["is_can"]].copy()
        df_b = bedtools_filter_out_multi(df_a, [args.canonical_bed], m,
                                         flank=args.canonical_flank,
                                         strand_aware=True)
        after_canon[m] = df_b
        print(f"  {m.upper()}: {before:,} -> "
              f"{len(df_a):,} (drop in-peak canonical) -> "
              f"{len(df_b):,} (drop bedtools-overlap canonical)")

    print()
    print("=" * 70)
    print(f"STEP 5: Drop peaks overlapping ANY atlas BED "
          f"(flank=±{args.atlas_flank} nt)")
    print("=" * 70)
    cryptic = {}
    for m in ["rf", "lr", "cnn"]:
        before = len(after_canon[m])
        df = bedtools_filter_out_multi(after_canon[m], args.atlas_bed,
                                       f"{m}_atlas",
                                       flank=args.atlas_flank,
                                       strand_aware=True)
        cryptic[m] = df
        n_filtered = before - len(df)
        print(f"  {m.upper()}: {before:,} -> {len(df):,}  "
              f"({n_filtered:,} dropped, "
              f"{n_filtered/max(before,1)*100:.1f}%)")

    print()
    print("=" * 70)
    print("FINAL: cryptic site counts (paper methodology)")
    print("=" * 70)
    print(f"  {'Model':<6}  {'Raw peaks':>11}  {'After -canon':>13}  "
          f"{'Cryptic':>10}")
    for m in ["rf", "lr", "cnn"]:
        print(f"  {m.upper():<6}  "
              f"{raw_counts[m]:>11,}  "
              f"{len(after_canon[m]):>13,}  "
              f"{len(cryptic[m]):>10,}")
    print()
    print("  Paper reported ~40,537 cryptic sites for their RF baseline.")
    print(f"  Parameters: merge_dist={args.merge_dist}, "
          f"atlas_flank=±{args.atlas_flank}, "
          f"n_atlas_files={len(args.atlas_bed)}")

    for m in ["rf", "lr", "cnn"]:
        out_bed = os.path.join(args.out_dir, f"cryptic_{m}.bed")
        df = cryptic[m].copy()
        df["score"] = df["max_prob"]
        df["name"]  = df["gene"] + "_cryptic"
        df[["chrom", "start", "end", "name", "score", "strand"]].to_csv(
            out_bed, sep="\t", header=False, index=False)
        print(f"  wrote {out_bed} ({len(df):,} cryptic peaks)")


if __name__ == "__main__":
    main()