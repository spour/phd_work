#!/usr/bin/env python3
# trace every pipeline stage for one gene/site
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


MODELS_3WAY = ("rf", "lr", "cnn")
PROB_COL = {"rf": "rf_prob", "lr": "lr_prob",
            "cnn": "cnn_prob", "aparent2": "aparent2_prob"}


def find_peak_caches(out_dir, models):
    out = {}
    for m in models:
        cands = glob.glob(os.path.join(out_dir,
                                        f"_cache_peaks_{m}_*.parquet"))
        if cands:
            out[m] = max(cands, key=os.path.getmtime)
    return out


def parse_threshold_from_filename(path):
    base = os.path.basename(path)
    bits = base.split("_")
    for b in bits:
        if b.startswith("thr"):
            try:
                return float(b[3:])
            except ValueError:
                return None
    return None


def get_pc_row(pc_genes_bed, gene):
    for line in open(pc_genes_bed):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4 and parts[3] == gene:
            return {
                "chrom":  parts[0],
                "start":  int(parts[1]),
                "end":    int(parts[2]),
                "gene":   parts[3],
                "strand": parts[5] if len(parts) >= 6 else "+",
            }
    return None


def offset_to_genomic_start(offset, gi, win_size=500):
    if gi["strand"] == "+":
        return gi["start"] + int(offset)
    return gi["end"] - int(offset) - win_size


def genomic_to_offset(genomic_pos, gi, win_size=500):
    if gi["strand"] == "+":
        return int(genomic_pos) - gi["start"]
    return gi["end"] - int(genomic_pos) - win_size


def check_3utr_overlap(chrom, start, end, strand, three_prime_zone_bed):
    if not os.path.exists(three_prime_zone_bed):
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
    tmp.write(f"{chrom}\t{start}\t{end}\t.\t0\t{strand}\n")
    tmp.close()
    res = subprocess.run(
        ["bedtools", "intersect", "-u", "-s",
         "-a", tmp.name, "-b", three_prime_zone_bed],
        stdout=subprocess.PIPE, text=True, check=False,
    )
    os.remove(tmp.name)
    return bool(res.stdout.strip())


def section(title):
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene", help="gene symbol (parquet basename)")
    ap.add_argument("site_offset", type=int, nargs="?", default=None,
                    help="transcript window_offset to focus on (optional)")
    ap.add_argument("--tolerance", type=int, default=500,
                    help="window_offset tolerance (nt) when reporting nearby "
                         "windows / peaks. Default 500.")
    ap.add_argument("--slack", type=int, default=250,
                    help="cluster slack to simulate (must match the value in "
                         "make_figure1.py). Default 250.")
    ap.add_argument("--scan_dir",     default="gene_scans")
    ap.add_argument("--figures_dir",  default="figures")
    ap.add_argument("--pc_genes_bed", default="data/pc_genes.bed")
    ap.add_argument("--three_prime_zone_bed",
                     default="data/annotation/three_prime_zone.bed")
    ap.add_argument("--three_way_bed",
                     default="figures/cpccpa_3way_rflrcnn_intragenic.bed")
    ap.add_argument("--canonical_bed",
                     default="data/Dominant_CPA_baseline_total.bed")
    ap.add_argument("--rf_thr",  type=float, default=None)
    ap.add_argument("--lr_thr",  type=float, default=None)
    ap.add_argument("--cnn_thr", type=float, default=None)
    args = ap.parse_args()

    section(f"Stage 0  pc_genes lookup for {args.gene}")
    gi = get_pc_row(args.pc_genes_bed, args.gene)
    if gi is None:
        print(f"FAIL: gene {args.gene!r} not in {args.pc_genes_bed}")
        print(f"      No parquet would be scanned for this gene. STOP.")
        sys.exit(1)
    span = gi["end"] - gi["start"]
    print(f"OK   {gi['chrom']}:{gi['start']:,}-{gi['end']:,} "
          f"strand={gi['strand']} span={span:,} bp")

    if args.site_offset is not None:
        site_gs = offset_to_genomic_start(args.site_offset, gi)
        site_ge = site_gs + 500
        print(f"     focal site: offset {args.site_offset} -> "
              f"genomic window {gi['chrom']}:{site_gs:,}-{site_ge:,} "
              f"({gi['strand']})")
    else:
        site_gs = site_ge = None

    section(f"Stage 1  raw parquet {args.gene}.parquet")
    pq = os.path.join(args.scan_dir, "per_gene", f"{args.gene}.parquet")
    if not os.path.exists(pq):
        print(f"FAIL: parquet not found at {pq}")
        sys.exit(1)
    mtime = os.path.getmtime(pq)
    print(f"OK   {pq}  (mtime: {pd.Timestamp(mtime, unit='s'):%Y-%m-%d %H:%M:%S})")
    df = pd.read_parquet(pq)
    print(f"     {len(df):,} windows; columns: {list(df.columns)}")
    if "cnn_scan_version" in df.columns:
        v = df["cnn_scan_version"].iloc[0]
        print(f"     cnn_scan_version (row 0): {v}")
    if "aparent2_scan_version" in df.columns:
        v = df["aparent2_scan_version"].iloc[0]
        print(f"     aparent2_scan_version (row 0): {v}")

    if args.site_offset is not None:
        lo = args.site_offset - args.tolerance
        hi = args.site_offset + args.tolerance
        sub = df[df["window_offset"].between(lo, hi)].copy()
        if len(sub) == 0:
            print(f"     no windows in [{lo}, {hi}]; widen --tolerance")
        else:
            print(f"\n     PROBS near offset {args.site_offset} "
                  f"(±{args.tolerance}, showing where any model >= 0.5):")
            cols = ["window_offset"] + [PROB_COL[m] for m in MODELS_3WAY
                                         if PROB_COL[m] in sub.columns]
            if PROB_COL["aparent2"] in sub.columns:
                cols.append(PROB_COL["aparent2"])
            if "is_constitutive_cpa" in sub.columns:
                cols.append("is_constitutive_cpa")
            sub_show = sub[cols].copy()
            mask = pd.Series(False, index=sub_show.index)
            for m in MODELS_3WAY:
                if PROB_COL[m] in sub_show.columns:
                    mask |= sub_show[PROB_COL[m]] >= 0.5
            sub_show = sub_show[mask]
            if len(sub_show) > 60:
                print(f"     (showing 30 around the focal offset)")
                center = args.site_offset
                sub_show["d"] = (sub_show["window_offset"] - center).abs()
                sub_show = sub_show.nsmallest(60, "d").drop(columns="d")\
                                    .sort_values("window_offset")
            print(sub_show.to_string(index=False))

    section(f"Stage 2  peak caches in {args.figures_dir}")
    caches = find_peak_caches(args.figures_dir, MODELS_3WAY)
    if not caches:
        print(f"FAIL: no _cache_peaks_*_*.parquet under {args.figures_dir}")
        print(f"      Run make_figure1.py once to generate them, then re-run.")
        sys.exit(1)

    thr_from_cache = {}
    for m in MODELS_3WAY:
        if m not in caches:
            print(f"  {m}: NO CACHE FOUND")
            continue
        path = caches[m]
        thr = parse_threshold_from_filename(path)
        thr_from_cache[m] = thr
        mtime = pd.Timestamp(os.path.getmtime(path), unit="s")
        print(f"  {m}:  thr={thr}  mtime={mtime:%Y-%m-%d %H:%M}")
        print(f"        {path}")

    cli_thr = {"rf": args.rf_thr, "lr": args.lr_thr, "cnn": args.cnn_thr}
    for m, v in cli_thr.items():
        if v is not None:
            print(f"  override {m} thr: cache says {thr_from_cache.get(m)}, "
                  f"CLI says {v}")
            thr_from_cache[m] = v

    print(f"\n  peaks per model for gene {args.gene}:")
    peak_centers = {}
    for m in MODELS_3WAY:
        if m not in caches:
            peak_centers[m] = []
            continue
        cache_df = pd.read_parquet(caches[m])
        sub = cache_df[cache_df["gene"] == args.gene].copy()
        if len(sub) == 0:
            print(f"\n    {m}: NO PEAKS for {args.gene} in cache")
            print(f"         (above-threshold filter rejected every window)")
            peak_centers[m] = []
            continue
        sub["center"] = ((sub["start"] + sub["end"]) // 2).astype(int)
        sub["offset"] = sub["start"].apply(lambda s: genomic_to_offset(s, gi))
        peak_centers[m] = sub.sort_values("center")
        print(f"\n    {m}: {len(sub):,} peaks for {args.gene}")
        show = sub.sort_values("max_prob", ascending=False)\
                  [["chrom", "start", "end", "strand", "max_prob",
                    "is_canonical", "center", "offset"]]
        if args.site_offset is not None:
            show = show.copy()
            show["near_focus"] = (show["offset"] - args.site_offset).abs() \
                                  <= args.tolerance
            near = show[show["near_focus"]]
            if len(near):
                print(f"      PEAKS NEAR FOCUS (±{args.tolerance} nt of "
                      f"offset {args.site_offset}):")
                print(near.drop(columns="near_focus").to_string(index=False))
            else:
                print(f"      NO PEAK within ±{args.tolerance} nt of offset "
                      f"{args.site_offset}  <-- this is the problem stage")
                show["abs_d"] = (show["offset"] - args.site_offset).abs()
                closest = show.nsmallest(3, "abs_d")
                print(f"      closest 3 peaks (by transcript-offset distance):")
                print(closest.drop(columns=["near_focus", "abs_d"])
                              .to_string(index=False))
        else:
            print(f"      top-5 by max_prob:")
            print(show.head(5).to_string(index=False))

    section(f"Stage 3  intragenic filter (subtract three_prime_zone.bed)")
    if not os.path.exists(args.three_prime_zone_bed):
        print(f"SKIP: {args.three_prime_zone_bed} not found")
        intragenic_peaks = peak_centers
    else:
        intragenic_peaks = {}
        for m in MODELS_3WAY:
            if not isinstance(peak_centers.get(m), pd.DataFrame):
                intragenic_peaks[m] = []
                continue
            df_pks = peak_centers[m].reset_index(drop=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".bed", delete=False,
                                                mode="w").name
            sub = df_pks.assign(idx=np.arange(len(df_pks)), score=0)[[
                "chrom", "start", "end", "idx", "score", "strand"]]
            sub.to_csv(tmp, sep="\t", header=False, index=False)
            subprocess.run(f"sort -k1,1 -k2,2n {tmp} -o {tmp}",
                           shell=True, check=True)
            out_p = tmp + ".intragenic"
            subprocess.run(
                ["bedtools", "intersect", "-v", "-s",
                 "-a", tmp, "-b", args.three_prime_zone_bed],
                stdout=open(out_p, "w"), check=True)
            kept_ids = set()
            if os.path.getsize(out_p) > 0:
                for line in open(out_p):
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 4:
                        kept_ids.add(int(p[3]))
            os.remove(tmp); os.remove(out_p)
            kept = df_pks.iloc[sorted(kept_ids)].reset_index(drop=True)
            n_drop = len(df_pks) - len(kept)
            print(f"  {m}: {len(df_pks):,} -> {len(kept):,} intragenic "
                  f"({n_drop:,} removed by 3'UTR filter)")
            intragenic_peaks[m] = kept
            if args.site_offset is not None and len(kept):
                near = kept[(kept["offset"] - args.site_offset).abs()
                            <= args.tolerance]
                if len(near):
                    print(f"      kept {len(near):,} peak(s) near focus "
                          f"(offsets: {sorted(near['offset'].tolist())})")
                else:
                    if len(df_pks):
                        was = df_pks[(df_pks["offset"] - args.site_offset).abs()
                                      <= args.tolerance]
                        if len(was):
                            print(f"      WARN: {len(was)} peak(s) near focus "
                                  f"existed BEFORE filter but were 3'UTR-removed:")
                            print(was[["start", "end", "strand", "max_prob",
                                        "offset"]].to_string(index=False))

    section(f"Stage 4  cluster_all_models simulation (slack={args.slack} nt)")
    items = []
    for m in MODELS_3WAY:
        df_m = intragenic_peaks.get(m)
        if not isinstance(df_m, pd.DataFrame) or len(df_m) == 0:
            continue
        for _, row in df_m.iterrows():
            items.append((int(row["center"]), m, int(row["offset"])))
    items.sort()

    if not items:
        print(f"  no intragenic peaks at all from any model -> nothing clusters")
    else:
        print(f"  {len(items):,} intragenic peaks total across "
              f"{len(MODELS_3WAY)} models")

    clusters = []
    for center, m, off in items:
        if clusters and (center - clusters[-1]["max_center"] <= args.slack):
            clusters[-1]["models"].add(m)
            clusters[-1]["centers"].append(center)
            clusters[-1]["offsets"].append(off)
            clusters[-1]["max_center"] = max(clusters[-1]["max_center"], center)
        else:
            clusters.append({"max_center": center,
                              "models": {m},
                              "centers": [center],
                              "offsets": [off]})

    tier_counts = {1: 0, 2: 0, 3: 0}
    three_way_clusters = []
    near_clusters = []
    for cl in clusters:
        n_m = len(cl["models"])
        tier_counts[n_m] = tier_counts.get(n_m, 0) + 1
        if n_m == 3:
            three_way_clusters.append(cl)
        if args.site_offset is not None:
            min_d = min(abs(o - args.site_offset) for o in cl["offsets"])
            if min_d <= args.tolerance:
                near_clusters.append((cl, min_d))

    print(f"\n  clustered into {len(clusters):,} clusters:")
    for k in sorted(tier_counts):
        print(f"    {k}-way clusters: {tier_counts[k]:,}")

    if args.site_offset is not None:
        if near_clusters:
            print(f"\n  CLUSTERS NEAR FOCUS (±{args.tolerance} nt of offset "
                  f"{args.site_offset}):")
            for cl, d in sorted(near_clusters, key=lambda t: t[1]):
                models_str = ",".join(sorted(cl["models"]))
                rep_center = int(np.median(cl["centers"]))
                rep_off = int(np.median(cl["offsets"]))
                marker = "  <-- 3-way!" if len(cl["models"]) == 3 else ""
                print(f"    rep_center={rep_center:,} rep_off={rep_off} "
                      f"models={{{models_str}}} ({len(cl['models'])}/3)"
                      f" closest_d={d}{marker}")
                print(f"      contributing offsets: "
                      f"{sorted(set(cl['offsets']))}")
        else:
            print(f"\n  NO CLUSTER within ±{args.tolerance} nt of focus.")
            print(f"  -> The site was already lost before clustering "
                  f"(see Stages 2/3).")

    section(f"Stage 5  final 3-way BED on disk")
    if not os.path.exists(args.three_way_bed):
        print(f"  not found: {args.three_way_bed}")
    else:
        n_total = sum(1 for _ in open(args.three_way_bed))
        gene_lines = []
        for line in open(args.three_way_bed):
            if args.gene in line:
                gene_lines.append(line.rstrip())
        print(f"  {args.three_way_bed}: {n_total:,} total entries")
        print(f"  entries containing '{args.gene}': {len(gene_lines)}")
        for line in gene_lines:
            print(f"    {line}")
        if args.site_offset is not None and gene_lines:
            print(f"  (focal offset: {args.site_offset}, "
                  f"genomic ~{site_gs:,}-{site_ge:,})")

    section("Diagnosis")
    if args.site_offset is None:
        print("  (no site_offset given; nothing to diagnose for a specific site)")
        return

    lost_at = None
    for m in MODELS_3WAY:
        df_m = peak_centers.get(m)
        if not isinstance(df_m, pd.DataFrame) or len(df_m) == 0:
            lost_at = f"Stage 2: model {m} has NO peaks for {args.gene} " \
                      "(threshold filter rejected everything)"
            break
        near = df_m[(df_m["offset"] - args.site_offset).abs()
                    <= args.tolerance]
        if len(near) == 0:
            lost_at = f"Stage 2: model {m} has peaks for {args.gene} but " \
                      f"NONE within ±{args.tolerance} nt of offset " \
                      f"{args.site_offset}"
            break

    if lost_at is None:
        for m in MODELS_3WAY:
            df_m = intragenic_peaks.get(m)
            if not isinstance(df_m, pd.DataFrame) or len(df_m) == 0:
                lost_at = f"Stage 3: 3'UTR filter removed all {m} peaks"
                break
            near = df_m[(df_m["offset"] - args.site_offset).abs()
                        <= args.tolerance]
            if len(near) == 0:
                lost_at = f"Stage 3: 3'UTR filter removed {m} peak near focus"
                break

    if lost_at is None:
        n3 = sum(1 for cl, _ in near_clusters if len(cl["models"]) == 3)
        if n3 == 0:
            lost_at = f"Stage 4: peaks exist for all 3 models near focus but " \
                      f"did NOT cluster into a single 3-way cluster " \
                      f"(slack={args.slack} nt). Try --slack 500 to test."
        else:
            lost_at = None

    if lost_at:
        print(f"  ❌  {lost_at}")
    else:
        print(f"  ✓  Pipeline simulation predicts the site IS a 3-way cluster.")
        print(f"     If the on-disk 3-way BED doesn't show it, the BED on")
        print(f"     disk is stale relative to the current peak caches. Rerun")
        print(f"     make_figure1.py to regenerate the BED.")


if __name__ == "__main__":
    main()