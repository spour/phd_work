#!/usr/bin/env python3
# extract cpccpa site coordinates
import argparse
import glob
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd

TARGET_MODELS = ["rf", "lr", "cnn"]


def find_peak_caches(figures_dir, models):
    cache_files = glob.glob(os.path.join(figures_dir, "_cache_peaks_*.parquet"))
    model_to_path = {}
    for cf in cache_files:
        base = os.path.basename(cf)
        toks = base.replace("_cache_peaks_", "").split("_thr")
        if not toks:
            continue
        m = toks[0]
        if m in models:
            if (m not in model_to_path or
                os.path.getmtime(cf) > os.path.getmtime(model_to_path[m])):
                model_to_path[m] = cf
    return model_to_path


def cluster_all_three(peaks_dict, slack=250):
    by_gene = defaultdict(list)
    for mi, m in enumerate(TARGET_MODELS):
        df = peaks_dict[m]
        for _, row in df.iterrows():
            center = (row["start"] + row["end"]) // 2
            key = (row["gene"], row["strand"])
            by_gene[key].append((center, mi, row["chrom"]))

    all_three = []
    for (gene, strand), items in by_gene.items():
        items.sort()
        clusters = []
        for center, mi, chrom in items:
            if clusters and (center - clusters[-1]["max_center"] <= slack):
                clusters[-1]["models"].add(mi)
                clusters[-1]["centers"].append(center)
                clusters[-1]["max_center"] = max(clusters[-1]["max_center"], center)
            else:
                clusters.append({"max_center": center, "models": {mi},
                                 "centers": [center], "chrom": chrom})
        for cl in clusters:
            if len(cl["models"]) == 3:
                rep = int(np.median(cl["centers"]))
                all_three.append({"chrom": cl["chrom"], "center": rep,
                                  "gene": gene, "strand": strand})
    return all_three


def write_bed(records, path, halfwidth=0):
    with open(path, "w") as f:
        for i, r in enumerate(records):
            start = max(0, r["center"] - halfwidth)
            end = r["center"] + halfwidth + 1
            f.write(f"{r['chrom']}\t{start}\t{end}\t"
                    f"cpccpa_{i}_{r['gene']}\t0\t{r['strand']}\n")


def bedtools_filter(in_bed, mask_bed, keep_overlap, slack=0, strand=True):
    mask = mask_bed
    if slack > 0:
        mask_t = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
        with open(mask_bed) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                s = max(0, int(p[1]) - slack); e = int(p[2]) + slack
                strand_c = p[5] if len(p) > 5 else "+"
                nm = p[3] if len(p) > 3 else "."
                mask_t.write(f"{p[0]}\t{s}\t{e}\t{nm}\t.\t{strand_c}\n")
        mask_t.close()
        mask = mask_t.name

    out = in_bed + ".filt"
    flag = "-u" if keep_overlap else "-v"
    cmd = ["bedtools", "intersect", flag, "-a", in_bed, "-b", mask]
    if strand:
        cmd.insert(2, "-s")
    subprocess.run(cmd, stdout=open(out, "w"), check=True)
    if slack > 0:
        os.remove(mask)
    os.replace(out, in_bed)
    return sum(1 for _ in open(in_bed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures_dir", required=True)
    ap.add_argument("--exclude_bed", action="append", default=[],
                    help="known-CPA BED(s) to exclude; repeatable")
    ap.add_argument("--introns_bed", default=None,
                    help="optional: also restrict to intronic")
    ap.add_argument("--out_bed", required=True)
    ap.add_argument("--slack", type=int, default=250,
                    help="clustering linkage distance")
    ap.add_argument("--exclude_slack", type=int, default=250,
                    help="exclude CPCCPA within this many nt of a known CPA")
    ap.add_argument("--site_halfwidth", type=int, default=250,
                    help="half-width of emitted site window (250 -> 500nt)")
    args = ap.parse_args()

    if not args.exclude_bed:
        sys.exit("need at least one --exclude_bed (the known-CPA set)")

    print("[1/3] Loading peak caches ...")
    cache_paths = find_peak_caches(args.figures_dir, TARGET_MODELS)
    missing = [m for m in TARGET_MODELS if m not in cache_paths]
    if missing:
        sys.exit(f"missing peak caches: {missing}")
    peaks = {}
    for m in TARGET_MODELS:
        peaks[m] = pd.read_parquet(cache_paths[m])
        print(f"  {m}: {len(peaks[m]):,} peaks")

    print(f"[2/3] Clustering all-three agreement (slack {args.slack}) ...")
    rec = cluster_all_three(peaks, slack=args.slack)
    print(f"  all-three clusters: {len(rec):,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_bed)), exist_ok=True)
    write_bed(rec, args.out_bed, halfwidth=args.site_halfwidth)
    subprocess.run(f"sort -k1,1 -k2,2n {args.out_bed} -o {args.out_bed}",
                   shell=True, check=True)

    print(f"[3/3] Excluding proximity to known CPAs (+/- {args.exclude_slack} nt) ...")
    for xb in args.exclude_bed:
        if not os.path.exists(xb):
            print(f"  WARN missing {xb}; skipping")
            continue
        n = bedtools_filter(args.out_bed, xb, keep_overlap=False,
                            slack=args.exclude_slack, strand=True)
        print(f"  after excluding {os.path.basename(xb)}: {n:,}")

    if args.introns_bed:
        print(f"  (optional) restricting to intronic ...")
        n = bedtools_filter(args.out_bed, args.introns_bed,
                            keep_overlap=True, strand=True)
        print(f"  intronic: {n:,}")

    n_final = sum(1 for _ in open(args.out_bed))
    print(f"\nDone. CPCCPA sites: {n_final:,}")
    print(f"  wrote {args.out_bed}")


if __name__ == "__main__":
    main()