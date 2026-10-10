#!/usr/bin/env python3
# 3-way venn diagram for rf/lr/cnn agreement
import argparse
import glob
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp


PAL = {"purple": "#7F77DD", "amber": "#BA7517", "teal": "#1D9E75"}
MODEL_INFO = {
    "rf":  {"display": "RF",  "color": PAL["purple"]},
    "lr":  {"display": "LR",  "color": PAL["amber"]},
    "cnn": {"display": "CNN", "color": PAL["teal"]},
}
TARGET_MODELS = ["rf", "lr", "cnn"]


def read_bed(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            p = line.rstrip("\n").split("\t")
            rows.append({"chrom": p[0], "start": int(p[1]), "end": int(p[2]),
                         "name": p[3] if len(p) > 3 else "",
                         "strand": p[5] if len(p) > 5 else "+"})
    return rows


def find_peak_caches(figures_dir, models):
    cache_files = glob.glob(os.path.join(figures_dir,
                                           "_cache_peaks_*.parquet"))
    model_to_path = {}
    for cf in cache_files:
        base = os.path.basename(cf)
        toks = base.replace("_cache_peaks_", "").split("_thr")
        if not toks: continue
        m = toks[0]
        if m in models:
            if (m not in model_to_path or
                os.path.getmtime(cf) > os.path.getmtime(model_to_path[m])):
                model_to_path[m] = cf
    return model_to_path


def bedtools_overlap_sites(sites_df, peaks_df, slack=250):
    if len(sites_df) == 0 or len(peaks_df) == 0:
        return set()
    sites_t = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
    sdf = sites_df.reset_index(drop=True)
    sdf.assign(idx=np.arange(len(sdf)))[[
        "chrom", "start", "end", "idx", "name", "strand"
    ]].to_csv(sites_t.name, sep="\t", header=False, index=False)
    sites_t.close()

    peaks_t = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
    pdf = peaks_df.copy()
    pdf["start"] = (pdf["start"] - slack).clip(lower=0)
    pdf["end"]   = pdf["end"] + slack
    pdf.assign(score=0)[[
        "chrom", "start", "end", "gene", "score", "strand"
    ]].to_csv(peaks_t.name, sep="\t", header=False, index=False)
    peaks_t.close()

    out = sites_t.name + ".hits"
    subprocess.run(["bedtools", "intersect", "-u", "-s",
                    "-a", sites_t.name, "-b", peaks_t.name],
                   stdout=open(out, "w"), check=True)
    hits = set()
    if os.path.getsize(out) > 0:
        h = pd.read_csv(out, sep="\t", header=None, names=[
            "chrom", "start", "end", "idx", "name", "strand"])
        hits = set(h["idx"].astype(int).tolist())
    os.remove(sites_t.name); os.remove(peaks_t.name); os.remove(out)
    return hits


def cluster_all_peaks_three_way(peaks_dict, slack=250):
    by_gene = defaultdict(list)
    for mi, m in enumerate(TARGET_MODELS):
        df = peaks_dict[m]
        if len(df) == 0: continue
        for _, row in df.iterrows():
            center = (row["start"] + row["end"]) // 2
            by_gene[(row["gene"], row["strand"])].append((center, mi))

    counts = defaultdict(int)
    total_clusters = 0
    for key, items in by_gene.items():
        items.sort()
        clusters = []
        for center, mi in items:
            if clusters and (center - clusters[-1]["max_center"] <= slack):
                clusters[-1]["models"].add(mi)
                clusters[-1]["max_center"] = max(clusters[-1]["max_center"],
                                                   center)
            else:
                clusters.append({"max_center": center, "models": {mi}})
        for cl in clusters:
            sig = "".join("1" if i in cl["models"] else "0"
                          for i in range(3))
            counts[sig] += 1
            total_clusters += 1

    return dict(counts), total_clusters


def draw_venn3(regions, set_labels, colors, set_totals, title, subtitle,
                out_path, footer=None):
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=500, facecolor="white")
    ax.set_aspect("equal"); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.2, 2.6)
    ax.axis("off")
    r = 1.1; dx = 0.6; dy = 0.55
    centers = [(-dx, -dy), (dx, -dy), (0, dy)]
    for (cx, cy), color in zip(centers, colors):
        circ = mp.Circle((cx, cy), r, facecolor=color, edgecolor="black",
                         linewidth=0.8, alpha=0.32, zorder=2)
        ax.add_patch(circ)
    pos = {"100": (-1.2, -0.95),
           "010": ( 1.2, -0.95),
           "001": ( 0.0,  1.65),
           "110": ( 0.0, -1.15),
           "101": (-0.85, 0.55),
           "011": ( 0.85, 0.55),
           "111": ( 0.0,  0.15)}
    for code, (x, y) in pos.items():
        n = regions.get(code, 0)
        if n == 0: continue
        ax.text(x, y, f"{n:,}", ha="center", va="center", fontsize=12,
                color="#2C2C2A", weight="bold", zorder=5)
    label_pos = [(-1.55, -1.85), (1.55, -1.85), (0.0, 2.1)]
    for (x, y), name, color in zip(label_pos, set_labels, colors):
        n_tot = set_totals.get(name, 0)
        ax.text(x, y, f"{name}\n(n = {n_tot:,})",
                ha="center", va="center", fontsize=12, color=color,
                weight="bold")
    fig.suptitle(title, fontsize=14, color="#2C2C2A",
                 x=0.5, ha="center", y=0.97)
    fig.text(0.5, 0.92, subtitle, fontsize=10, color="#5F5E5A",
             ha="center")
    if footer:
        fig.text(0.5, 0.04, footer, fontsize=9, color="#5F5E5A",
                 ha="center", style="italic")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def venn_from_site_hits(hits_per_model):
    a, b, c = (hits_per_model[m] for m in TARGET_MODELS)
    regions = {
        "100": len(a - b - c),
        "010": len(b - a - c),
        "001": len(c - a - b),
        "110": len((a & b) - c),
        "101": len((a & c) - b),
        "011": len((b & c) - a),
        "111": len(a & b & c),
    }
    set_totals = {MODEL_INFO[m]["display"]: len(hits_per_model[m])
                  for m in TARGET_MODELS}
    return regions, set_totals


def print_venn_breakdown(regions, set_totals, n_total, slack):
    a_only = regions.get("100", 0)
    b_only = regions.get("010", 0)
    c_only = regions.get("001", 0)
    ab     = regions.get("110", 0)
    ac     = regions.get("101", 0)
    bc     = regions.get("011", 0)
    abc    = regions.get("111", 0)
    union  = a_only + b_only + c_only + ab + ac + bc + abc
    missed = n_total - union
    print(f"  RF only:        {a_only:>8,}  ({a_only/n_total*100:5.1f}%)")
    print(f"  LR only:        {b_only:>8,}  ({b_only/n_total*100:5.1f}%)")
    print(f"  CNN only:       {c_only:>8,}  ({c_only/n_total*100:5.1f}%)")
    print(f"  RF & LR:        {ab:>8,}  ({ab/n_total*100:5.1f}%)")
    print(f"  RF & CNN:       {ac:>8,}  ({ac/n_total*100:5.1f}%)")
    print(f"  LR & CNN:       {bc:>8,}  ({bc/n_total*100:5.1f}%)")
    print(f"  All three:      {abc:>8,}  ({abc/n_total*100:5.1f}%)")
    print(f"  -------")
    print(f"  Union:          {union:>8,}  ({union/n_total*100:5.1f}%)")
    print(f"  Missed by all:  {missed:>8,}  ({missed/n_total*100:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical_bed", required=True)
    ap.add_argument("--cryptic_bed",   required=True)
    ap.add_argument("--figures_dir",   required=True,
                    help="Directory with _cache_peaks_<model>_*.parquet")
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--mode", default="all",
                    choices=["peaks", "canonical", "cryptic", "all"])
    ap.add_argument("--slack", type=int, default=250,
                    help="Two peaks within this many nt count as the same site "
                         "(default 250). Smaller = stricter agreement.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load peak caches
    print(f"Looking for peak caches in {args.figures_dir}")
    cache_paths = find_peak_caches(args.figures_dir, TARGET_MODELS)
    missing = [m for m in TARGET_MODELS if m not in cache_paths]
    if missing:
        sys.exit(f"missing peak caches for: {missing}")
    peaks = {}
    for m in TARGET_MODELS:
        peaks[m] = pd.read_parquet(cache_paths[m])
        print(f"  {MODEL_INFO[m]['display']:<4}: {len(peaks[m]):,} merged peaks  "
              f"({cache_paths[m]})")

    set_labels = [MODEL_INFO[m]["display"] for m in TARGET_MODELS]
    colors     = [MODEL_INFO[m]["color"]   for m in TARGET_MODELS]

    # 1) canonical-site venn
    if args.mode in ("canonical", "all"):
        print(f"\n[Canonical-site Venn]")
        sites = pd.DataFrame(read_bed(args.canonical_bed))
        n_total = len(sites)
        print(f"  total canonical sites: {n_total:,}")
        hits = {}
        for m in TARGET_MODELS:
            hits[m] = bedtools_overlap_sites(sites, peaks[m], slack=args.slack)
            print(f"  {MODEL_INFO[m]['display']:<4}: recovered "
                  f"{len(hits[m]):,} ({len(hits[m])/n_total*100:.1f}%)")
        regions, set_totals = venn_from_site_hits(hits)
        print(f"\n  Venn breakdown:")
        print_venn_breakdown(regions, set_totals, n_total, args.slack)
        out_path = os.path.join(args.out_dir,
                                 "fig_venn3_rf_lr_cnn_canonical.png")
        draw_venn3(regions, set_labels, colors, set_totals,
                    title="Canonical CPA recovery: RF, LR, CNN",
                    subtitle=(f"n = {n_total:,} canonical sites; "
                              f"+/- {args.slack} nt overlap allowance"),
                    out_path=out_path,
                    footer=f"Union: {sum(regions.values()):,} "
                           f"({sum(regions.values())/n_total*100:.1f}%)   "
                           f"Missed by all: {n_total - sum(regions.values()):,} "
                           f"({(n_total - sum(regions.values()))/n_total*100:.1f}%)")

    # 2) cryptic-site venn
    if args.mode in ("cryptic", "all"):
        print(f"\n[Cryptic-site Venn]")
        sites = pd.DataFrame(read_bed(args.cryptic_bed))
        n_total = len(sites)
        print(f"  total cryptic sites: {n_total:,}")
        hits = {}
        for m in TARGET_MODELS:
            hits[m] = bedtools_overlap_sites(sites, peaks[m], slack=args.slack)
            print(f"  {MODEL_INFO[m]['display']:<4}: recovered "
                  f"{len(hits[m]):,} ({len(hits[m])/n_total*100:.1f}%)")
        regions, set_totals = venn_from_site_hits(hits)
        print(f"\n  Venn breakdown:")
        print_venn_breakdown(regions, set_totals, n_total, args.slack)
        out_path = os.path.join(args.out_dir,
                                 "fig_venn3_rf_lr_cnn_cryptic.png")
        draw_venn3(regions, set_labels, colors, set_totals,
                    title="Cryptic CPA recovery: RF, LR, CNN",
                    subtitle=(f"n = {n_total:,} cryptic sites; "
                              f"+/- {args.slack} nt overlap allowance"),
                    out_path=out_path,
                    footer=f"Union: {sum(regions.values()):,} "
                           f"({sum(regions.values())/n_total*100:.1f}%)   "
                           f"Missed by all: {n_total - sum(regions.values()):,} "
                           f"({(n_total - sum(regions.values()))/n_total*100:.1f}%)")

    # 3) peak-cluster venn (no reference sites)
    if args.mode in ("peaks", "all"):
        print(f"\n[Peak-cluster Venn -- all high-confidence peaks across genome]")
        counts, total_clusters = cluster_all_peaks_three_way(
            peaks, slack=args.slack)
        regions = counts
        set_totals = {MODEL_INFO[m]["display"]:
                      sum(c for sig, c in counts.items() if sig[i] == "1")
                      for i, m in enumerate(TARGET_MODELS)}
        print(f"  total distinct peak clusters: {total_clusters:,}")
        print(f"\n  Venn breakdown:")
        print_venn_breakdown(regions, set_totals, total_clusters, args.slack)
        out_path = os.path.join(args.out_dir,
                                 "fig_venn3_rf_lr_cnn_peaks.png")
        draw_venn3(regions, set_labels, colors, set_totals,
                    title="High-confidence peak agreement: RF, LR, CNN",
                    subtitle=(f"{total_clusters:,} distinct peak clusters "
                              f"(strand-aware, +/- {args.slack} nt linkage)"),
                    out_path=out_path,
                    footer="Each cluster = peaks from any model within slack nt")

if __name__ == "__main__":
    main()