#!/usr/bin/env python3
# test whether cpccpa sites show experimental usage
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PAL = {"teal": "#1D9E75", "coral": "#D85A30", "gray": "#888780"}
CLASS_COLOR = {"Constitutive": PAL["teal"], "CPCCPA": PAL["coral"],
               "Random intronic": PAL["gray"]}
CLASS_ORDER = ["Constitutive", "CPCCPA", "Random intronic"]


def read_bed_n(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            score = np.nan
            if len(p) > 4:
                try: score = float(p[4])
                except ValueError: score = np.nan
            rows.append({"chrom": p[0], "start": int(p[1]), "end": int(p[2]),
                         "name": p[3] if len(p) > 3 else "",
                         "score": score,
                         "strand": p[5] if len(p) > 5 else "+"})
    return pd.DataFrame(rows)


def overlap_with_expr(sites_df, ref_bed, slack=50):
    if len(sites_df) == 0:
        return np.array([], dtype=bool), np.array([])
    sites_t = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
    s = sites_df.reset_index(drop=True)
    s2 = s.copy()
    s2["start"] = (s2["start"] - slack).clip(lower=0)
    s2["end"]   = s2["end"] + slack
    s2.assign(idx=np.arange(len(s2)))[[
        "chrom", "start", "end", "idx", "score", "strand"
    ]].to_csv(sites_t.name, sep="\t", header=False, index=False)
    sites_t.close()

    res = subprocess.run(
        ["bedtools", "intersect", "-wa", "-wb", "-s",
         "-a", sites_t.name, "-b", ref_bed],
        stdout=subprocess.PIPE, check=True, text=True)
    os.remove(sites_t.name)

    hit_mask = np.zeros(len(s), dtype=bool)
    best_score = np.full(len(s), np.nan)
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        idx = int(parts[3])
        hit_mask[idx] = True
        try:
            ref_score = float(parts[6 + 4])
        except (ValueError, IndexError):
            ref_score = np.nan
        if not np.isnan(ref_score):
            if np.isnan(best_score[idx]) or ref_score > best_score[idx]:
                best_score[idx] = ref_score
    return hit_mask, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpccpa_bed",       required=True)
    ap.add_argument("--constitutive_bed", required=True)
    ap.add_argument("--random_bed",       required=True)
    ap.add_argument("--polyasite_bed",    required=True)
    ap.add_argument("--polyadb_main_bed", required=True)
    ap.add_argument("--polyadb_max_bed",  required=True)
    ap.add_argument("--out_dir",          required=True)
    ap.add_argument("--slack", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[Loading site classes]")
    sites = {
        "Constitutive":    read_bed_n(args.constitutive_bed),
        "CPCCPA":          read_bed_n(args.cpccpa_bed),
        "Random intronic": read_bed_n(args.random_bed),
    }
    for c, df in sites.items():
        print(f"  {c:<18}: {len(df):,} sites")

    refs = [
        ("PolyASite 2.0\n(curated, filtered on)", args.polyasite_bed),
        ("PolyA_DB v4 Main\n(validated)",         args.polyadb_main_bed),
        ("PolyA_DB v4 Max\n(permissive, 1.4M)",   args.polyadb_max_bed),
    ]

    print("\n[Panel A] Overlap rate per class per reference")
    rowsA = []
    max_hit = {}; max_score = {}
    for cls, df in sites.items():
        for ref_label, ref_path in refs:
            if not os.path.exists(ref_path):
                print(f"  WARN missing {ref_path}; skipping")
                continue
            hit, score = overlap_with_expr(df, ref_path, slack=args.slack)
            rate = hit.mean() * 100 if len(hit) else 0
            rowsA.append({"Class": cls, "Reference": ref_label,
                          "rate": rate, "n_hit": int(hit.sum()),
                          "n_total": len(df)})
            print(f"  {cls:<18} vs {ref_label.split(chr(10))[0]:<22}: "
                  f"{int(hit.sum()):>7,}/{len(df):<7,} = {rate:5.1f}%")
            if "Max" in ref_label:
                max_hit[cls] = hit; max_score[cls] = score

    dfA = pd.DataFrame(rowsA)
    ref_order = [r[0] for r in refs]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300, facecolor="white")
    n_ref = len(ref_order)
    bar_w = 0.25
    x = np.arange(n_ref)
    for i, cls in enumerate(CLASS_ORDER):
        vals = []
        for r in ref_order:
            v = dfA[(dfA["Class"] == cls) & (dfA["Reference"] == r)]["rate"].values
            vals.append(v[0] if len(v) else 0)
        ax.bar(x + (i - 1) * bar_w, vals, bar_w, label=cls,
               color=CLASS_COLOR[cls], edgecolor="black", linewidth=0.4)
        for xi, v in zip(x + (i - 1) * bar_w, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=8, color="#2C2C2A")
    ax.set_xticks(x); ax.set_xticklabels(ref_order, fontsize=9)
    ax.set_ylabel("Sites overlapping reference (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title("Even against the most permissive atlas, CPCCPA sites are absent",
                 fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9)
    for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.6)
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    pathA = os.path.join(args.out_dir, "figU_overlap_rate.png")
    fig.savefig(pathA, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {pathA}")

    print("\n[Panel B] Expression (RPM) where sites overlap PolyA_DB Max")
    rowsB = []
    for cls in CLASS_ORDER:
        if cls not in max_score:
            continue
        sc = max_score[cls]; sc = sc[~np.isnan(sc)]
        for v in sc:
            rowsB.append({"Class": cls, "rpm": v})
        if len(sc):
            print(f"  {cls:<18}: n={len(sc):,} overlapping, "
                  f"median RPM={np.median(sc):.3f}")
        else:
            print(f"  {cls:<18}: 0 overlapping Max")
    dfB = pd.DataFrame(rowsB)
    if len(dfB) > 0 and dfB["rpm"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300, facecolor="white")
        data = []; labels = []; colors = []
        for cls in CLASS_ORDER:
            sub = dfB[dfB["Class"] == cls]["rpm"].values
            sub = sub[sub > 0]
            if len(sub) == 0:
                continue
            data.append(np.log10(sub))
            labels.append(f"{cls}\n(n={len(sub):,})")
            colors.append(CLASS_COLOR[cls])
        if data:
            parts = ax.violinplot(data, showmedians=True, showextrema=False)
            for pc, col in zip(parts["bodies"], colors):
                pc.set_facecolor(col); pc.set_alpha(0.7)
                pc.set_edgecolor("black"); pc.set_linewidth(0.5)
            parts["cmedians"].set_color("black")
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("log10(RPM) where site overlaps PolyA_DB Max", fontsize=11)
            ax.set_title("The few CPCCPA sites that overlap are barely expressed",
                         fontsize=12, loc="left", pad=10)
            for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.6)
            ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
            ax.set_axisbelow(True)
            fig.tight_layout()
            pathB = os.path.join(args.out_dir, "figU_expression.png")
            fig.savefig(pathB, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  wrote {pathB}")
    else:
        print("  no expression data in overlaps; skipping Panel B")

    summary_path = os.path.join(args.out_dir, "figU_summary.csv")
    dfA.to_csv(summary_path, index=False)
    print(f"\n  wrote summary {summary_path}")


if __name__ == "__main__":
    main()