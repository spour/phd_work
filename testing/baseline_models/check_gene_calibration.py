#!/usr/bin/env python3
# visualize a single gene's cnn track before and after applying the fitted t
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EPS = 1e-7


def prob_to_logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def read_pc_gene(pc_bed, gene):
    with open(pc_bed) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if p[3] == gene:
                return {
                    "chrom":  p[0],
                    "start":  int(p[1]),
                    "end":    int(p[2]),
                    "strand": p[5] if len(p) > 5 else "+",
                }
    return None


def read_atlas_for_gene(atlas_beds, gene_info):
    out = []
    if not atlas_beds:
        return out
    if isinstance(atlas_beds, str):
        atlas_beds = [atlas_beds]
    chrom = gene_info["chrom"]
    strand = gene_info["strand"]
    for bp in atlas_beds:
        if not os.path.exists(bp):
            continue
        label = os.path.basename(bp)
        with open(bp) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if p[0] != chrom:
                    continue
                if len(p) > 5 and p[5] != strand:
                    continue
                mid = (int(p[1]) + int(p[2])) // 2
                if gene_info["start"] <= mid <= gene_info["end"]:
                    out.append((mid, label))
    return out


def to_transcript_kb(genomic_pos, gene_info):
    if gene_info["strand"] == "+":
        return (genomic_pos - gene_info["start"]) / 1000.0
    return (gene_info["end"] - genomic_pos) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir",          required=True)
    ap.add_argument("--pc_genes_bed",      required=True)
    ap.add_argument("--gene",              required=True,
                    help="Gene name, e.g. FGF20")
    ap.add_argument("--calibration_json",  required=True)
    ap.add_argument("--model", default="cnn", choices=["rf", "lr", "cnn"])
    ap.add_argument("--atlas_bed", default=None, nargs="+",
                    help="Optional atlas BED(s) for tick marks")
    ap.add_argument("--out", required=True, help="Output figure path (.png)")
    args = ap.parse_args()

    # load data
    pq_path = os.path.join(args.scan_dir, "per_gene", f"{args.gene}.parquet")
    if not os.path.exists(pq_path):
        raise SystemExit(f"ERROR: no parquet at {pq_path}")
    df = pd.read_parquet(pq_path)
    gi = read_pc_gene(args.pc_genes_bed, args.gene)
    if gi is None:
        raise SystemExit(f"ERROR: {args.gene} not in {args.pc_genes_bed}")
    with open(args.calibration_json) as f:
        cal = json.load(f)
    T = float(cal["T"])
    print(f"Loaded T = {T:.4f} (model={cal.get('model', args.model)})")
    print(f"Gene: {args.gene}  {gi['chrom']}:{gi['start']:,}-{gi['end']:,} "
          f"({gi['strand']}, {(gi['end']-gi['start'])/1000:.1f} kb)")
    print(f"Windows in parquet: {len(df):,}")

    pcol = f"{args.model}_prob"
    raw = df[pcol].values.astype(np.float64)
    logits = prob_to_logit(raw)
    cal_p = sigmoid(logits / T)

    offsets = df["window_offset"].values
    x_kb = (offsets + 250) / 1000.0

    canon_kb = None; ci = None
    if df["is_constitutive_cpa"].any():
        ci = int(df.index[df["is_constitutive_cpa"]][0])
        canon_off = int(df["window_offset"].iloc[ci])
        canon_kb  = (canon_off + 250) / 1000.0

    atlas_sites = read_atlas_for_gene(args.atlas_bed, gi) if args.atlas_bed else []
    atlas_kbs = [to_transcript_kb(m, gi) for m, _ in atlas_sites]

    # print summary
    print(f"\n  --- Summary for {args.gene} ---")
    print(f"  Logit range from stored prob: [{logits.min():.2f}, "
          f"{logits.max():.2f}]")
    n_capped = int((np.abs(logits) > 15.5).sum())
    print(f"  Windows at logit cap (|logit|>15.5): {n_capped:,}  "
          f"({n_capped/len(logits)*100:.1f}%)")
    print(f"  {'':>26} {'raw':>10} {'calibrated':>12}")
    for thr in [0.99, 0.95, 0.90, 0.50]:
        nr = int((raw   >= thr).sum())
        nc = int((cal_p >= thr).sum())
        print(f"  Windows with prob >= {thr}: {'':>4} {nr:>10,} {nc:>12,}")
    print(f"  {'min / max prob':>26} "
          f"{raw.min():>5.3f}/{raw.max():>5.3f}  "
          f"{cal_p.min():>5.3f}/{cal_p.max():>5.3f}")

    if ci is not None:
        away = np.abs(offsets - df["window_offset"].iloc[ci]) >= 500
        max_other_raw = float(raw[away].max())   if away.any() else 0.0
        max_other_cal = float(cal_p[away].max()) if away.any() else 0.0
        rank_raw = int((raw   > raw[ci]).sum()) + 1
        rank_cal = int((cal_p > cal_p[ci]).sum()) + 1
        print(f"\n  Canonical window stats:")
        print(f"    canonical prob:        raw={raw[ci]:.4f}    "
              f"cal={cal_p[ci]:.4f}")
        print(f"    max non-canonical:     raw={max_other_raw:.4f}    "
              f"cal={max_other_cal:.4f}")
        print(f"    margin (canon - other): raw={raw[ci]-max_other_raw:+.4f}    "
              f"cal={cal_p[ci]-max_other_cal:+.4f}")
        print(f"    strict rank of canon:  raw={rank_raw}        cal={rank_cal}    "
              f"(should be equal; T-scaling is monotonic)")

    # plot
    fig = plt.figure(figsize=(12, 8), dpi=300, facecolor="white")
    gs  = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0],
                            width_ratios=[3, 1],
                            hspace=0.30, wspace=0.25,
                            left=0.07, right=0.97, top=0.92, bottom=0.08)

    order = np.argsort(x_kb)
    xs    = x_kb[order]
    yr    = raw[order]
    yc    = cal_p[order]

    axA = fig.add_subplot(gs[0, 0])
    axA.fill_between(xs, 0, yr, color="#1D9E75", alpha=0.20, linewidth=0)
    axA.plot(xs, yr, color="#1D9E75", linewidth=0.7)
    if canon_kb is not None:
        axA.axvline(canon_kb, color="black", linewidth=0.9, linestyle="--",
                    alpha=0.55, zorder=3)
    for ak in atlas_kbs:
        axA.plot([ak], [1.02], marker="v", color="#888780", markersize=4,
                 clip_on=False)
    axA.set_ylim(0, 1.05)
    axA.set_ylabel("CNN prob (raw)", fontsize=10)
    axA.set_title(f"A. {args.gene} — raw CNN probability",
                  fontsize=11, loc="left", color="#2C2C2A")

    axB = fig.add_subplot(gs[1, 0], sharex=axA)
    axB.fill_between(xs, 0, yc, color="#7F77DD", alpha=0.20, linewidth=0)
    axB.plot(xs, yc, color="#7F77DD", linewidth=0.7)
    if canon_kb is not None:
        axB.axvline(canon_kb, color="black", linewidth=0.9, linestyle="--",
                    alpha=0.55, zorder=3)
    for ak in atlas_kbs:
        axB.plot([ak], [1.02], marker="v", color="#888780", markersize=4,
                 clip_on=False)
    axB.set_ylim(0, 1.05)
    axB.set_ylabel(f"CNN prob (T={T:.2f})", fontsize=10)
    axB.set_title(f"B. {args.gene} — calibrated CNN probability",
                  fontsize=11, loc="left", color="#2C2C2A")

    axC = fig.add_subplot(gs[2, 0], sharex=axA)
    axC.plot(xs, yr, color="#1D9E75", linewidth=0.7, label="raw",   alpha=0.7)
    axC.plot(xs, yc, color="#7F77DD", linewidth=0.7, label="cal.",  alpha=0.7)
    if canon_kb is not None:
        axC.axvline(canon_kb, color="black", linewidth=0.9, linestyle="--",
                    alpha=0.55, zorder=3)
    axC.set_yscale("log")
    axC.set_ylim(1e-4, 1.05)
    axC.set_ylabel("CNN prob (log)", fontsize=10)
    axC.set_xlabel(f"Position along {args.gene}, 5' → 3' (kb)",
                   fontsize=10, color="#2C2C2A")
    axC.set_title("C. Raw vs calibrated, log y-axis",
                  fontsize=11, loc="left", color="#2C2C2A")
    axC.legend(loc="upper right", fontsize=9, frameon=False)

    axH = fig.add_subplot(gs[:, 1])
    bins = np.linspace(0, 1, 41)
    axH.hist(raw,   bins=bins, color="#1D9E75", alpha=0.55, label="raw",
             orientation="horizontal", edgecolor="black", linewidth=0.3)
    axH.hist(cal_p, bins=bins, color="#7F77DD", alpha=0.55, label="cal.",
             orientation="horizontal", edgecolor="black", linewidth=0.3)
    axH.set_xscale("log")
    axH.set_ylim(0, 1.0)
    axH.set_xlabel("# windows (log)", fontsize=9, color="#2C2C2A")
    axH.set_title("Distribution of probs", fontsize=10, loc="left",
                  color="#2C2C2A")
    axH.legend(loc="center right", fontsize=8, frameon=False)

    for ax in [axA, axB, axC, axH]:
        for s in ax.spines.values():
            s.set_color("black"); s.set_linewidth(0.8)
        ax.tick_params(colors="#444441", labelsize=8)
        ax.grid(axis="y" if ax is not axH else "x",
                color="#D3D1C7", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)

    title_bits = [f"{gi['chrom']}:{gi['start']:,}-{gi['end']:,} "
                  f"(strand {gi['strand']}, "
                  f"{(gi['end']-gi['start'])/1000:.1f} kb)"]
    if canon_kb is not None:
        title_bits.append("dashed line = canonical CPA")
    if atlas_kbs:
        title_bits.append(f"▼ atlas ({len(atlas_kbs)} sites)")
    fig.suptitle(f"Calibration check: {args.model.upper()} on {args.gene} "
                 f"(T = {T:.2f})",
                 fontsize=13, color="#2C2C2A", x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.95, "  ·  ".join(title_bits),
             fontsize=9, color="#5F5E5A", ha="left")

    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()