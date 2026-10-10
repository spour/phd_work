#!/usr/bin/env python3
# render the score traces stored in a single per-gene parquet
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS = {
    "rf_prob":      "#7F77DD",
    "lr_prob":      "#BA7517",
    "cnn_prob":     "#1D9E75",
    "cnn_prob_v1":  "#888780",
    "cnn_logit":    "#D85A30",
}


def read_pc_gene(pc_bed, gene):
    with open(pc_bed) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if p[3] == gene:
                return {"chrom":  p[0], "start": int(p[1]),
                        "end":    int(p[2]),
                        "strand": p[5] if len(p) > 5 else "+"}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--gene",    default=None,
                    help="Optional gene name shown in the title")
    ap.add_argument("--pc_genes_bed", default=None,
                    help="Optional pc_genes BED to label the x-axis in kb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", nargs="+", default=None,
                    help="Which columns to plot. Default: every column "
                         "ending in '_prob' plus cnn_prob_v1 if present")
    ap.add_argument("--include_logit", action="store_true",
                    help="Also plot cnn_logit if present")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {args.parquet}")
    print(f"  {len(df):,} windows")
    print(f"  columns: {list(df.columns)}")

    if args.cols:
        cols = [c for c in args.cols if c in df.columns]
    else:
        cols = [c for c in df.columns if c.endswith("_prob")]
        if "cnn_prob_v1" in df.columns:
            cols.append("cnn_prob_v1")
    if args.include_logit and "cnn_logit" in df.columns:
        cols.append("cnn_logit")
    cols = list(dict.fromkeys(cols))
    if not cols:
        raise SystemExit("No plottable columns found in parquet.")
    print(f"  plotting: {cols}")

    offsets = df["window_offset"].values if "window_offset" in df.columns \
              else np.arange(len(df))
    gi = None
    if args.gene and args.pc_genes_bed:
        gi = read_pc_gene(args.pc_genes_bed, args.gene)
    if gi is not None:
        x = (offsets + 250) / 1000.0
        xlabel = f"Position along {args.gene}, 5' → 3' (kb)"
    else:
        x = offsets
        xlabel = "Window offset (nt, 5' → 3')"
    order = np.argsort(x)
    xs = x[order]

    canon_x = None
    if "is_constitutive_cpa" in df.columns and df["is_constitutive_cpa"].any():
        ci = int(df.index[df["is_constitutive_cpa"]][0])
        canon_off = int(df["window_offset"].iloc[ci])
        canon_x = (canon_off + 250) / 1000.0 if gi is not None else canon_off

    print(f"\n  Per-column stats:")
    print(f"    {'column':<16} {'min':>8} {'max':>8} {'mean':>8} "
          f"{'n>=0.5':>7} {'n>=0.95':>8}")
    for c in cols:
        v = df[c].values
        n50 = int((v >= 0.5).sum())  if v.dtype.kind == "f" else 0
        n95 = int((v >= 0.95).sum()) if v.dtype.kind == "f" else 0
        print(f"    {c:<16} {v.min():>8.3f} {v.max():>8.3f} {v.mean():>8.3f} "
              f"{n50:>7,} {n95:>8,}")
        if canon_x is not None and v.dtype.kind == "f":
            rank = int((v > v[ci]).sum()) + 1
            print(f"      ^ canonical: prob={v[ci]:.4f}  rank={rank}")

    n_panels = len(cols)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 1.6 * n_panels + 1),
                              dpi=300, sharex=True, facecolor="white")
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10,
                         hspace=0.20)

    for ax, col in zip(axes, cols):
        ys = df[col].values[order]
        color = COLORS.get(col, "#444441")
        is_prob = col.endswith("_prob") or col.endswith("_prob_v1")
        if is_prob:
            ax.fill_between(xs, 0, ys, color=color, alpha=0.20, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=0.7)
            ax.set_ylim(0, 1.05)
        else:
            ax.plot(xs, ys, color=color, linewidth=0.7)
            ax.axhline(0, color="#888780", linewidth=0.4, linestyle="--")
            lo, hi = float(ys.min()), float(ys.max())
            pad = (hi - lo) * 0.05 or 0.1
            ax.set_ylim(lo - pad, hi + pad)
        if canon_x is not None:
            ax.axvline(canon_x, color="black", linewidth=0.9,
                       linestyle="--", alpha=0.55, zorder=3)

        v = df[col].values
        rank_txt = ""
        if canon_x is not None and v.dtype.kind == "f":
            rank = int((v > v[ci]).sum()) + 1
            rank_txt = f"   rank(canon)={rank}   p(canon)={v[ci]:.3f}"
        ax.text(0.005, 0.96,
                f"{col}: min={v.min():.3f}  max={v.max():.3f}"
                f"  mean={v.mean():.3f}{rank_txt}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#2C2C2A",
                family="DejaVu Sans Mono",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, boxstyle="round,pad=0.25"))
        ax.set_ylabel(col, fontsize=9, color="#2C2C2A")
        for s in ax.spines.values():
            s.set_color("black"); s.set_linewidth(0.8)
        ax.tick_params(colors="#444441", labelsize=8)
        ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")

    axes[-1].set_xlabel(xlabel, fontsize=10, color="#2C2C2A")

    title = os.path.basename(args.parquet)
    if args.gene:
        title = f"{args.gene}  ({title})"
    fig.suptitle(title, fontsize=12, color="#2C2C2A",
                 x=0.02, ha="left", y=0.98)
    if gi is not None:
        strand = "+" if gi["strand"] == "+" else "−"
        sub = (f"{gi['chrom']}:{gi['start']:,}-{gi['end']:,}  "
               f"(strand {strand}, {(gi['end']-gi['start'])/1000:.1f} kb)  "
               f"·  dashed = canonical CPA")
        fig.text(0.02, 0.955, sub, fontsize=9, color="#5F5E5A", ha="left")

    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()