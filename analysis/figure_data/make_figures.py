#!/usr/bin/env python3
# publication figures from cpa_out/ usage python make_figures.py --outdir cpa_out --figdir

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# house style: single-column figures, 7pt type, no chartjunk
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_PERT = "#B2182B"
C_CTRL = "#4393C3"
C_NEUT = "#999999"
C_CIRC = "#D9D9D9"
CM = 1 / 2.54


def save(fig, figdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, name + "." + ext),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  wrote " + name + ".pdf / .png")


def load(outdir, fname):
    p = os.path.join(outdir, fname)
    if not os.path.exists(p):
        print("  skip: " + fname + " not found")
        return None
    return pd.read_csv(p, sep="\t")


# figure 1
def fig_paired_fold(paired, figdir):
    if paired is None or not len(paired):
        return
    pairs = paired[["perturbation", "control"]].drop_duplicates().values
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * CM * n + 1.2 * CM, 5 * CM),
                             sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (pert, ctl) in zip(axes, pairs):
        d = paired[(paired.perturbation == pert) &
                   (paired.control == ctl)].sort_values("tolerance")
        x = np.arange(len(d))
        w = 0.38
        ax.bar(x - w / 2, d.fold_pert, w, color=C_PERT, label=pert, lw=0)
        ax.bar(x + w / 2, d.fold_ctrl, w, color=C_CTRL, label=ctl, lw=0)
        ax.axhline(1, color="k", lw=0.6, ls=(0, (3, 2)), zorder=0)

        for xi, r in zip(x, d.fold_ratio):
            if np.isfinite(r):
                top = max(d.fold_pert.iloc[xi], d.fold_ctrl.iloc[xi])
                ax.text(xi, top * 1.04, f"{r:.2f}\u00d7", ha="center",
                        va="bottom", fontsize=6, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(t)) for t in d.tolerance])
        ax.set_xlabel("tolerance (nt)")
        ax.set_title(pert + "\nvs " + ctl, pad=4)
        ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2)
        ax.margins(y=0.22)

    axes[0].set_ylabel("fold enrichment vs displaced null")
    fig.text(0.5, -0.13,
             "Ratio above each pair. Dashed line: no enrichment. The control is "
             "the same lab, assay,\npeak caller and cut, so the ratio is the "
             "perturbation-specific effect.",
             ha="center", va="top", fontsize=6, color="#555555")
    save(fig, figdir, "fig1_paired_fold")


# figure 2
def fig_discordant(quant, figdir):
    if quant is None or not len(quant):
        return
    q = quant.copy()
    q["total_disc"] = q.n_fp_pert_only + q.n_fp_ctrl_only
    q = q[q.total_disc > 0]
    if not len(q):
        return

    fig, ax = plt.subplots(figsize=(8.5 * CM, 4.6 * CM))
    y = np.arange(len(q))[::-1]
    h = 0.32
    ax.barh(y + h / 2, q.n_fp_pert_only, h, color=C_PERT,
            label="perturbation only", lw=0)
    ax.barh(y - h / 2, q.n_fp_ctrl_only, h, color=C_CTRL,
            label="control only", lw=0)

    for yi, (_, r) in zip(y, q.iterrows()):
        xmax = max(r.n_fp_pert_only, r.n_fp_ctrl_only)
        if r.total_disc < 20:
            txt = f"n={int(r.total_disc)}, underpowered"
            col = "#999999"
        else:
            txt = (f"p = {r.p_discordant:.1e}" if r.p_discordant < 1e-3
                   else f"p = {r.p_discordant:.3f}")
            col = "#333333"
        ax.text(xmax + q.n_fp_pert_only.max() * 0.02, yi, txt,
                va="center", fontsize=6, color=col)

    ax.set_yticks(y)
    ax.set_yticklabels([p + "\nvs " + c
                        for p, c in zip(q.perturbation, q.control)])
    ax.set_xlabel("false-positive sites detected in one condition only")
    ax.legend(loc="lower right", handlelength=1.0)
    ax.margins(x=0.28)
    fig.text(0.0, -0.16,
             "Only sites where detection changed contribute. Two-sided binomial "
             "against 1:1. Unlike median\nlog-ratios this does not assume RPM is "
             "comparable across conditions.",
             ha="left", va="top", fontsize=6, color="#555555")
    save(fig, figdir, "fig2_discordant")


# figure 3
def fig_overlap_fold(overlap, figdir, tol=50):
    if overlap is None or not len(overlap):
        return
    d = overlap[overlap.tolerance == tol].copy()
    d = d[d.derived_from.isna() | (d.derived_from == "")]
    if not len(d):
        return
    d = d.sort_values("fold", ascending=True)

    rate = d.pct_fp_hit.to_numpy()
    scale = np.where(rate > 0, d.fold.to_numpy() / np.where(rate > 0, rate, 1), 0)
    lo = d.ci_lo.to_numpy() * scale
    hi = d.ci_hi.to_numpy() * scale
    err = np.vstack([np.clip(d.fold - lo, 0, None), np.clip(hi - d.fold, 0, None)])

    colors = [C_CIRC if c else C_PERT for c in d.circular.astype(bool)]
    fig, ax = plt.subplots(figsize=(8.5 * CM, 0.55 * CM * len(d) + 2.2 * CM))
    y = np.arange(len(d))
    bars = ax.barh(y, d.fold, 0.62, color=colors, lw=0)
    for b, circ in zip(bars, d.circular.astype(bool)):
        if circ:
            b.set_hatch("///")
            b.set_edgecolor("#888888")
            b.set_linewidth(0.4)
    ax.errorbar(d.fold, y, xerr=err, fmt="none", ecolor="#444444",
                elinewidth=0.6, capsize=1.6, capthick=0.6)
    ax.axvline(1, color="k", lw=0.6, ls=(0, (3, 2)), zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(d.dataset)
    ax.set_xlabel("fold enrichment vs displaced null (" + str(tol) + " nt)")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r.fold + ax.get_xlim()[1] * 0.015, yi,
                f"{int(r.n_fp_hit)}/{int(r.n_fp)}", va="center", fontsize=6,
                color="#333333")
    ax.margins(x=0.20)
    fig.text(0.0, -0.10 - 0.01 * len(d),
             "Hatched: dataset used to build the APA filter, so its overlap with "
             "the filtered set is\ntautological. Bars are fold; whiskers are the "
             "Wilson interval on the hit rate, rescaled.",
             ha="left", va="top", fontsize=6, color="#555555")
    save(fig, figdir, "fig3_overlap_fold")


def fig_atlas_subsets(overlap, acct, figdir, tol=50):
    if overlap is None:
        return
    d = overlap[(overlap.tolerance == tol) &
                overlap.dataset.str.contains("__", na=False)].copy()
    if not len(d):
        return
    union = overlap[(overlap.tolerance == tol) &
                    (overlap.dataset == d.derived_from.iloc[0])]
    d["cond"] = d.dataset.str.split("__").str[-1]
    d = d.sort_values("n_fp_hit", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11 * CM, 4.6 * CM))

    if acct is not None:
        row = acct[acct.dataset == d.derived_from.iloc[0]]
        sizes = {}
        if len(row) and isinstance(row.subsets.iloc[0], str):
            for kv in row.subsets.iloc[0].split(";"):
                if "=" in kv:
                    k, v = kv.split("=")
                    sizes[k] = int(v)
        n_union = int(row.n_final.iloc[0]) if len(row) else np.nan
        conds = [c for c in d.cond if c in sizes]
        vals = [sizes[c] for c in conds]
        x = np.arange(len(conds))
        ax1.bar(x, vals, 0.62, color=C_NEUT, lw=0)
        ax1.axhline(n_union, color=C_PERT, lw=0.8, ls=(0, (3, 2)))
        ax1.text(len(conds) - 0.4, n_union, " union", color=C_PERT,
                 va="bottom", ha="right", fontsize=6)
        ax1.set_xticks(x)
        ax1.set_xticklabels(conds, rotation=30, ha="right")
        ax1.set_ylabel("atlas sites used in condition")
        ax1.set_title("subsets are 76\u201395% of the union", pad=4)

    x = np.arange(len(d))
    ax2.bar(x, d.n_fp_hit, 0.62, color=C_NEUT, lw=0)
    if len(union):
        ax2.axhline(union.n_fp_hit.iloc[0], color=C_PERT, lw=0.8,
                    ls=(0, (3, 2)))
        ax2.text(len(d) - 0.4, union.n_fp_hit.iloc[0], " union", color=C_PERT,
                 va="bottom", ha="right", fontsize=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(d.cond, rotation=30, ha="right")
    ax2.set_ylabel("FP hits at " + str(tol) + " nt")
    ax2.set_title("shared across conditions", pad=4)

    fig.text(0.0, -0.22,
             "Sites are flagged per condition at TPM > 0 in \u2265 2 samples. "
             "Raising the threshold does not separate\nthe conditions (TPM > 5 "
             "still flags 49% of CLL sites; median nonzero TPM 2.4), so these "
             "are broadly\nused sites rather than condition-specific ones.",
             ha="left", va="top", fontsize=6, color="#555555")
    save(fig, figdir, "fig4_atlas_subsets")


def fig_coverage(quant, figdir):
    if quant is None or not len(quant):
        return
    q = quant.copy()
    q["pct_inf"] = 100 * q.n_fp_informative / q.n_fp
    q = q.sort_values("pct_inf", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5 * CM, 0.7 * CM * len(q) + 2.2 * CM))
    y = np.arange(len(q))
    ax.barh(y, q.pct_inf, 0.55,
            color=[C_PERT if v >= 1 else C_NEUT for v in q.pct_inf], lw=0)
    for yi, (_, r) in zip(y, q.iterrows()):
        ax.text(r.pct_inf + 0.05, yi,
                f"{int(r.n_fp_informative):,} / {int(r.n_fp):,}",
                va="center", fontsize=6, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([p + "\nvs " + c
                        for p, c in zip(q.perturbation, q.control)])
    ax.set_xlabel("% of analysis-set false positives with coverage in "
                  "either condition")
    ax.margins(x=0.30)
    fig.text(0.0, -0.13 - 0.02 * len(q),
             "Sites with zero coverage in both conditions are ties, not "
             "evidence. A dataset covering a fraction\nof a percent has not "
             "tested the hypothesis, whatever its p-value.",
             ha="left", va="top", fontsize=6, color="#555555")
    save(fig, figdir, "fig5_coverage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="cpa_out")
    ap.add_argument("--figdir", default="figs")
    ap.add_argument("--tolerance", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    overlap = load(args.outdir, "overlap.tsv")
    paired = load(args.outdir, "overlap_paired.tsv")
    quant = load(args.outdir, "quant.tsv")
    acct = load(args.outdir, "accounting_external.tsv")

    if overlap is not None and "derived_from" not in overlap.columns:
        overlap["derived_from"] = ""

    print("figures ->", args.figdir)
    fig_paired_fold(paired, args.figdir)
    fig_discordant(quant, args.figdir)
    fig_overlap_fold(overlap, args.figdir, args.tolerance)
    fig_atlas_subsets(overlap, acct, args.figdir, args.tolerance)
    fig_coverage(quant, args.figdir)
    print("done")


if __name__ == "__main__":
    main()