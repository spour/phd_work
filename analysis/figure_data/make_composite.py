#!/usr/bin/env python3
# three-panel figure

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cpa_pipeline import nearest, Genome, load_peaks

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

HEX = ("AATAAA", "ATTAAA")
RC = str.maketrans("ACGTN", "TGCAN")
CM = 1 / 2.54

C_FP    = "#B2182B"
C_CANON = "#2166AC"
C_RAND  = "#CFCFCF"

EXPT = [
    ("atlas_3seq", "IpA atlas\n(Singh/Lee)",  False, "#1B7837"),
    ("oh_u1amo",   "U1 AMO\n(telescripting)", True,  "#B2182B"),
    ("oh_camo",    "control AMO",             True,  "#4393C3"),
    ("pcf11_kd",   "PCF11 KD",                False, "#762A83"),
    ("pcf11_ctrl", "PCF11 siLUC",             False, "#C2A5CF"),
]

QUERY = [
    ("fp",    "model false positives", C_FP),
    ("canon", "canonical CPA",         C_CANON),
]


def load_all(outdir, peaks_path):
    q = load_peaks(peaks_path)
    sets, unstr, labels, colors = {}, {}, {}, {}

    d = q[~q.is_canonical][["chrom", "pos", "strand"]].reset_index(drop=True)
    sets["fp"] = d; unstr["fp"] = False
    labels["fp"] = "model false positives"; colors["fp"] = C_FP

    d = q[q.is_canonical][["chrom", "pos", "strand"]].reset_index(drop=True)
    sets["canon"] = d; unstr["canon"] = False
    labels["canon"] = "canonical CPA"; colors["canon"] = C_CANON

    for sid, lab, us, col in EXPT:
        p = os.path.join(outdir, "sites", sid + ".tsv")
        if not os.path.exists(p):
            print("  MISSING " + p + " -- skipping " + sid)
            continue
        d = pd.read_csv(p, sep="\t")[["chrom", "pos", "strand"]].dropna()
        d["pos"] = d["pos"].astype(int)
        d["strand"] = d["strand"].astype(str)
        sets[sid] = d.reset_index(drop=True)
        unstr[sid] = us
        labels[sid] = lab
        colors[sid] = col

    for k in sets:
        print("  " + labels[k].replace("\n", " ").ljust(26)
              + f"{len(sets[k]):>8,} sites")
    return sets, unstr, labels, colors


# distances
def all_distances(sets, unstr):
    dist = {}
    keys = list(sets)
    for a in keys:
        for b in keys:
            if a == b:
                continue
            ig = unstr[a] or unstr[b]
            dist[(a, b)] = nearest(sets[a].reset_index(drop=True), sets[b], ig)
    return dist


def displaced_reference(sets, unstr, a, b, shift=50000):
    t = sets[b].copy()
    t["pos"] = t["pos"] + shift
    return nearest(sets[a].reset_index(drop=True), t, unstr[a] or unstr[b])


# hexamer
def pct_hexamer(sites, gen, unstranded, up=60, max_n=20000, seed=0):
    if len(sites) == 0:
        return np.nan, 0
    d = (sites.sample(max_n, random_state=seed).reset_index(drop=True)
         if len(sites) > max_n else sites.reset_index(drop=True))
    hit = tot = 0
    for c, p, s in zip(d["chrom"], d["pos"], d["strand"]):
        if unstranded or s not in ("+", "-"):
            seqs = [gen.fetch(c, int(p) - up, int(p)),
                    gen.fetch(c, int(p), int(p) + up).translate(RC)[::-1]]
        elif s == "-":
            seqs = [gen.fetch(c, int(p), int(p) + up).translate(RC)[::-1]]
        else:
            seqs = [gen.fetch(c, int(p) - up, int(p))]
        if not any(len(x) >= up for x in seqs):
            continue
        tot += 1
        if any(h in x for x in seqs for h in HEX):
            hit += 1
    return (100 * hit / tot if tot else np.nan), tot


# figure
def make_figure(sets, unstr, labels, colors, dist, hexdf, figdir, tol):
    expt_ids = [e[0] for e in EXPT if e[0] in sets]
    all_ids = ["fp", "canon"] + expt_ids
    n_ecdf = len(all_ids)
    ncol = 3
    nrow = int(np.ceil(n_ecdf / ncol))

    fig = plt.figure(figsize=(18 * CM, 7.0 * CM + 4.4 * CM * nrow))
    gs = gridspec.GridSpec(2, 1, figure=fig,
                           height_ratios=[1.0, 1.05 * nrow], hspace=0.42)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0], width_ratios=[1.25, 1.0], wspace=0.38)
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        nrow, ncol, subplot_spec=gs[1], hspace=0.62, wspace=0.30)

    # a: overlap percentages
    axA = fig.add_subplot(gs_top[0, 0])
    x = np.arange(len(expt_ids))
    w = 0.38
    for j, (qid, qlab, qcol) in enumerate(QUERY):
        vals = [100 * (dist[(qid, e)] <= tol).mean() for e in expt_ids]
        pos = x + (j - 0.5) * w
        axA.bar(pos, vals, w, color=qcol, lw=0, label=qlab)
        for xi, v, e in zip(pos, vals, expt_ids):
            k = int((dist[(qid, e)] <= tol).sum())
            axA.text(xi, v * 1.06 + 0.4, f"{v:.2f}%\n{k:,}", ha="center",
                     va="bottom", fontsize=5.4, color="#333333")
    axA.set_xticks(x)
    axA.set_xticklabels([labels[e] for e in expt_ids], fontsize=6)
    axA.set_ylabel("% of query sites within " + str(tol) + " nt")
    axA.set_yscale("log")
    axA.set_ylim(0.05, 200)
    axA.legend(loc="upper left", handlelength=1.0, borderpad=0.2)
    axA.set_title("A   overlap with experimentally measured sites",
                  loc="left", pad=4)

    axC = fig.add_subplot(gs_top[0, 1])
    hx = hexdf.set_index("sid").loc[expt_ids]
    y = np.arange(len(expt_ids))[::-1]
    h = 0.34
    axC.barh(y + h / 2, hx.pct_obs, h, lw=0,
             color=[colors[e] for e in expt_ids], label="measured sites")
    axC.barh(y - h / 2, hx.pct_rand, h, color=C_RAND, lw=0,
             label="+5 kb displaced")
    for yi, (o, r) in zip(y, zip(hx.pct_obs, hx.pct_rand)):
        if np.isfinite(o):
            axC.text(max(o, r) + 1.5, yi, f"{o:.0f}%", va="center",
                     fontsize=6, color="#333333")
    axC.set_yticks(y)
    axC.set_yticklabels([labels[e] for e in expt_ids], fontsize=6)
    axC.set_xlabel("% with AATAAA or ATTAAA in 60 nt upstream")
    axC.set_xlim(0, 100)
    axC.legend(loc="lower right", handlelength=1.0, borderpad=0.2)
    axC.set_title("C   canonical PAS element at measured sites",
                  loc="left", pad=4)

    grid = np.logspace(0, 5, 400)
    for i, qid in enumerate(all_ids):
        ax = fig.add_subplot(gs_bot[i // ncol, i % ncol])
        for tid in all_ids:
            if tid == qid:
                continue
            d = dist[(qid, tid)]
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            ecdf = np.searchsorted(np.sort(d), grid, side="right") / len(d)
            ax.plot(grid, 100 * ecdf, lw=1.0, color=colors[tid],
                    label=labels[tid].replace("\n", " "))
        ref = displaced_reference(sets, unstr, qid, all_ids[1] if qid != all_ids[1]
                                  else all_ids[0])
        ref = ref[np.isfinite(ref)]
        if len(ref):
            e = np.searchsorted(np.sort(ref), grid, side="right") / len(ref)
            ax.plot(grid, 100 * e, lw=0.8, ls=(0, (3, 2)), color="#777777",
                    label="displaced (no relationship)")
        ax.axvline(tol, color="#333333", lw=0.5, ls=(0, (1, 2)))
        ax.set_xscale("log")
        ax.set_xlim(1, 1e5)
        ax.set_ylim(0, 100)
        ax.set_title("from " + labels[qid].replace("\n", " "),
                     fontsize=6.8, pad=3)
        if i % ncol == 0:
            ax.set_ylabel("% of sites within distance")
        if i // ncol == nrow - 1:
            ax.set_xlabel("distance to nearest site (nt)")
        if i == 0:
            ax.legend(loc="upper left", fontsize=5.2, handlelength=1.2,
                      labelspacing=0.25)
    fig.text(0.008, 0.985, "B   nearest-neighbour distance between site sets",
             ha="left", va="top", fontsize=8)

    fig.text(0.008, 0.006,
             "A and B match sites by nearest-neighbour distance; the Oh AMO "
             "sets are unstranded, so any comparison involving them ignores "
             "strand. Dotted vertical line in B marks " + str(tol) + " nt, the "
             "cut used in A.\nC excludes the model false positives on purpose: "
             "the CNN scored those windows highly because they carry PAS-like "
             "sequence, so their hexamer rate is a property of how the set was "
             "built, not evidence of use.\nThe +5 kb displaced bars give the "
             "local background rate for the same sequence composition.",
             ha="left", va="bottom", fontsize=5.8, color="#555555")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, "fig_composite." + ext),
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("\nwrote fig_composite.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="cpa_out")
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--figdir", default="figs")
    ap.add_argument("--tolerance", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    print("loading site sets")
    sets, unstr, labels, colors = load_all(args.outdir, args.peaks)
    if len(sets) < 3:
        sys.exit("need the peak parquet plus at least one site table")

    print("\nnearest-neighbour distances")
    dist = all_distances(sets, unstr)

    rows = []
    for a, b in dist:
        d = dist[(a, b)]
        d = d[np.isfinite(d)]
        rows.append(dict(query=labels[a].replace("\n", " "),
                         target=labels[b].replace("\n", " "),
                         n_query=len(sets[a]), n_target=len(sets[b]),
                         median_dist=float(np.median(d)) if len(d) else np.nan,
                         pct_within_50=100 * float((d <= 50).mean()) if len(d) else np.nan,
                         pct_within_500=100 * float((d <= 500).mean()) if len(d) else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "distances.tsv"),
                              sep="\t", index=False)
    print("  -> distances.tsv")

    print("\nhexamer scan")
    gen = Genome(args.genome)
    hrows = []
    for sid in sets:
        obs, n = pct_hexamer(sets[sid], gen, unstr[sid])
        d2 = sets[sid].copy()
        d2["pos"] = d2["pos"] + 5000
        rnd, _ = pct_hexamer(d2, gen, unstr[sid])
        hrows.append(dict(sid=sid, label=labels[sid].replace("\n", " "),
                          unstranded=unstr[sid], pct_obs=obs, pct_rand=rnd,
                          n_scored=n, n_total=len(sets[sid])))
        print("  " + labels[sid].replace("\n", " ").ljust(26)
              + f"{obs:5.1f}%  vs displaced {rnd:5.1f}%")
    hexdf = pd.DataFrame(hrows)
    hexdf.to_csv(os.path.join(args.outdir, "hexamer.tsv"), sep="\t", index=False)

    make_figure(sets, unstr, labels, colors, dist, hexdf, args.figdir,
                args.tolerance)


if __name__ == "__main__":
    main()