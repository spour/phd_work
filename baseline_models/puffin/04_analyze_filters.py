#!/usr/bin/env python3
# extract and visualize all motif filters from a trained puffincpa
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from puffin_cpa import (
    PuffinCPA, PuffinCPAConfig,
    PAS_MOTIF_NAMES, SPLICE_MOTIF_NAMES,
    N_PAS_PWMS, N_SPLICE_PWMS,
)

try:
    import logomaker
    HAVE_LOGOMAKER = True
except ImportError:
    HAVE_LOGOMAKER = False
    warnings.warn("logomaker not installed; using fallback bar-plot logos. "
                   "Install with: pip install logomaker")

ALPHA = "ACGT"


def kernel_to_pwm(weights, eps=1e-9):
    w = weights.copy()
    w = w - w.max(axis=0, keepdims=True)
    expw = np.exp(w)
    probs = expw / (expw.sum(axis=0, keepdims=True) + eps)
    return probs.T


def information_content(probs, eps=1e-9):
    return (probs * np.log2(probs * 4 + eps)).sum(axis=1).clip(min=0)


def consensus_sequence(probs, info_threshold=0.5):
    ic = information_content(probs)
    best_base = probs.argmax(axis=1)
    cons = []
    for p in range(probs.shape[0]):
        cons.append(ALPHA[best_base[p]] if ic[p] >= info_threshold else "N")
    return "".join(cons)


def trim_consensus(consensus):
    return consensus.strip("N")


def render_logo(probs, title, ax, color_scheme="classic"):
    L = probs.shape[0]
    if HAVE_LOGOMAKER:
        df = pd.DataFrame(probs, columns=list(ALPHA))
        ic = information_content(probs)
        df_scaled = df.multiply(ic, axis=0)
        logo = logomaker.Logo(df_scaled, ax=ax, color_scheme=color_scheme,
                                show_spines=False)
        logo.style_spines(visible=False)
        ax.set_ylim(0, 2)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["0", "1", "2"])
        ax.set_ylabel("bits", fontsize=8)
    else:
        ic = information_content(probs)
        colors = {"A": "#109648", "C": "#255C99",
                   "G": "#F7B32B", "T": "#D62839"}
        x = np.arange(L)
        for b, base in enumerate(ALPHA):
            heights = probs[:, b] * ic
            bottom = np.zeros(L)
            for bb in range(b):
                bottom += probs[:, bb] * ic
            ax.bar(x, heights, bottom=bottom, color=colors[base],
                   width=0.85, label=base if x[0] == 0 else None)
        ax.set_ylim(0, 2)
        ax.set_ylabel("bits", fontsize=8)
        if title.startswith("PAS_AATAAA"):
            ax.legend(fontsize=6, loc="upper right", ncol=4)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    return ax


def render_filter_grid(filter_data, out_path, n_cols=4, fig_per_panel=(3, 1.5)):
    n = len(filter_data)
    n_rows = (n + n_cols - 1) // n_cols
    w = n_cols * fig_per_panel[0]
    h = n_rows * fig_per_panel[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w, h),
                              dpi=120, squeeze=False)
    for i, fd in enumerate(filter_data):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        title = f"{fd['name']}  IC={fd['max_ic']:.2f}"
        if fd["is_pwm"]:
            title += " [PWM init]"
        render_logo(fd["probs"], title, ax)
        ax.set_xlabel(fd["consensus"], fontsize=8, family="monospace")
    for i in range(n, n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def effect_kernel_stats(effect_weights, filter_idx):
    w = effect_weights[0, filter_idx, :]
    k = len(w)
    center = k // 2
    sum_w     = float(w.sum())
    sum_abs_w = float(np.abs(w).sum())
    peak_idx  = int(np.argmax(np.abs(w)))
    peak_val  = float(w[peak_idx])
    peak_offset = peak_idx - center
    return {
        "sum":          sum_w,
        "sum_abs":      sum_abs_w,
        "peak_val":     peak_val,
        "peak_offset":  peak_offset,
    }


def extract_track_filters(track, motif_names, pwm_count, track_label):
    motif_w  = track.motif_conv.weight.detach().cpu().numpy()
    motif_b  = track.motif_conv.bias.detach().cpu().numpy()
    effect_w = track.effect_conv.weight.detach().cpu().numpy()

    out = []
    for i, name in enumerate(motif_names):
        kernel = motif_w[i]
        probs = kernel_to_pwm(kernel)
        ic = information_content(probs)
        max_ic = float(ic.max())
        cons_full = consensus_sequence(probs)
        cons = trim_consensus(cons_full)
        eff = effect_kernel_stats(effect_w, i)
        out.append({
            "track":           track_label,
            "name":            name,
            "is_pwm":          i < pwm_count,
            "max_ic":          max_ic,
            "total_ic":        float(ic.sum()),
            "consensus":       cons or "(low-info)",
            "consensus_full":  cons_full,
            "probs":           probs,
            "kernel":          kernel,
            "motif_bias":      float(motif_b[i]),
            "effect_sum":      eff["sum"],
            "effect_sum_abs":  eff["sum_abs"],
            "effect_peak_val": eff["peak_val"],
            "effect_peak_offset": eff["peak_offset"],
        })
    out.sort(key=lambda x: -(x["effect_sum_abs"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--top_n",      type=int, default=20,
                    help="Top N filters per track to render in the logo grid "
                         "(sorted by sum-abs effect-kernel weight). All "
                         "filters always appear in the TSV.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"[load] {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = PuffinCPAConfig(**cfg_dict)
    model = PuffinCPA(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"  epoch:           {ckpt.get('epoch', '?')}")
    print(f"  PAS PWMs:        {N_PAS_PWMS} warm-started, "
          f"{cfg.pas_n_free} free")
    print(f"  Splice PWMs:     {N_SPLICE_PWMS} warm-started, "
          f"{cfg.splice_n_free} free")
    print(f"  parameters:      {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n[extract] computing per-filter logos and statistics...")
    pas_filters = extract_track_filters(
        model.pas_track, model.pas_motif_names, N_PAS_PWMS, "pas")
    splice_filters = extract_track_filters(
        model.splice_track, model.splice_motif_names, N_SPLICE_PWMS, "splice")

    print(f"  PAS-track filters:    {len(pas_filters)}")
    print(f"  splice-track filters: {len(splice_filters)}")

    # tsv summary
    rows = []
    for f in pas_filters + splice_filters:
        rows.append({k: v for k, v in f.items()
                      if k not in ("probs", "kernel")})
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "filter_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\n[summary] -> {summary_path}")

    # top filters per track to stdout
    print(f"\n[top filters by effect-kernel |sum|]")
    print(f"\nPAS track (top {min(args.top_n, len(pas_filters))} of {len(pas_filters)}):")
    print(f"  {'name':<25s}  {'PWM?':>5s}  {'IC':>5s}  "
          f"{'eff_sumA':>8s}  {'eff_peak':>8s}  {'peak_off':>8s}  consensus")
    for f in pas_filters[:args.top_n]:
        print(f"  {f['name']:<25s}  {'yes' if f['is_pwm'] else 'no':>5s}  "
              f"{f['max_ic']:>5.2f}  {f['effect_sum_abs']:>8.3f}  "
              f"{f['effect_peak_val']:>+8.3f}  {f['effect_peak_offset']:>+8d}  "
              f"{f['consensus']}")
    print(f"\nSplice track (top {min(args.top_n, len(splice_filters))} of "
          f"{len(splice_filters)}):")
    print(f"  {'name':<25s}  {'PWM?':>5s}  {'IC':>5s}  "
          f"{'eff_sumA':>8s}  {'eff_peak':>8s}  {'peak_off':>8s}  consensus")
    for f in splice_filters[:args.top_n]:
        print(f"  {f['name']:<25s}  {'yes' if f['is_pwm'] else 'no':>5s}  "
              f"{f['max_ic']:>5.2f}  {f['effect_sum_abs']:>8.3f}  "
              f"{f['effect_peak_val']:>+8.3f}  {f['effect_peak_offset']:>+8d}  "
              f"{f['consensus']}")

    # fasta files of consensus sequences for tomtom/jaspar comparison
    pas_fasta = os.path.join(args.out_dir, "pas_filters.fasta")
    with open(pas_fasta, "w") as f:
        for fd in pas_filters:
            cons = fd["consensus"]
            if not cons or cons == "(low-info)":
                continue
            tag = "PWM_init" if fd["is_pwm"] else "free"
            f.write(f">{fd['name']}__{tag}__IC{fd['max_ic']:.2f}\n{cons}\n")
    print(f"\n[fasta] -> {pas_fasta}")
    splice_fasta = os.path.join(args.out_dir, "splice_filters.fasta")
    with open(splice_fasta, "w") as f:
        for fd in splice_filters:
            cons = fd["consensus"]
            if not cons or cons == "(low-info)":
                continue
            tag = "PWM_init" if fd["is_pwm"] else "free"
            f.write(f">{fd['name']}__{tag}__IC{fd['max_ic']:.2f}\n{cons}\n")
    print(f"        -> {splice_fasta}")

    # logo grids
    print(f"\n[logos] rendering filter logos...")
    pas_logo_path = os.path.join(args.out_dir, "pas_filter_logos.pdf")
    splice_logo_path = os.path.join(args.out_dir, "splice_filter_logos.pdf")
    render_filter_grid(pas_filters[:args.top_n], pas_logo_path)
    render_filter_grid(splice_filters[:args.top_n], splice_logo_path)

    print(f"\n[effect curves] rendering top-N filter effect kernels...")
    plot_effect_kernels(model.pas_track,    pas_filters[:8],
                         os.path.join(args.out_dir, "pas_effect_kernels.pdf"))
    plot_effect_kernels(model.splice_track, splice_filters[:8],
                         os.path.join(args.out_dir, "splice_effect_kernels.pdf"))

    meta = {
        "checkpoint":      args.checkpoint,
        "timestamp":       datetime.now().isoformat(),
        "epoch":           int(ckpt.get("epoch", -1)),
        "n_pas_filters":   len(pas_filters),
        "n_splice_filters": len(splice_filters),
        "top_pas_consensus":    [f["consensus"] for f in pas_filters[:10]],
        "top_splice_consensus": [f["consensus"] for f in splice_filters[:10]],
    }
    with open(os.path.join(args.out_dir, "filter_analysis_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] outputs in {args.out_dir}/")
    print(f"\nNext steps for biological annotation:")
    print(f"  1. Upload pas_filters.fasta and splice_filters.fasta to TomTom")
    print(f"     (https://meme-suite.org/meme/tools/tomtom)")
    print(f"     to match against JASPAR/CIS-BP databases.")
    print(f"  2. Or use STREME for de novo discovery comparison.")
    print(f"  3. Inspect the logo PDFs visually to identify known motifs:")
    print(f"     - PAS: AATAAA, ATTAAA, UGUA, T-rich, GU-rich")
    print(f"     - Splice: 5'SS (MAGGTRAGT), 3'SS branch point (YNYTRAC),")
    print(f"               polypyrimidine tract (Y_n), ESE motifs (SRSF*)")


def plot_effect_kernels(track, filters, out_path):
    effect_w = track.effect_conv.weight.detach().cpu().numpy()
    k = effect_w.shape[2]
    center = k // 2
    offsets = np.arange(k) - center
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    name_to_idx = {nm: i for i, nm in enumerate(
        track.pas_motif_names if hasattr(track, "pas_motif_names")
        else (track.splice_motif_names if hasattr(track, "splice_motif_names")
              else []))}
    motif_w = track.motif_conv.weight.detach().cpu().numpy()
    for fd in filters:
        for i in range(motif_w.shape[0]):
            if np.allclose(motif_w[i], fd["kernel"], atol=1e-6):
                ax.plot(offsets, effect_w[0, i, :],
                        label=f"{fd['name']} ({fd['consensus']})",
                        linewidth=1.2)
                break
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("offset from motif center (nt)")
    ax.set_ylabel("effect contribution")
    ax.set_title("Top filter effect-kernels")
    ax.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()