#!/usr/bin/env python3
# test the v2 cnn on a handful of genes before committing to a full-genome rescan
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from cpa_cnn import CPACNN, one_hot_sequence


# fasta accessor

class GenomeFasta:
    def __init__(self, path):
        self.path = path
        try:
            import pysam
            self.fa = pysam.FastaFile(path)
            self.engine = "pysam"
            self.refs = set(self.fa.references)
        except Exception:
            from pyfaidx import Fasta
            self.fa = Fasta(path, sequence_always_upper=True)
            self.engine = "pyfaidx"
            self.refs = set(self.fa.keys())

    def fetch(self, chrom, start, end, strand="+"):
        chrom_use = chrom if chrom in self.refs else chrom.lstrip("chr")
        if chrom_use not in self.refs:
            return None
        if self.engine == "pysam":
            seq = self.fa.fetch(chrom_use, start, end).upper()
        else:
            seq = str(self.fa[chrom_use][start:end]).upper()
        if strand == "-":
            comp = str.maketrans("ACGTN", "TGCAN")
            seq = seq.translate(comp)[::-1]
        return seq


# model loading

def load_v2_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    model = CPACNN(**arch).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, arch


def batched_score(model, sequences, seq_len, device, batch_size=512):
    model.eval()
    probs = np.empty(len(sequences), dtype=np.float32)
    logits = np.empty(len(sequences), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            X = np.stack([one_hot_sequence(s, seq_len) for s in batch])
            X = torch.from_numpy(X).to(device, non_blocking=True)
            lg = model(X).cpu().numpy()
            logits[i:i + batch_size] = lg
            probs [i:i + batch_size] = 1.0 / (1.0 + np.exp(-lg))
    return logits, probs


# helpers

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


def to_transcript_kb(genomic_mid, gi):
    if gi["strand"] == "+":
        return (genomic_mid - gi["start"]) / 1000.0
    return (gi["end"] - genomic_mid) / 1000.0


def atlas_kb_in_gene(atlas_beds, gi):
    out = []
    if not atlas_beds:
        return out
    for bp in atlas_beds:
        if not os.path.exists(bp):
            continue
        with open(bp) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if p[0] != gi["chrom"]:
                    continue
                if len(p) > 5 and p[5] != gi["strand"]:
                    continue
                mid = (int(p[1]) + int(p[2])) // 2
                if gi["start"] <= mid <= gi["end"]:
                    out.append(to_transcript_kb(mid, gi))
    return out


# figure

MODEL_COLORS = {"RF": "#7F77DD", "LR": "#BA7517",
                "CNN v1": "#1D9E75", "CNN v2": "#3C3489"}


def plot_gene_comparison(gene, gi, df, v2_logits, v2_probs,
                          atlas_kbs, out_path):
    offsets = df["window_offset"].values
    x_kb    = (offsets + 250) / 1000.0
    order   = np.argsort(x_kb)
    xs      = x_kb[order]

    canon_kb = None
    if df["is_constitutive_cpa"].any():
        ci = int(df.index[df["is_constitutive_cpa"]][0])
        canon_off = int(df["window_offset"].iloc[ci])
        canon_kb  = (canon_off + 250) / 1000.0

    panels = [
        ("RF",     df["rf_prob"].values,  MODEL_COLORS["RF"]),
        ("LR",     df["lr_prob"].values,  MODEL_COLORS["LR"]),
        ("CNN v1", df["cnn_prob"].values, MODEL_COLORS["CNN v1"]),
        ("CNN v2", v2_probs,              MODEL_COLORS["CNN v2"]),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), dpi=300,
                              sharex=True, facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.07,
                        hspace=0.18)

    for ax, (name, probs, color) in zip(axes, panels):
        ys = probs[order]
        ax.fill_between(xs, 0, ys, color=color, alpha=0.20, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=0.7)
        if canon_kb is not None:
            ax.axvline(canon_kb, color="black", linewidth=0.9,
                       linestyle="--", alpha=0.55, zorder=3)
        for ak in atlas_kbs:
            ax.plot([ak], [1.02], marker="v", color="#888780",
                    markersize=4, clip_on=False)

        n_high = int((probs >= 0.5).sum())
        p_max  = float(probs.max())
        p_canon = float(probs[ci]) if canon_kb is not None else float("nan")
        rank   = int((probs > probs[ci]).sum()) + 1 if canon_kb is not None else -1
        ax.text(0.005, 0.96,
                f"max={p_max:.3f}    "
                f"p(canon)={p_canon:.3f}    "
                f"rank(canon)={rank}    "
                f"#≥0.5={n_high}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#2C2C2A",
                family="DejaVu Sans Mono",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          boxstyle="round,pad=0.25"))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{name}\nprob", fontsize=10, color="#2C2C2A")
        for s in ax.spines.values():
            s.set_color("black"); s.set_linewidth(0.8)
        ax.tick_params(colors="#444441", labelsize=8)
        ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")

    axes[-1].set_xlabel(f"Position along {gene}, 5' → 3' (kb)",
                        fontsize=10, color="#2C2C2A")

    strand_label = "+" if gi["strand"] == "+" else "−"
    subtitle_bits = [f"{gi['chrom']}:{gi['start']:,}-{gi['end']:,} "
                     f"(strand {strand_label}, "
                     f"{(gi['end']-gi['start'])/1000:.1f} kb)",
                     "dashed line = canonical CPA"]
    if atlas_kbs:
        subtitle_bits.append(f"▼ atlas ({len(atlas_kbs)} sites)")
    fig.suptitle(f"v1 vs v2 CNN on {gene}",
                 fontsize=13, color="#2C2C2A", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.945, "  ·  ".join(subtitle_bits),
             fontsize=9, color="#5F5E5A", ha="left")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir",          required=True,
                    help="Per-gene scan dir (contains per_gene/*.parquet "
                         "with v1 cnn_prob)")
    ap.add_argument("--pc_genes_bed",      required=True)
    ap.add_argument("--reference_fasta",   required=True,
                    help="Path to hg38.fa (or whichever reference the "
                         "model was trained on)")
    ap.add_argument("--v2_checkpoint",     required=True,
                    help="Path to v2 cnn_model.pt")
    ap.add_argument("--gene", nargs="+",   required=True,
                    help="One or more gene names to test, e.g. "
                         "--gene FGF20 ERP27")
    ap.add_argument("--atlas_bed", nargs="+", default=None,
                    help="Optional atlas BED(s) for tick marks")
    ap.add_argument("--out_dir",           required=True)
    ap.add_argument("--seq_len", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n[Loading v2 model] {args.v2_checkpoint}")
    model, arch = load_v2_model(args.v2_checkpoint, device)
    print(f"  arch: {arch}")

    print(f"\n[Opening reference FASTA] {args.reference_fasta}")
    fa = GenomeFasta(args.reference_fasta)
    print(f"  engine: {fa.engine}")

    for gene in args.gene:
        print(f"\n=== {gene} ===")
        gi = read_pc_gene(args.pc_genes_bed, gene)
        if gi is None:
            print(f"  WARN: gene not in {args.pc_genes_bed}; skipping")
            continue
        print(f"  {gi['chrom']}:{gi['start']:,}-{gi['end']:,} "
              f"({gi['strand']}, {(gi['end']-gi['start'])/1000:.1f} kb)")

        pq_path = os.path.join(args.scan_dir, "per_gene", f"{gene}.parquet")
        if not os.path.exists(pq_path):
            print(f"  WARN: no parquet at {pq_path}; skipping")
            continue
        df = pd.read_parquet(pq_path)
        n_w = len(df)
        print(f"  {n_w:,} windows in parquet")

        offsets = df["window_offset"].values.astype(int)
        if gi["strand"] == "+":
            starts = gi["start"] + offsets
        else:
            starts = gi["end"] - offsets - args.seq_len
        ends = starts + args.seq_len
        print(f"  fetching {n_w:,} window sequences from FASTA...")
        seqs = []
        bad = 0
        for s, e in zip(starts, ends):
            seq = fa.fetch(gi["chrom"], int(s), int(e), strand=gi["strand"])
            if seq is None or len(seq) != args.seq_len:
                bad += 1
                seq = "N" * args.seq_len
            seqs.append(seq)
        if bad:
            print(f"  WARN: {bad} windows had FASTA fetch issues (filled with Ns)")

        print(f"  scoring with v2 model...")
        v2_logits, v2_probs = batched_score(
            model, seqs, args.seq_len, device, batch_size=args.batch_size)

        v1_probs = df["cnn_prob"].values
        canon_idx = df.index[df["is_constitutive_cpa"]] if df["is_constitutive_cpa"].any() else None
        ci = int(canon_idx[0]) if canon_idx is not None and len(canon_idx) else None

        def summarize(name, p):
            n_99 = int((p >= 0.99).sum())
            n_95 = int((p >= 0.95).sum())
            n_50 = int((p >= 0.50).sum())
            pmin, pmax = float(p.min()), float(p.max())
            line = (f"    {name:>7}: min/max={pmin:.4f}/{pmax:.4f}  "
                    f"n>=0.99={n_99:<5}  n>=0.95={n_95:<5}  n>=0.5={n_50}")
            if ci is not None:
                rank = int((p > p[ci]).sum()) + 1
                line += f"  rank(canon)={rank}  p(canon)={p[ci]:.4f}"
            print(line)

        print(f"\n  Summary for {gene}:")
        summarize("v1 CNN", v1_probs)
        summarize("v2 CNN", v2_probs)

        atlas_kbs = atlas_kb_in_gene(args.atlas_bed, gi) if args.atlas_bed else []
        out_path = os.path.join(args.out_dir, f"v1_vs_v2_{gene}.png")
        plot_gene_comparison(gene, gi, df, v2_logits, v2_probs,
                              atlas_kbs, out_path)

    print(f"\nAll comparisons written to {args.out_dir}/")


if __name__ == "__main__":
    main()