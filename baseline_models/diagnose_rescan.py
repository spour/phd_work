#!/usr/bin/env python3
# find out why the parquet looks different from a fresh re-score of the same gene
import argparse
import os

import numpy as np
import pandas as pd
import torch

from cpa_cnn import CPACNN, one_hot_sequence


EPS = 1e-7


def prob_to_logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class GenomeFasta:
    def __init__(self, path):
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
        c = chrom if chrom in self.refs else chrom.lstrip("chr")
        if c not in self.refs:
            return None
        if self.engine == "pysam":
            seq = self.fa.fetch(c, start, end).upper()
        else:
            seq = str(self.fa[c][start:end]).upper()
        if strand == "-":
            seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        return seq


def read_pc_gene(pc_bed, gene):
    rows = []
    with open(pc_bed) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if p[3] == gene:
                rows.append({
                    "chrom":  p[0],
                    "start":  int(p[1]),
                    "end":    int(p[2]),
                    "strand": p[5] if len(p) > 5 else "+",
                })
    if not rows:
        return None
    rows.sort(key=lambda r: r["end"] - r["start"], reverse=True)
    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",          required=True)
    ap.add_argument("--pc_genes_bed",     required=True)
    ap.add_argument("--reference_fasta",  required=True)
    ap.add_argument("--v2_checkpoint",    required=True)
    ap.add_argument("--gene",             required=True)
    ap.add_argument("--n_random", type=int, default=10,
                    help="Number of random windows to sample (plus canonical)")
    ap.add_argument("--n_top", type=int, default=5,
                    help="Also include the top-N highest-prob windows from "
                         "the parquet for the comparison")
    ap.add_argument("--seq_len", type=int, default=500)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    print(f"=== Diagnostic for {args.gene} ===\n")
    print(f"  parquet:    {args.parquet}")
    print(f"  checkpoint: {args.v2_checkpoint}")
    print(f"  FASTA:      {args.reference_fasta}")
    print(f"  pc_bed:     {args.pc_genes_bed}\n")

    # load parquet
    df = pd.read_parquet(args.parquet)
    print(f"[Parquet]")
    print(f"  rows:    {len(df):,}")
    print(f"  columns: {list(df.columns)}")
    has_logit  = "cnn_logit"     in df.columns
    has_v1     = "cnn_prob_v1"   in df.columns
    has_marker = "cnn_scan_version" in df.columns
    print(f"  has cnn_logit:        {has_logit}")
    print(f"  has cnn_prob_v1:      {has_v1}")
    print(f"  has cnn_scan_version: {has_marker}")
    if has_marker:
        print(f"  scan_version:         {df['cnn_scan_version'].iloc[0]!r}")
    if "cnn_prob" in df.columns:
        v = df["cnn_prob"].values
        print(f"  cnn_prob (current):   min={v.min():.4f}  max={v.max():.4f}  "
              f"mean={v.mean():.4f}  #>=0.99={int((v >= 0.99).sum())}")
    if has_v1:
        v = df["cnn_prob_v1"].values
        print(f"  cnn_prob_v1 (backup): min={v.min():.4f}  max={v.max():.4f}  "
              f"mean={v.mean():.4f}  #>=0.99={int((v >= 0.99).sum())}")
    if has_logit:
        v = df["cnn_logit"].values
        print(f"  cnn_logit:            min={v.min():.2f}   max={v.max():.2f}   "
              f"p99={np.percentile(np.abs(v), 99):.2f}")

    # load gene info
    gi = read_pc_gene(args.pc_genes_bed, args.gene)
    if gi is None:
        raise SystemExit(f"ERROR: gene {args.gene} not in {args.pc_genes_bed}")
    print(f"\n[Gene]")
    print(f"  {gi['chrom']}:{gi['start']:,}-{gi['end']:,} ({gi['strand']}, "
          f"{(gi['end']-gi['start'])/1000:.1f} kb)")

    # load v2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Model] loading on {device}")
    ckpt = torch.load(args.v2_checkpoint, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    print(f"  arch in checkpoint: {arch}")
    print(f"  checkpoint best_epoch / val_auprc (if present): "
          f"{ckpt.get('epoch')}  / {ckpt.get('val_auprc')}")
    model = CPACNN(**arch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            assert not mod.training, f"BatchNorm {name} not in eval mode!"
            print(f"  BatchNorm {name}: training={mod.training}, "
                  f"running_mean.shape={tuple(mod.running_mean.shape)}, "
                  f"running_mean[0:3]={mod.running_mean[:3].cpu().numpy()}")

    # pick the windows to check
    rng = np.random.default_rng(args.seed)
    idx_to_check = set()
    if "is_constitutive_cpa" in df.columns and df["is_constitutive_cpa"].any():
        ci = int(df.index[df["is_constitutive_cpa"]][0])
        idx_to_check.add(ci)
        print(f"\n  canonical window index: {ci}  "
              f"(window_offset={int(df['window_offset'].iloc[ci])})")
    else:
        ci = None
    if args.n_top > 0 and "cnn_prob" in df.columns:
        top_idx = np.argsort(-df["cnn_prob"].values)[:args.n_top]
        for j in top_idx:
            idx_to_check.add(int(j))
    for j in rng.choice(len(df), size=min(args.n_random, len(df)), replace=False):
        idx_to_check.add(int(j))

    idx_list = sorted(idx_to_check)
    print(f"\n  Inspecting {len(idx_list)} windows "
          f"(canonical + top-{args.n_top} + {args.n_random} random)")

    fa = GenomeFasta(args.reference_fasta)
    print(f"  FASTA engine: {fa.engine}")

    offsets_check = df["window_offset"].values[idx_list].astype(int)
    if gi["strand"] == "+":
        starts = gi["start"] + offsets_check
    else:
        starts = gi["end"] - offsets_check - args.seq_len
    ends = starts + args.seq_len

    seqs = []
    for s, e in zip(starts, ends):
        seq = fa.fetch(gi["chrom"], int(s), int(e), strand=gi["strand"])
        if seq is None or len(seq) != args.seq_len:
            seq = "N" * args.seq_len
        seqs.append(seq)

    X = np.stack([one_hot_sequence(s, args.seq_len) for s in seqs])
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        live_logits = model(X_t).cpu().numpy().astype(np.float64)
    live_probs = sigmoid(live_logits)

    print(f"\n  {'idx':>6}  {'offset':>7}  {'is_can':>6}  {'seq8':>10}  "
          f"{'pq_logit':>10}  {'live_logit':>11}  {'Δlogit':>9}  "
          f"{'pq_prob':>9}  {'live_prob':>10}  {'pq_prob_v1':>11}")
    big_disagreements = 0
    for k, j in enumerate(idx_list):
        offset    = int(df["window_offset"].iloc[j])
        is_can    = bool(df.get("is_constitutive_cpa", pd.Series([False])*len(df)).iloc[j]) \
                    if "is_constitutive_cpa" in df.columns else False
        pq_logit  = float(df["cnn_logit"].iloc[j])  if has_logit else float("nan")
        pq_prob   = float(df["cnn_prob"].iloc[j])   if "cnn_prob" in df.columns else float("nan")
        pq_v1     = float(df["cnn_prob_v1"].iloc[j]) if has_v1 else float("nan")
        ll        = float(live_logits[k])
        lp        = float(live_probs[k])
        delta     = ll - pq_logit if has_logit and not np.isnan(pq_logit) else float("nan")
        if not np.isnan(delta) and abs(delta) > 0.1:
            big_disagreements += 1
        print(f"  {j:>6}  {offset:>7}  {('YES' if is_can else ''):>6}  "
              f"{seqs[k][:8]:>10}  "
              f"{pq_logit:>10.3f}  {ll:>11.3f}  {delta:>+9.3f}  "
              f"{pq_prob:>9.4f}  {lp:>10.4f}  {pq_v1:>11.4f}")

    print(f"\n  Windows with |Δlogit| > 0.1: {big_disagreements} / {len(idx_list)}")

    print("\n=== Verdict ===")
    if not has_marker:
        print("  Parquet has no cnn_scan_version marker -> rescan did not run "
              "on this gene. The 'cnn_prob' column is still v1.")
    elif big_disagreements == 0 and has_logit:
        print("  Parquet matches live re-score. The rescan was correct; the "
              "v2 model genuinely produces these logits on this gene's "
              "windows. If the spot-check looked different, it was a "
              "different gene OR a different checkpoint.")
        if has_logit:
            v = df["cnn_logit"].values
            n_extreme = int((np.abs(v) > 10).sum())
            print(f"  Note: {n_extreme:,} windows have |logit| > 10, far "
                  f"above the training-set p99 of ~4. This is the model's "
                  f"out-of-distribution behavior on gene-internal sequences.")
    else:
        print(f"  Parquet does NOT match live re-score "
              f"({big_disagreements} disagreements). The rescan wrote "
              f"wrong values OR a different checkpoint was used. Likely "
              f"causes:")
        print(f"    1. Different checkpoint path between rescan and this "
              f"diagnostic")
        print(f"    2. Off-by-one or strand handling bug in coordinate calc")
        print(f"    3. Model wasn't in eval() mode -> BatchNorm using "
              f"batch stats")


if __name__ == "__main__":
    main()