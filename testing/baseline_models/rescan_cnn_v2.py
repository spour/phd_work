#!/usr/bin/env python3
# replace cnn_prob in every per-gene parquet with v2 scores
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from cpa_cnn import CPACNN, one_hot_sequence


SCAN_VERSION = "v2"


# fasta

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


# model

def load_v2_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    model = CPACNN(**arch).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, arch


@torch.no_grad()
def score_batch(model, seqs, seq_len, device):
    X = np.stack([one_hot_sequence(s, seq_len) for s in seqs])
    X = torch.from_numpy(X).to(device, non_blocking=True)
    logits = model(X).cpu().numpy().astype(np.float32)
    probs = np.where(logits >= 0,
                      1.0 / (1.0 + np.exp(-logits)),
                      np.exp(logits) / (1.0 + np.exp(logits))).astype(np.float32)
    return logits, probs


# helpers

def read_pc_bed(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            chrom, start, end, name = p[0], int(p[1]), int(p[2]), p[3]
            strand = p[5] if len(p) > 5 else "+"
            if name in out:
                a = out[name]
                if (end - start) <= (a["end"] - a["start"]):
                    continue
            out[name] = {"chrom": chrom, "start": start, "end": end,
                         "strand": strand}
    return out


def atomic_write_parquet(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, compression="snappy")
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# per-gene rescan

def rescan_one_gene(pq_path, gene, gene_info, fa, model, device,
                    seq_len, batch_size, save_backup, save_logits,
                    force, dry_run):
    t0 = time.time()
    df = pd.read_parquet(pq_path)
    n_w = len(df)

    if (not force) and ("cnn_scan_version" in df.columns) and \
            (str(df["cnn_scan_version"].iloc[0]) == SCAN_VERSION):
        return {"status": "skipped_already_v2", "n_windows": n_w,
                "time_sec": time.time() - t0}
    if "window_offset" not in df.columns:
        return {"status": "no_window_offset", "n_windows": n_w,
                "time_sec": time.time() - t0}
    if "cnn_prob" not in df.columns:
        return {"status": "no_cnn_prob_column", "n_windows": n_w,
                "time_sec": time.time() - t0}

    offsets = df["window_offset"].values.astype(np.int64)
    strand  = gene_info["strand"]
    chrom   = gene_info["chrom"]
    if strand == "+":
        starts = gene_info["start"] + offsets
    else:
        starts = gene_info["end"] - offsets - seq_len
    ends = starts + seq_len

    seqs = []
    n_bad = 0
    for s, e in zip(starts, ends):
        seq = fa.fetch(chrom, int(s), int(e), strand=strand)
        if seq is None or len(seq) != seq_len:
            seqs.append("N" * seq_len)
            n_bad += 1
        else:
            seqs.append(seq)

    all_logits = np.empty(n_w, dtype=np.float32)
    all_probs  = np.empty(n_w, dtype=np.float32)
    for i in range(0, n_w, batch_size):
        lg, pb = score_batch(model, seqs[i:i + batch_size], seq_len, device)
        all_logits[i:i + batch_size] = lg
        all_probs [i:i + batch_size] = pb

    max_old = float(df["cnn_prob"].max())
    max_new = float(all_probs.max())

    if dry_run:
        return {"status": "dry_run", "n_windows": n_w, "n_bad_fetch": n_bad,
                "max_prob_old": max_old, "max_prob_new": max_new,
                "time_sec": time.time() - t0}

    if save_backup and "cnn_prob_v1" not in df.columns:
        df["cnn_prob_v1"] = df["cnn_prob"].astype(np.float32)
    df["cnn_prob"] = all_probs
    if save_logits:
        df["cnn_logit"] = all_logits
    df["cnn_scan_version"] = SCAN_VERSION

    atomic_write_parquet(df, pq_path)

    return {"status": "ok", "n_windows": n_w, "n_bad_fetch": n_bad,
            "max_prob_old": max_old, "max_prob_new": max_new,
            "time_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir",        required=True,
                    help="Contains per_gene/*.parquet")
    ap.add_argument("--pc_genes_bed",    required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--v2_checkpoint",   required=True,
                    help="Path to cnnv2 cnn_model.pt")
    ap.add_argument("--seq_len",     type=int, default=500)
    ap.add_argument("--batch_size",  type=int, default=512)
    ap.add_argument("--dry_run",     action="store_true",
                    help="Score but don't rewrite parquets")
    ap.add_argument("--force",       action="store_true",
                    help="Re-score parquets even if already at "
                         f"cnn_scan_version={SCAN_VERSION}")
    ap.add_argument("--no_backup",   action="store_true",
                    help="Do not preserve old cnn_prob as cnn_prob_v1")
    ap.add_argument("--save_logits", action="store_true",
                    help="Add a cnn_logit column alongside cnn_prob "
                         "(useful for future post-hoc calibration)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N genes (0 = all). Useful with --dry_run.")
    ap.add_argument("--only_gene", nargs="+", default=None,
                    help="Only process these gene names")
    ap.add_argument("--only_gene_file", default=None,
                    help="Path to a file with one gene name per line. "
                         "Merged with --only_gene if both are given. Safer "
                         "than --only_gene when the list is very long.")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--log_json", default=None,
                    help="Path to write a per-gene JSON summary at the end")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n[Loading v2 checkpoint] {args.v2_checkpoint}")
    model, arch = load_v2_model(args.v2_checkpoint, device)
    print(f"  arch: {arch}")

    print(f"\n[Opening FASTA] {args.reference_fasta}")
    fa = GenomeFasta(args.reference_fasta)
    print(f"  engine: {fa.engine}")

    print(f"\n[Reading {args.pc_genes_bed}]")
    pc = read_pc_bed(args.pc_genes_bed)
    print(f"  {len(pc):,} unique genes")

    parquets = sorted(glob.glob(os.path.join(args.scan_dir, "per_gene",
                                              "*.parquet")))
    print(f"  {len(parquets):,} parquet files in scan_dir")

    if args.only_gene or args.only_gene_file:
        keep = set(args.only_gene or [])
        if args.only_gene_file:
            with open(args.only_gene_file) as f:
                for line in f:
                    g = line.strip()
                    if g:
                        keep.add(g)
        parquets = [p for p in parquets
                    if os.path.basename(p).replace(".parquet", "") in keep]
        print(f"  --only_gene/_file filter -> {len(parquets):,} parquets")

    if args.limit:
        parquets = parquets[:args.limit]
        print(f"  --limit -> {len(parquets):,} parquets")

    if args.dry_run:
        print("\n*** DRY RUN: parquets will not be modified ***\n")

    t_start = time.time()
    statuses = {"ok": 0, "skipped_already_v2": 0, "no_window_offset": 0,
                "no_cnn_prob_column": 0, "no_pc_entry": 0, "dry_run": 0}
    summary_rows = []
    for i, pq in enumerate(parquets):
        gene = os.path.basename(pq).replace(".parquet", "")
        gi = pc.get(gene)
        if gi is None:
            statuses["no_pc_entry"] += 1
            continue

        res = rescan_one_gene(
            pq_path=pq, gene=gene, gene_info=gi, fa=fa, model=model,
            device=device, seq_len=args.seq_len, batch_size=args.batch_size,
            save_backup=(not args.no_backup), save_logits=args.save_logits,
            force=args.force, dry_run=args.dry_run,
        )
        statuses[res["status"]] = statuses.get(res["status"], 0) + 1
        res["gene"] = gene
        summary_rows.append(res)

        if (i + 1) % args.log_every == 0 or i < 10:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (len(parquets) - i - 1) / max(rate, 1e-9)
            extra = ""
            if res["status"] in ("ok", "dry_run"):
                extra = (f"max:{res['max_prob_old']:.3f}->"
                         f"{res['max_prob_new']:.3f}  "
                         f"n_w={res['n_windows']}  "
                         f"bad={res['n_bad_fetch']}")
            print(f"  [{i+1:>5}/{len(parquets):>5}] {gene:<20} "
                  f"{res['status']:<22} {extra}  "
                  f"rate={rate:.1f}/s  eta={eta/60:.1f}m")

    elapsed = time.time() - t_start
    print(f"\n[Done in {elapsed/60:.1f} min]")
    print(f"  status breakdown:")
    for k, v in sorted(statuses.items(), key=lambda x: -x[1]):
        if v:
            print(f"    {k:<24} {v:>6,}")

    ok_rows = [r for r in summary_rows if r["status"] in ("ok", "dry_run")]
    if ok_rows:
        max_old_mean = np.mean([r["max_prob_old"] for r in ok_rows])
        max_new_mean = np.mean([r["max_prob_new"] for r in ok_rows])
        n_99_old = sum(1 for r in ok_rows if r["max_prob_old"] >= 0.99)
        n_99_new = sum(1 for r in ok_rows if r["max_prob_new"] >= 0.99)
        print(f"\n  Sanity check across {len(ok_rows):,} processed genes:")
        print(f"    mean per-gene max prob   OLD={max_old_mean:.4f}   "
              f"NEW={max_new_mean:.4f}")
        print(f"    # genes with max >= 0.99 OLD={n_99_old:,}   "
              f"NEW={n_99_new:,}")
        if max_new_mean > 0.95 and max_new_mean - max_old_mean > -0.02:
            print("    WARNING: v2 probs look very high. Check the checkpoint.")
        elif max_new_mean < 0.5:
            print("    NOTE: v2 mean max is low — expected if label smoothing "
                  "was strong (calibrated probabilities don't saturate at 1).")

    if args.log_json:
        with open(args.log_json, "w") as f:
            json.dump({
                "checkpoint":   args.v2_checkpoint,
                "scan_version": SCAN_VERSION,
                "elapsed_sec":  elapsed,
                "statuses":     statuses,
                "per_gene":     summary_rows,
            }, f, indent=2)
        print(f"\n  per-gene summary written to {args.log_json}")

    if not args.dry_run and statuses.get("ok", 0) > 0:
        print("\nNEXT STEPS:")
        print("  1) Regenerate manifest_all.parquet (cnn_top_offset is still v1)")
        print("  2) Delete stale peak caches:")
        print("     rm -f ../train/figures/_cache_peaks_*.parquet")
        print("  3) Either delete or regenerate the cryptic *.rep.bed files")
        print("     in ../train/cryptic_sites_paper/ (they're still v1)")
        print("  4) Rerun the figures sbatch with USE_REP_BEDS=0")


if __name__ == "__main__":
    main()