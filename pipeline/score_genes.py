#!/usr/bin/env python3
# score every 500-nt window of every gene with rf, lr, and cnn models
import argparse
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from featurization import read_pwms, ENG_FAMILIES, one_hot, \
    correlate_pwm_over_seq, sliding_max_1d, bins_index_matrix, \
    engineered_counts_per_macro_bins
from cpa_cnn import CPACNN

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    pass


# featurization core

def featurize_long_sequence(ohe, pwms, pwm_names,
                            macro_L=500, macro_stride=1,
                            micro_L=30, micro_stride=10):
    seq_len = ohe.shape[0]
    n_macro = (seq_len - macro_L) // macro_stride + 1
    n_bins = (macro_L - micro_L) // micro_stride + 1

    if n_macro <= 0:
        return None

    # pwm features (vectorized across all macro windows)
    pwm_blocks = []
    for name in pwm_names:
        logodds = pwms[name]
        Lp = logodds.shape[1]
        S = correlate_pwm_over_seq(ohe, logodds)
        win = micro_L - Lp + 1
        if win < 1:
            raise ValueError(f"PWM {name} length {Lp} > micro window {micro_L}")
        M = sliding_max_1d(S, win)
        starts = bins_index_matrix(n_macro, n_bins, micro_stride)
        pwm_blocks.append(M[starts])
    pwm_block = (np.concatenate(pwm_blocks, axis=1)
                 if pwm_blocks else np.empty((n_macro, 0), dtype=np.float32))

    # engineered k-mer features (per-window, but vectorized inside)
    s0, s1 = ohe.strides
    macros = np.lib.stride_tricks.as_strided(
        ohe,
        shape=(n_macro, macro_L, 4),
        strides=(macro_stride * s0, s0, s1),
        writeable=False,
    )
    eng_rows = np.empty((n_macro, n_bins * len(ENG_FAMILIES)), dtype=np.float32)
    for i in range(n_macro):
        eng_rows[i] = engineered_counts_per_macro_bins(macros[i]).reshape(-1)

    X = np.concatenate([pwm_block, eng_rows], axis=1).astype(np.float32)
    return X


def feature_column_names(pwm_names, n_bins=48):
    cols = []
    for name in pwm_names:
        cols.extend([f"{name}_{i+1}" for i in range(n_bins)])
    for fam in ENG_FAMILIES:
        cols.extend([f"{fam}_{i+1}" for i in range(n_bins)])
    return cols


# cnn inference

@torch.no_grad()
def cnn_score_windows(model, ohe, macro_L, device, batch_size=512):
    seq_len = ohe.shape[0]
    n_macro = seq_len - macro_L + 1
    if n_macro <= 0:
        return np.empty(0, dtype=np.float32)

    s0, s1 = ohe.strides
    macros = np.lib.stride_tricks.as_strided(
        ohe,
        shape=(n_macro, macro_L, 4),
        strides=(s0, s0, s1),
        writeable=False,
    )

    probs = np.empty(n_macro, dtype=np.float32)
    model.eval()
    for i in range(0, n_macro, batch_size):
        j = min(i + batch_size, n_macro)
        batch = np.transpose(macros[i:j], (0, 2, 1)).astype(np.float32)
        t = torch.from_numpy(batch).to(device, non_blocking=True)
        logits = model(t)
        probs[i:j] = torch.sigmoid(logits).cpu().numpy()
    return probs


def load_rf(run_dir):
    cal = joblib.load(os.path.join(run_dir, "rf_calibrated.joblib"))
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    cols = meta["winner_meta"]["columns"]
    thr  = meta["winner_meta"]["val_summary"]["threshold"]
    return cal, cols, thr


def load_lr(run_dir):
    lr = joblib.load(os.path.join(run_dir, "lr_model.joblib"))
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    cols = meta["winner_meta"]["columns"]
    thr  = meta["winner_meta"]["val_summary"]["threshold"]
    return lr, cols, thr


def load_cnn(run_dir, device):
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    arch = meta["winner_meta"]["model_arch"].copy()
    arch.pop("n_params", None)
    thr  = meta["winner_meta"]["val_summary"]["threshold"]
    model = CPACNN(**arch).to(device)
    ckpt = torch.load(os.path.join(run_dir, "cnn_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, arch["seq_len"], thr


def read_bed(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            rows.append({
                "chrom": p[0],
                "start": int(p[1]),
                "end":   int(p[2]),
                "name":  p[3] if len(p) > 3 else f"{p[0]}:{p[1]}-{p[2]}",
                "strand": p[5] if len(p) > 5 else "+",
            })
    return rows


def run_bedtools_getfasta(bed_path, genome, out_fa, strand=True):
    cmd = ["bedtools", "getfasta", "-fi", genome, "-bed", bed_path,
           "-name", "-fo", out_fa]
    if strand:
        cmd.append("-s")
    subprocess.run(cmd, check=True)


def score_gene(seq, strand, gene_rec, cpa_genomic_pos,
               pwms, pwm_names,
               rf, rf_cols, lr, lr_cols,
               cnn, cnn_seq_len, device,
               feat_cols, cnn_batch=512):
    macro_L = 500
    seq_len = len(seq)
    if seq_len < macro_L:
        return None, {"skipped": True, "reason": f"length {seq_len} < {macro_L}"}

    ohe = one_hot(seq)
    n_macro = seq_len - macro_L + 1

    X = featurize_long_sequence(ohe, pwms, pwm_names)
    df_X = pd.DataFrame(X, columns=feat_cols)

    rf_in = df_X[rf_cols].values if rf_cols != feat_cols else X
    lr_in = df_X[lr_cols].values if lr_cols != feat_cols else X
    rf_prob = rf.predict_proba(rf_in)[:, 1].astype(np.float32)
    lr_prob = lr.predict_proba(lr_in)[:, 1].astype(np.float32)

    cnn_prob = cnn_score_windows(cnn, ohe, cnn_seq_len, device, batch_size=cnn_batch)
    if len(cnn_prob) != n_macro:
        raise RuntimeError(
            f"CNN window count {len(cnn_prob)} != feature window count {n_macro}")

    canonical_window_start = None
    if cpa_genomic_pos is not None:
        if strand == "+":
            cpa_offset = cpa_genomic_pos - gene_rec["start"]
        else:
            cpa_offset = (gene_rec["end"] - 1) - cpa_genomic_pos
        if 0 <= cpa_offset < seq_len:
            canonical_window_start = cpa_offset - 250

    is_constitutive = np.zeros(n_macro, dtype=bool)
    if (canonical_window_start is not None
            and 0 <= canonical_window_start < n_macro):
        is_constitutive[canonical_window_start] = True
        has_canonical = True
    else:
        has_canonical = False

    window_offset = np.arange(n_macro, dtype=np.int64)
    if strand == "+":
        window_start_genomic = gene_rec["start"] + window_offset
    else:
        window_start_genomic = gene_rec["end"] - window_offset - macro_L

    out = pd.DataFrame({
        "window_start_genomic": window_start_genomic.astype(np.int64),
        "window_offset":        window_offset.astype(np.int32),
        "rf_prob":              rf_prob,
        "lr_prob":              lr_prob,
        "cnn_prob":             cnn_prob,
        "is_constitutive_cpa":  is_constitutive,
    })

    stats = {
        "skipped":             False,
        "n_windows":           int(n_macro),
        "rf_top_offset":       int(np.argmax(rf_prob)),
        "lr_top_offset":       int(np.argmax(lr_prob)),
        "cnn_top_offset":      int(np.argmax(cnn_prob)),
        "rf_top_prob":         float(rf_prob.max()),
        "lr_top_prob":         float(lr_prob.max()),
        "cnn_top_prob":        float(cnn_prob.max()),
        "canonical_window_offset": (int(canonical_window_start)
                                    if has_canonical else None),
        "rf_picks_canonical":  (bool(np.argmax(rf_prob) == canonical_window_start)
                                if has_canonical else None),
        "lr_picks_canonical":  (bool(np.argmax(lr_prob) == canonical_window_start)
                                if has_canonical else None),
        "cnn_picks_canonical": (bool(np.argmax(cnn_prob) == canonical_window_start)
                                if has_canonical else None),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes_bed", required=True, help="BED of genes to scan")
    ap.add_argument("--cpa_bed",   default=None,
                    help="BED of constitutive CPA sites (positives BED used "
                         "for training). Used to flag the canonical window per "
                         "gene. The BED 'name' field must match gene names. "
                         "If positives were 500nt windows around CPA, set "
                         "--cpa_bed_is_window to use the window center as CPA.")
    ap.add_argument("--cpa_bed_is_window", action="store_true",
                    help="If set, treat each cpa_bed entry as a 500nt window "
                         "and use (start+end)//2 as the CPA position.")
    ap.add_argument("--genome",    required=True, help="hg38 FASTA")
    ap.add_argument("--pwms",      required=True)
    ap.add_argument("--bg", nargs=4, type=float, default=[0.3, 0.2, 0.2, 0.3])
    ap.add_argument("--rf_dir",  required=True)
    ap.add_argument("--lr_dir",  required=True)
    ap.add_argument("--cnn_dir", required=True)
    ap.add_argument("--outdir",  required=True)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--n_shards",    type=int, default=1)
    ap.add_argument("--cnn_batch", type=int, default=512)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "per_gene"), exist_ok=True)

    print("[Loading models]")
    pwms = read_pwms(args.pwms, max_len=20, bg=np.array(args.bg, dtype=float))
    pwm_names = list(pwms.keys())
    feat_cols = feature_column_names(pwm_names)
    print(f"  PWMs: {len(pwm_names)}  features: {len(feat_cols)}")

    rf, rf_cols, rf_thr = load_rf(args.rf_dir)
    lr, lr_cols, lr_thr = load_lr(args.lr_dir)
    print(f"  RF features: {len(rf_cols)}  threshold: {rf_thr:.3f}")
    print(f"  LR features: {len(lr_cols)}  threshold: {lr_thr:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn, cnn_seq_len, cnn_thr = load_cnn(args.cnn_dir, device)
    print(f"  CNN device: {device}  seq_len: {cnn_seq_len}  threshold: {cnn_thr:.3f}")

    if set(rf_cols) != set(feat_cols) or set(lr_cols) != set(feat_cols):
        print(f"  WARNING: feature column sets differ between RF/LR and featurizer")
        missing_rf = set(rf_cols) - set(feat_cols)
        missing_lr = set(lr_cols) - set(feat_cols)
        if missing_rf: print(f"    RF needs but not produced: {sorted(missing_rf)[:5]}")
        if missing_lr: print(f"    LR needs but not produced: {sorted(missing_lr)[:5]}")
        sys.exit(1)

    genes = read_bed(args.genes_bed)
    genes_shard = [g for i, g in enumerate(genes) if i % args.n_shards == args.shard_index]
    print(f"\n[Shard {args.shard_index}/{args.n_shards}] {len(genes_shard)} genes "
          f"(total: {len(genes)})")

    cpa_by_gene = {}
    if args.cpa_bed:
        cpa_entries = read_bed(args.cpa_bed)
        for e in cpa_entries:
            if args.cpa_bed_is_window:
                cpa_pos = (e["start"] + e["end"]) // 2
            else:
                cpa_pos = e["start"] if e["strand"] == "+" else e["end"] - 1
            cpa_by_gene.setdefault(e["name"], cpa_pos)
        matched = sum(1 for g in genes_shard if g["name"] in cpa_by_gene)
        print(f"  CPA BED: {len(cpa_entries):,} entries, "
              f"{matched}/{len(genes_shard)} genes in shard have a canonical CPA")

    shard_bed = os.path.join(args.outdir, f"_shard{args.shard_index}.bed")
    shard_fa  = os.path.join(args.outdir, f"_shard{args.shard_index}.fa")
    with open(shard_bed, "w") as f:
        for g in genes_shard:
            f.write(f"{g['chrom']}\t{g['start']}\t{g['end']}\t{g['name']}\t0\t{g['strand']}\n")
    print(f"[Extracting sequences via bedtools getfasta]")
    run_bedtools_getfasta(shard_bed, args.genome, shard_fa, strand=True)

    by_name = {g["name"]: g for g in genes_shard}

    manifest_rows = []
    t0 = time.time()
    n_done = n_skipped = 0
    for i, rec in enumerate(SeqIO.parse(shard_fa, "fasta")):
        rid = rec.id.split("::")[0]
        if rid not in by_name:
            print(f"  WARN: gene name {rid!r} from FASTA not in BED map; skipping")
            continue
        g = by_name[rid]
        out_path = os.path.join(args.outdir, "per_gene", f"{rid}.parquet")
        if args.resume and os.path.exists(out_path):
            n_done += 1
            continue

        try:
            cpa_pos = cpa_by_gene.get(rid)
            df, stats = score_gene(
                str(rec.seq).upper(), g["strand"], g, cpa_pos,
                pwms, pwm_names, rf, rf_cols, lr, lr_cols,
                cnn, cnn_seq_len, device, feat_cols,
                cnn_batch=args.cnn_batch,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_batch = max(64, args.cnn_batch // 2)
            print(f"  OOM on {rid}, retrying with cnn_batch={new_batch}")
            args.cnn_batch = new_batch
            df, stats = score_gene(
                str(rec.seq).upper(), g["strand"], g, cpa_pos,
                pwms, pwm_names, rf, rf_cols, lr, lr_cols,
                cnn, cnn_seq_len, device, feat_cols,
                cnn_batch=args.cnn_batch,
            )

        if df is None:
            n_skipped += 1
            manifest_rows.append({"gene": rid, **g, **stats})
            continue

        df.to_parquet(out_path, index=False, compression="snappy")
        manifest_rows.append({"gene": rid, **g, **stats, "parquet": out_path})
        n_done += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(genes_shard)}  "
                  f"done={n_done}  skipped={n_skipped}  "
                  f"elapsed={elapsed/60:.1f}min  "
                  f"rate={(i+1)/elapsed:.2f} genes/s")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.outdir, f"manifest_shard{args.shard_index}.parquet")
    manifest.to_parquet(manifest_path, index=False, compression="snappy")

    total_time = time.time() - t0
    print(f"\nDone. {n_done} scored, {n_skipped} skipped, {total_time/60:.1f} min total.")
    print(f"Manifest: {manifest_path}")
    print(f"Per-gene parquets in: {args.outdir}/per_gene/")

    for p in [shard_bed, shard_fa, shard_fa + ".fai"]:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()