#!/usr/bin/env python3
# fit a temperature-scaling parameter t for the trained cnn (or rf/lr)
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


EPS = 1e-7


def prob_to_logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def neg_log_likelihood(T, logits, labels):
    s = logits / T
    nll = np.maximum(0, s) - s * labels + np.log1p(np.exp(-np.abs(s)))
    return float(nll.mean())


def read_chrom_set(path):
    return set(open(path).read().split())


def read_pc_bed(path, allowed_chroms=None):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            chrom, start, end, name = p[0], int(p[1]), int(p[2]), p[3]
            strand = p[5] if len(p) > 5 else "+"
            if allowed_chroms is not None and chrom not in allowed_chroms:
                continue
            out[name] = (chrom, start, end, strand)
    return out


def collect_calibration_pairs(scan_dir, val_genes, model, k_per_gene, seed):
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    parquets = [p for p in parquets
                if os.path.basename(p).replace(".parquet", "") in val_genes]
    print(f"  {len(parquets):,} val parquets to scan")

    rng = np.random.default_rng(seed)
    pcol = f"{model}_prob"
    logits, labels = [], []
    n_pos = n_neg_rand = n_neg_hard = 0
    n_genes_used = 0

    for i, pq in enumerate(parquets):
        df = pd.read_parquet(pq, columns=[pcol, "is_constitutive_cpa",
                                            "window_offset"])
        if not df["is_constitutive_cpa"].any():
            continue
        ci = int(df.index[df["is_constitutive_cpa"]][0])
        canon_off = int(df["window_offset"].iloc[ci])
        offsets = df["window_offset"].values
        probs = df[pcol].values.astype(np.float64)

        logits.append(prob_to_logit(np.array([probs[ci]]))[0])
        labels.append(1.0)
        n_pos += 1

        away = np.abs(offsets - canon_off) >= 500
        cand_probs = probs[away]
        cand_offs  = offsets[away]
        if len(cand_probs) == 0:
            continue

        k_rand = min(k_per_gene, len(cand_probs))
        rand_idx = rng.choice(len(cand_probs), size=k_rand, replace=False)
        for j in rand_idx:
            logits.append(prob_to_logit(np.array([cand_probs[j]]))[0])
            labels.append(0.0)
            n_neg_rand += 1

        k_hard = min(k_per_gene, len(cand_probs))
        hard_idx = np.argpartition(cand_probs, -k_hard)[-k_hard:]
        for j in hard_idx:
            logits.append(prob_to_logit(np.array([cand_probs[j]]))[0])
            labels.append(0.0)
            n_neg_hard += 1

        n_genes_used += 1
        if (i + 1) % 1000 == 0:
            print(f"    scanned {i+1:,}/{len(parquets):,}")

    print(f"  genes contributing: {n_genes_used:,}")
    print(f"  positives: {n_pos:,}  random negs: {n_neg_rand:,}  "
          f"hard negs: {n_neg_hard:,}")
    return np.array(logits, dtype=np.float64), np.array(labels, dtype=np.float64)


def reliability_table(p, labels, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            rows.append((lo, hi, 0, np.nan, np.nan))
        else:
            rows.append((lo, hi, n, float(p[m].mean()), float(labels[m].mean())))
    return rows


def expected_calibration_error(p, labels, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        conf = float(p[m].mean())
        acc  = float(labels[m].mean())
        ece += (m.sum() / len(p)) * abs(conf - acc)
    return float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir",     required=True,
                    help="Per-gene scan directory containing per_gene/*.parquet")
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--val_chroms",   required=True,
                    help="File listing validation chromosomes (one per line)")
    ap.add_argument("--model", default="cnn", choices=["rf", "lr", "cnn"])
    ap.add_argument("--k_per_gene", type=int, default=10,
                    help="K random + K hard negatives per gene (default 10+10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  required=True, help="Output JSON path")
    args = ap.parse_args()

    val_chroms = read_chrom_set(args.val_chroms)
    print(f"Validation chromosomes: {sorted(val_chroms)}")
    pc = read_pc_bed(args.pc_genes_bed, allowed_chroms=val_chroms)
    val_genes = set(pc.keys())
    print(f"Val genes (PC-only, on val chroms): {len(val_genes):,}")

    print(f"\n[Collecting calibration pairs for model={args.model}]")
    logits, labels = collect_calibration_pairs(
        args.scan_dir, val_genes, args.model, args.k_per_gene, args.seed)
    print(f"  total samples: {len(logits):,}")
    print(f"  logit range: [{logits.min():.2f}, {logits.max():.2f}]")
    pct_at_cap = (np.abs(logits) > 15.5).mean() * 100
    if pct_at_cap > 1:
        print(f"  WARN: {pct_at_cap:.1f}% of logits are at the recovery cap "
              f"(|logit| > 15.5). The parquets may have stored probs rounded "
              f"to ~1.0; calibration will be conservative for those windows.")

    nll_raw = neg_log_likelihood(1.0, logits, labels)
    print(f"\n[Fitting T]")
    print(f"  NLL at T=1 (uncalibrated): {nll_raw:.4f}")
    res = minimize_scalar(neg_log_likelihood, args=(logits, labels),
                          bounds=(0.5, 30.0), method="bounded",
                          options={"xatol": 1e-4})
    T = float(res.x)
    nll_cal = float(res.fun)
    print(f"  fitted T = {T:.4f}")
    print(f"  NLL at T={T:.3f}: {nll_cal:.4f}  "
          f"(reduction: {(nll_raw - nll_cal) / nll_raw * 100:.1f}%)")

    p_raw = sigmoid(logits)
    p_cal = sigmoid(logits / T)
    ece_raw = expected_calibration_error(p_raw, labels)
    ece_cal = expected_calibration_error(p_cal, labels)
    print(f"  ECE at T=1: {ece_raw:.4f}   ECE at T={T:.3f}: {ece_cal:.4f}")

    print(f"\n[Reliability diagram]")
    print(f"  {'bin':>14} {'n':>7} | {'raw_p':>7} {'raw_acc':>9}  | "
          f"{'cal_p':>7} {'cal_acc':>9}")
    rows_raw = reliability_table(p_raw, labels)
    rows_cal = reliability_table(p_cal, labels)
    for (lo, hi, n, mp, ma), (_, _, _, mp_c, ma_c) in zip(rows_raw, rows_cal):
        rp = f"{mp:.3f}" if not np.isnan(mp) else "  --  "
        ra = f"{ma:.3f}" if not np.isnan(ma) else "  --  "
        cp = f"{mp_c:.3f}" if not np.isnan(mp_c) else "  --  "
        ca = f"{ma_c:.3f}" if not np.isnan(ma_c) else "  --  "
        print(f"  [{lo:.2f}, {hi:.2f}) {n:>7,} | {rp:>7} {ra:>9}  | "
              f"{cp:>7} {ca:>9}")

    out = {
        "model":           args.model,
        "T":               T,
        "n_samples":       int(len(logits)),
        "n_positive":      int(labels.sum()),
        "raw_nll":         float(nll_raw),
        "calibrated_nll":  nll_cal,
        "raw_ece":         ece_raw,
        "calibrated_ece":  ece_cal,
        "val_chroms":      sorted(val_chroms),
        "k_per_gene":      args.k_per_gene,
        "seed":            args.seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote calibration spec -> {args.out}")
    if T <= 1.05:
        print(f"\nNote: T ≈ 1 means the model was already well-calibrated; "
              f"applying it will not change much.")
    elif T > 5:
        print(f"\nNote: T = {T:.2f} is large. The CNN is severely "
              f"over-confident; calibration will substantially compress "
              f"the high-probability mass.")


if __name__ == "__main__":
    main()