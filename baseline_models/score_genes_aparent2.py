#!/usr/bin/env python3
# add aparent2 columns to every per-gene parquet
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from aparent.model.aparent_model_plasmid_resnet_var_batch_size_inference_mode \
    import load_aparent_model


SCAN_VERSION = "aparent2_v1"
DEFAULT_CKPT = ("/opt/aparent-resnet/saved_models/"
                 "aparent_all_libs_resnet_no_clinvar_wt_ep_5_"
                 "var_batch_size_inference_mode_no_drop.h5")
SEQ_LEN = 205
CUT_POS = 70
LIB_DIM = 13
LIB_IDX = 11


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


# one-hot

_BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def one_hot_aparent(seq, length=SEQ_LEN):
    arr = np.zeros((length, 4), dtype=np.float32)
    s = seq.upper()
    if len(s) < length:
        s = s + "N" * (length - len(s))
    for i in range(length):
        j = _BASE_IDX.get(s[i])
        if j is not None:
            arr[i, j] = 1.0
    return arr


def build_inputs(seqs, library_idx):
    n = len(seqs)
    X_seq = np.zeros((n, 1, SEQ_LEN, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        X_seq[i, 0] = one_hot_aparent(s, SEQ_LEN)
    X_lib = np.zeros((n, LIB_DIM), dtype=np.float32)
    X_lib[:, library_idx] = 1.0
    return X_seq, X_lib


# model loading

def load_aparent_via_builder(ckpt_path):
    print(f"  building architecture via load_aparent_model(...)")
    result = load_aparent_model(batch_size=1, use_sample_weights=False,
                                  drop_rate=0.0)
    model = result[0][1]
    print(f"  inputs : {[i.shape.as_list() for i in model.inputs]}")
    print(f"  outputs: {[o.shape.as_list() for o in model.outputs]}")
    print(f"  params : {model.count_params():,}")

    print(f"  loading weights from {ckpt_path}")
    model.load_weights(ckpt_path)
    print(f"  [OK] weights loaded")
    return model


def score_aparent_batch(model, seqs, library_idx, batch_size):
    X_seq, X_lib = build_inputs(seqs, library_idx)
    pred = model.predict([X_seq, X_lib], batch_size=batch_size, verbose=0)
    iso = pred[0].astype(np.float32).reshape(-1)
    cleavage = pred[1].astype(np.float32)
    return iso, cleavage


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


def safe_logit(p, eps=1e-7):
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def score_one_gene(pq_path, gene, gene_info, fa, model, library_idx,
                    cnn_window_len, cut_position, cut_radius,
                    batch_size, force, dry_run):
    t0 = time.time()

    df = pd.read_parquet(pq_path)
    n_w = len(df)

    if (not force) and ("aparent2_scan_version" in df.columns) and \
            (str(df["aparent2_scan_version"].iloc[0]) == SCAN_VERSION):
        return {"status": "skipped_already_done", "n_windows": n_w,
                "time_sec": time.time() - t0}
    if "window_offset" not in df.columns:
        return {"status": "no_window_offset", "n_windows": n_w,
                "time_sec": time.time() - t0}

    offsets = df["window_offset"].values.astype(np.int64)
    strand  = gene_info["strand"]
    chrom   = gene_info["chrom"]

    tx_centers = offsets + cnn_window_len // 2
    if strand == "+":
        starts = gene_info["start"] + tx_centers - cut_position
    else:
        starts = gene_info["end"] - tx_centers - (SEQ_LEN - cut_position)
    ends = starts + SEQ_LEN

    seqs = []
    n_bad = 0
    for s, e in zip(starts, ends):
        seq = fa.fetch(chrom, int(s), int(e), strand=strand)
        if seq is None or len(seq) != SEQ_LEN:
            seqs.append("N" * SEQ_LEN)
            n_bad += 1
        else:
            seqs.append(seq)

    CHUNK = max(batch_size * 8, 1024)
    iso_chunks      = []
    cleavage_chunks = []
    for i in range(0, n_w, CHUNK):
        iso_c, cv_c = score_aparent_batch(
            model, seqs[i:i + CHUNK],
            library_idx=library_idx, batch_size=batch_size,
        )
        iso_chunks.append(iso_c)
        cleavage_chunks.append(cv_c)
    iso = np.concatenate(iso_chunks, axis=0)
    cleavage = np.concatenate(cleavage_chunks, axis=0)

    lo = max(cut_position - cut_radius, 0)
    hi = min(cut_position + cut_radius + 1, cleavage.shape[1])
    cleavage_sum = cleavage[:, lo:hi].sum(axis=1).astype(np.float32)
    cleavage_sum = np.clip(cleavage_sum, 0.0, 1.0)

    iso = np.clip(iso, 0.0, 1.0)
    iso_logit = safe_logit(iso)

    if dry_run:
        return {"status": "dry_run", "n_windows": n_w, "n_bad_fetch": n_bad,
                "max_iso": float(iso.max()), "mean_iso": float(iso.mean()),
                "max_cleavage_sum": float(cleavage_sum.max()),
                "time_sec": time.time() - t0}

    df["aparent2_prob"]          = iso
    df["aparent2_cleavage_sum"]  = cleavage_sum
    df["aparent2_logit"]         = iso_logit
    df["aparent2_scan_version"]  = SCAN_VERSION
    atomic_write_parquet(df, pq_path)

    return {"status": "ok", "n_windows": n_w, "n_bad_fetch": n_bad,
            "max_iso": float(iso.max()), "mean_iso": float(iso.mean()),
            "max_cleavage_sum": float(cleavage_sum.max()),
            "time_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir",        required=True,
                    help="Existing CNN scan dir; per_gene/*.parquet")
    ap.add_argument("--pc_genes_bed",    required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--aparent_model",   default=DEFAULT_CKPT,
                    help="APARENT2 .h5 checkpoint (default: inference-mode "
                         "no-drop variant)")

    ap.add_argument("--cnn_window_len", type=int, default=500,
                    help="The CNN's window length (we use it to find each "
                         "window's center). Default 500.")
    ap.add_argument("--cut_position",   type=int, default=CUT_POS,
                    help=f"Position in APARENT2 input where the candidate "
                         f"cleavage site sits (default {CUT_POS}).")
    ap.add_argument("--cut_radius",     type=int, default=5,
                    help="Sum cleavage prob over [cut-r, cut+r]. "
                         "Default 5 (an 11-nt cleavage window).")
    ap.add_argument("--library_idx",    type=int, default=LIB_IDX,
                    help=f"Library one-hot index (default {LIB_IDX})")
    ap.add_argument("--batch_size",     type=int, default=256)

    ap.add_argument("--dry_run",   action="store_true")
    ap.add_argument("--force",     action="store_true")
    ap.add_argument("--limit",     type=int, default=0)
    ap.add_argument("--only_gene", nargs="+", default=None)
    ap.add_argument("--only_gene_file", default=None)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--log_json",  default=None)
    args = ap.parse_args()

    print(f"TF version: {tf.__version__}")
    print(f"GPU available: {tf.test.is_gpu_available()}")

    print(f"\n[Loading APARENT2]")
    model = load_aparent_via_builder(args.aparent_model)

    print(f"\n[Opening FASTA] {args.reference_fasta}")
    fa = GenomeFasta(args.reference_fasta)
    print(f"  engine: {fa.engine}")

    print(f"\n[Reading {args.pc_genes_bed}]")
    pc = read_pc_bed(args.pc_genes_bed)
    print(f"  {len(pc):,} unique genes")

    parquets = sorted(glob.glob(os.path.join(args.scan_dir, "per_gene",
                                              "*.parquet")))
    print(f"  {len(parquets):,} parquet files")

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
        print(f"  --only_gene/_file filter -> {len(parquets):,}")

    if args.limit:
        parquets = parquets[:args.limit]
        print(f"  --limit -> {len(parquets):,}")

    if args.dry_run:
        print("\n*** DRY RUN ***\n")

    t_start = time.time()
    statuses = {}
    summary_rows = []
    for i, pq in enumerate(parquets):
        gene = os.path.basename(pq).replace(".parquet", "")
        gi = pc.get(gene)
        if gi is None:
            statuses["no_pc_entry"] = statuses.get("no_pc_entry", 0) + 1
            continue

        res = score_one_gene(
            pq_path=pq, gene=gene, gene_info=gi, fa=fa, model=model,
            library_idx=args.library_idx,
            cnn_window_len=args.cnn_window_len,
            cut_position=args.cut_position, cut_radius=args.cut_radius,
            batch_size=args.batch_size,
            force=args.force, dry_run=args.dry_run,
        )
        statuses[res["status"]] = statuses.get(res["status"], 0) + 1
        res["gene"] = gene
        summary_rows.append(res)

        if (i + 1) % args.log_every == 0 or i < 5:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-9)
            eta  = (len(parquets) - i - 1) / max(rate, 1e-9)
            extra = ""
            if res["status"] in ("ok", "dry_run"):
                extra = (f"max_iso={res['max_iso']:.3f}  "
                         f"mean_iso={res['mean_iso']:.3f}  "
                         f"max_cv={res['max_cleavage_sum']:.3f}  "
                         f"n_w={res['n_windows']}  bad={res['n_bad_fetch']}")
            print(f"  [{i+1:>5}/{len(parquets):>5}] {gene:<20} "
                  f"{res['status']:<22} {extra}  "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f}m")

    elapsed = time.time() - t_start
    print(f"\n[Done in {elapsed/60:.1f} min]")
    print(f"  status breakdown:")
    for k, v in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"    {k:<22} {v:>6,}")

    ok_rows = [r for r in summary_rows if r["status"] in ("ok", "dry_run")]
    if ok_rows:
        print(f"\n  Sanity check across {len(ok_rows):,} genes:")
        print(f"    mean per-gene max(aparent2_prob): "
              f"{np.mean([r['max_iso']         for r in ok_rows]):.4f}")
        print(f"    mean per-gene max(cleavage_sum):  "
              f"{np.mean([r['max_cleavage_sum'] for r in ok_rows]):.4f}")

    if args.log_json:
        with open(args.log_json, "w") as f:
            json.dump({
                "model":         args.aparent_model,
                "scan_version":  SCAN_VERSION,
                "elapsed_sec":   elapsed,
                "statuses":      statuses,
                "per_gene":      summary_rows,
                "config": {
                    "cnn_window_len": args.cnn_window_len,
                    "cut_position":   args.cut_position,
                    "cut_radius":     args.cut_radius,
                    "library_idx":    args.library_idx,
                    "batch_size":     args.batch_size,
                    "seq_len":        SEQ_LEN,
                },
            }, f, indent=2)
        print(f"\n  per-gene summary written to {args.log_json}")


if __name__ == "__main__":
    main()