#!/usr/bin/env python3
# aparent2 gradient saliency at cpa sites
import argparse
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from keras import backend as K

from aparent.model.aparent_model_plasmid_resnet_var_batch_size_inference_mode \
    import load_aparent_model


SEQ_LEN = 205
CUT_POS = 70
LIB_DIM = 13
LIB_IDX = 11
BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def read_bed(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            p = line.rstrip("\n").split("\t")
            rows.append({"chrom": p[0], "start": int(p[1]), "end": int(p[2]),
                         "name": p[3] if len(p) > 3 else "",
                         "strand": p[5] if len(p) > 5 else "+"})
    return rows


def one_hot(seq):
    arr = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    s = seq.upper()
    for i in range(min(SEQ_LEN, len(s))):
        j = BASES.get(s[i])
        if j is not None:
            arr[i, j] = 1.0
    return arr[None, None, :, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_bed", required=True,
                    help="BED of sites to compute saliency around (e.g. "
                         "canonical or cryptic CPAs). The site MIDPOINT is "
                         "placed at APARENT2's cut position (70). Gene name "
                         "is taken from column 4, splitting on '::' if present.")
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--aparent_model",   required=True,
                    help="Path INSIDE container to the h5 checkpoint")
    ap.add_argument("--genes",  required=True, nargs="+",
                    help="Gene names to compute saliency for. For cryptic "
                         "sites where a gene may have multiple cryptic CPAs, "
                         "ALL sites for that gene are computed.")
    ap.add_argument("--window_radius", type=int, default=60,
                    help="Output saliency at +/- this many nt around CUT_POS "
                         "(default 60; APARENT2's input is 205 with cut at 70 "
                         "so the usable range is -70..+134)")
    ap.add_argument("--site_label", default="canonical",
                    help="String tag stored in the output parquet 'site_class' "
                         "column (default 'canonical'). Use 'cryptic' for "
                         "cryptic-site runs so a single downstream figure can "
                         "tell them apart.")
    ap.add_argument("--max_sites_per_gene", type=int, default=0,
                    help="If a gene has multiple sites in the BED, limit to "
                         "this many (top-scoring by BED column 5, or first "
                         "encountered if scores tied). 0 = no limit. Use this "
                         "for cryptic runs where a single gene may have dozens "
                         "of cryptic CPAs to avoid runaway compute.")
    ap.add_argument("--out", required=True,
                    help="Output parquet path")
    args = ap.parse_args()

    try:
        import pysam
        fa = pysam.FastaFile(args.reference_fasta)
        fa_refs = set(fa.references)
    except Exception as e:
        sys.exit(f"failed to open FASTA: {e}")

    def fetch(chrom, s, e, strand):
        c = chrom if chrom in fa_refs else chrom.lstrip("chr")
        if c not in fa_refs: return None
        seq = fa.fetch(c, s, e).upper()
        if strand == "-":
            seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        return seq

    sites_df = pd.DataFrame(read_bed(args.sites_bed))
    if "name" in sites_df.columns:
        sites_df["gene"] = sites_df["name"].str.split("::").str[0]
    else:
        sites_df["gene"] = ""
    sites_df = sites_df[sites_df["gene"].isin(args.genes)].copy()
    print(f"sites for requested genes: {len(sites_df):,}")
    if args.max_sites_per_gene > 0:
        try:
            sites_df = (sites_df.groupby("gene", group_keys=False)
                                  .head(args.max_sites_per_gene))
            print(f"  after --max_sites_per_gene={args.max_sites_per_gene}: "
                  f"{len(sites_df):,}")
        except Exception as e:
            print(f"  WARN: failed to limit sites per gene: {e}")
    if len(sites_df) == 0:
        sys.exit("no sites for the requested genes; nothing to do")

    print(f"building APARENT2...")
    result = load_aparent_model(batch_size=1, use_sample_weights=False,
                                  drop_rate=0.0)
    model = result[0][1]
    model.load_weights(args.aparent_model)
    print(f"  inputs : {[i.shape.as_list() for i in model.inputs]}")
    print(f"  outputs: {[o.shape.as_list() for o in model.outputs]}")

    seq_input = model.inputs[0]
    iso_output = model.outputs[0]
    grad_op = K.gradients(K.sum(iso_output), seq_input)[0]
    sess = K.get_session()

    out_rows = []
    n_done = 0
    n_skipped = 0
    for _, site in sites_df.iterrows():
        gene = site["gene"]
        mid = (site["start"] + site["end"]) // 2
        strand = site["strand"]
        chrom = site["chrom"]
        site_id = f"{gene}:{chrom}:{mid}:{strand}"

        if strand == "+":
            w_start = mid - CUT_POS
        else:
            w_start = mid - (SEQ_LEN - CUT_POS - 1)
        w_end = w_start + SEQ_LEN
        seq = fetch(chrom, w_start, w_end, strand)
        if seq is None or len(seq) != SEQ_LEN:
            n_skipped += 1
            continue

        oh = one_hot(seq)
        lib = np.zeros((1, LIB_DIM), dtype=np.float32)
        lib[0, LIB_IDX] = 1.0

        grad = sess.run(grad_op, feed_dict={
            seq_input: oh,
            model.inputs[1]: lib,
            K.learning_phase(): 0,
        })
        attr_per_pos = np.abs((grad * oh).sum(axis=-1)).reshape(-1)

        lo = max(0, CUT_POS - args.window_radius)
        hi = min(SEQ_LEN, CUT_POS + args.window_radius + 1)
        for pos in range(lo, hi):
            out_rows.append({
                "gene":         gene,
                "site_id":      site_id,
                "site_class":   args.site_label,
                "position_rel": int(pos - CUT_POS),
                "saliency":     float(attr_per_pos[pos]),
            })

        peak_idx = int(np.argmax(attr_per_pos[lo:hi])) + lo
        n_done += 1
        if n_done <= 20 or n_done % 50 == 0:
            print(f"  [{n_done:>4}] {site_id}  max saliency at position "
                  f"{peak_idx - CUT_POS:+d} (value {attr_per_pos[peak_idx]:.4f})")

    print(f"\nfinished {n_done:,} sites ({n_skipped:,} skipped for bad fetch)")

    if not out_rows:
        sys.exit("no genes produced saliency; nothing to write")

    df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                 exist_ok=True)
    df.to_parquet(args.out, index=False, compression="snappy")
    print(f"\nwrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()