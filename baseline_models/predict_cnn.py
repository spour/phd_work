#!/usr/bin/env python3
# run a trained cpacnn on a fasta file
import argparse
import json
import os

import numpy as np
import torch
import pandas as pd
from Bio import SeqIO

from cpa_cnn import CPACNN, one_hot_sequence


def encode_fasta(fasta_path, seq_len=500):
    seqs, ids, n_skipped = [], [], 0
    for rec in SeqIO.parse(fasta_path, "fasta"):
        s = str(rec.seq)
        if len(s) != seq_len:
            n_skipped += 1
            continue
        seqs.append(one_hot_sequence(s, seq_len))
        ids.append(rec.id)
    if not seqs:
        raise ValueError(f"No sequences of length {seq_len} in {fasta_path}")
    X = torch.from_numpy(np.stack(seqs))
    print(f"  encoded {len(ids):,} sequences (skipped {n_skipped:,} with length != {seq_len})")
    return X, ids


@torch.no_grad()
def predict_batched(model, X, batch_size, device):
    model.eval()
    probs = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch_size):
        j = min(i + batch_size, len(X))
        batch = X[i:j].to(device, non_blocking=True)
        logits = model(batch)
        probs[i:j] = torch.sigmoid(logits).cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta",     required=True, help="Input FASTA (sequences must be seq_len)")
    ap.add_argument("--model_dir", required=True,
                    help="Dir with cnn_model.pt + metadata.json")
    ap.add_argument("--out",       required=True, help="Output TSV")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override threshold from metadata")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    # load metadata + model
    meta_path = os.path.join(args.model_dir, "metadata.json")
    model_path = os.path.join(args.model_dir, "cnn_model.pt")
    with open(meta_path) as f:
        metadata = json.load(f)

    arch = metadata["winner_meta"]["model_arch"].copy()
    arch.pop("n_params", None)
    threshold = (args.threshold
                 if args.threshold is not None
                 else metadata["winner_meta"]["val_summary"]["threshold"])
    mode = metadata.get("selected_mode", "?")
    print(f"Loaded model [{mode}]: arch={arch}  threshold={threshold:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPACNN(**arch).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  device: {device}  best_epoch={ckpt.get('epoch', '?')}")

    # encode fasta
    print(f"\n[Encoding {args.fasta}]")
    X, ids = encode_fasta(args.fasta, seq_len=arch["seq_len"])

    # predict
    print(f"\n[Predicting on {len(X):,} sequences]")
    probs = predict_batched(model, X, args.batch_size, device)
    preds = (probs >= threshold).astype(int)

    out_df = pd.DataFrame({
        "seq_id":     ids,
        "prob_cpa":   probs,
        "pred_label": preds,
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    n_pos = int(preds.sum())
    print(f"\nWrote {args.out}: {len(out_df):,} predictions  "
          f"({n_pos:,} positive @ threshold {threshold:.4f}, "
          f"{n_pos / len(out_df) * 100:.1f}%)")


if __name__ == "__main__":
    main()