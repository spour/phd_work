#!/usr/bin/env python3
# run a trained rf on a fasta file
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from featurization import read_pwms, featurize_fasta_to_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta",     required=True, help="Input FASTA (sequences >= 500 nt)")
    ap.add_argument("--model_dir", required=True,
                    help="Dir with rf_calibrated.joblib + metadata.json")
    ap.add_argument("--pwms",      required=True,
                    help="PWM list file (must match the one used at training)")
    ap.add_argument("--bg", nargs=4, type=float, default=[0.3, 0.2, 0.2, 0.3])
    ap.add_argument("--out",       required=True, help="Output TSV")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override threshold from metadata")
    ap.add_argument("--batch_size", type=int, default=0,
                    help="If >0, predict in chunks of this many rows (memory)")
    args = ap.parse_args()

    # load model + metadata
    meta_path = os.path.join(args.model_dir, "metadata.json")
    model_path = os.path.join(args.model_dir, "rf_calibrated.joblib")
    with open(meta_path) as f:
        metadata = json.load(f)

    expected_cols = metadata["winner_meta"]["columns"]
    threshold = (args.threshold
                 if args.threshold is not None
                 else metadata["winner_meta"]["val_summary"]["threshold"])
    mode = metadata.get("selected_mode", "?")
    print(f"Loaded model [{mode}]: {len(expected_cols)} features, threshold={threshold:.4f}")

    calibrator = joblib.load(model_path)

    # featurize
    print(f"\n[Featurizing {args.fasta}]")
    pwms = read_pwms(args.pwms, max_len=20, bg=np.array(args.bg, dtype=float))
    X = featurize_fasta_to_df(args.fasta, pwms, bg=np.array(args.bg, dtype=float))

    # align columns
    X = X.sort_index(axis=1)
    missing = set(expected_cols) - set(X.columns)
    extra   = set(X.columns) - set(expected_cols)
    if missing:
        raise ValueError(
            f"Featurized data is missing {len(missing)} columns expected by the model. "
            f"PWM list likely differs from training. First missing: {sorted(missing)[:5]}"
        )
    if extra:
        print(f"  dropping {len(extra)} extra features not used at training")
    X = X[expected_cols]

    # predict
    print(f"\n[Predicting on {len(X):,} rows]")
    if args.batch_size and args.batch_size < len(X):
        probs = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), args.batch_size):
            j = min(i + args.batch_size, len(X))
            probs[i:j] = calibrator.predict_proba(X.values[i:j])[:, 1]
    else:
        probs = calibrator.predict_proba(X.values)[:, 1]
    preds = (probs >= threshold).astype(int)

    out_df = pd.DataFrame({
        "seq_id":     X.index,
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