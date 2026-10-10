#!/usr/bin/env python3
# train a logistic regression baseline for cpa prediction
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)

from featurization import read_pwms, featurize_fasta_to_df


def _is_fasta(path):
    return path.endswith((".fa", ".fasta", ".fa.gz", ".fasta.gz"))


def load_split(pos_path, neg_path, pwms=None, bg=None):
    def load_one(path, label):
        if _is_fasta(path):
            if pwms is None:
                raise ValueError(f"PWMs required to featurize {path}")
            df = featurize_fasta_to_df(path, pwms, bg=bg)
        else:
            df = pd.read_csv(path, sep="\t", index_col=0)
        y = np.full(len(df), label, dtype=np.int8)
        return df, y

    print(f"  pos: {pos_path}")
    Xp, yp = load_one(pos_path, 1)
    print(f"  neg: {neg_path}")
    Xn, yn = load_one(neg_path, 0)

    X = pd.concat([Xp, Xn], axis=0)
    y = np.concatenate([yp, yn])
    X = X.sort_index(axis=1)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_train", required=True)
    ap.add_argument("--neg_train", required=True)
    ap.add_argument("--pos_val",   required=True)
    ap.add_argument("--neg_val",   required=True)
    ap.add_argument("--pos_test",  required=True)
    ap.add_argument("--neg_test",  required=True)
    ap.add_argument("--pwms", default=None,
                    help="PWM list file. Required if any input is a FASTA.")
    ap.add_argument("--bg", nargs=4, type=float, default=[0.3, 0.2, 0.2, 0.3])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--penalty",      default="l1")
    ap.add_argument("--C",            type=float, default=0.1)
    ap.add_argument("--tol",          type=float, default=0.01)
    ap.add_argument("--solver",       default="saga")
    ap.add_argument("--max_iter",     type=int, default=2000)
    ap.add_argument("--class_weight", default="balanced")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--n_jobs",       type=int, default=8)
    ap.add_argument("--mode_label",   default="lr_l1_C0.1")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pwms = None
    any_fasta = any(_is_fasta(p) for p in [
        args.pos_train, args.neg_train, args.pos_val, args.neg_val,
        args.pos_test, args.neg_test,
    ])
    if any_fasta:
        if args.pwms is None:
            raise ValueError("--pwms required when any input is a FASTA")
        bg = np.array(args.bg, dtype=float)
        print(f"\n[Loading PWMs from {args.pwms}]")
        pwms = read_pwms(args.pwms, max_len=20, bg=bg)
    else:
        bg = None

    # load data
    print("\n[Loading training data]")
    X_train, y_train = load_split(args.pos_train, args.neg_train, pwms, bg)
    print(f"  train: X={X_train.shape}  pos={int(y_train.sum()):,}  "
          f"neg={int((1 - y_train).sum()):,}")

    print("\n[Loading validation data]")
    X_val, y_val = load_split(args.pos_val, args.neg_val, pwms, bg)
    print(f"  val: X={X_val.shape}  pos={int(y_val.sum()):,}  "
          f"neg={int((1 - y_val).sum()):,}")

    print("\n[Loading test data]")
    X_test, y_test = load_split(args.pos_test, args.neg_test, pwms, bg)
    print(f"  test: X={X_test.shape}  pos={int(y_test.sum()):,}  "
          f"neg={int((1 - y_test).sum()):,}")

    cols = list(X_train.columns)
    if list(X_val.columns) != cols or list(X_test.columns) != cols:
        print("  reindexing val/test columns to match train")
        X_val = X_val[cols]
        X_test = X_test[cols]

    # train
    print(f"\n[Training Logistic Regression]")
    print(f"  penalty={args.penalty}  C={args.C}  solver={args.solver}  "
          f"tol={args.tol}  class_weight={args.class_weight}  max_iter={args.max_iter}")
    t0 = time.time()
    lr = LogisticRegression(
        penalty=args.penalty,
        C=args.C,
        tol=args.tol,
        solver=args.solver,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    lr.fit(X_train.values, y_train)
    train_time = time.time() - t0
    n_iter = int(lr.n_iter_[0])
    print(f"  done in {train_time / 60:.1f} min  n_iter={n_iter}")
    if n_iter >= args.max_iter:
        print(f"  WARNING: hit max_iter={args.max_iter} -- model may not have converged. "
              f"Consider increasing --max_iter or loosening --tol.")

    coefs = lr.coef_[0]
    nonzero_mask = coefs != 0
    nonzero = int(nonzero_mask.sum())
    model_stats = {
        "n_features":       len(cols),
        "n_nonzero_coefs":  nonzero,
        "sparsity":         float(1 - nonzero / len(cols)),
        "intercept":        float(lr.intercept_[0]),
        "n_iter":           n_iter,
        "coef_l1_norm":     float(np.abs(coefs).sum()),
        "coef_max_abs":     float(np.abs(coefs).max()),
    }

    # val: threshold selection
    val_probs = lr.predict_proba(X_val.values)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val, val_probs >= t) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])

    val_summary = {
        "calibration": "none",
        "threshold":   best_threshold,
        "F1":          float(f1s[best_idx]),
        "AUPRC":       float(average_precision_score(y_val, val_probs)),
        "AUC":         float(roc_auc_score(y_val, val_probs)),
        "ACC":         float(accuracy_score(y_val, val_probs >= best_threshold)),
    }
    print(f"\n  Val: AUC={val_summary['AUC']:.4f}  AUPRC={val_summary['AUPRC']:.4f}  "
          f"F1@thr={val_summary['F1']:.4f}  thr={best_threshold:.4f}")
    print(f"  Coefficients: {nonzero}/{len(cols)} non-zero "
          f"({model_stats['sparsity'] * 100:.1f}% sparse)")

    # test
    print("\n[Evaluating on test set]")
    test_probs = lr.predict_proba(X_test.values)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, test_preds)
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    test_metrics = {
        "label":     f"TEST_{args.mode_label}",
        "AUC":       float(roc_auc_score(y_test, test_probs)),
        "AUPRC":     float(average_precision_score(y_test, test_probs)),
        "ACC":       float(accuracy_score(y_test, test_preds)),
        "F1":        float(f1_score(y_test, test_preds)),
        "CM":        cm.tolist(),
        "report":    report,
        "threshold": best_threshold,
    }
    print(f"  Test: AUC={test_metrics['AUC']:.4f}  AUPRC={test_metrics['AUPRC']:.4f}  "
          f"F1@thr={test_metrics['F1']:.4f}")

    metadata = {
        "selected_mode": args.mode_label,
        "winner_meta": {
            "columns":        cols,
            "mode":           args.mode_label,
            "selected_by":    "paper_baseline",
            "seed":           args.seed,
            "penalty":        args.penalty,
            "C":              args.C,
            "tol":            args.tol,
            "solver":         args.solver,
            "max_iter":       args.max_iter,
            "class_weight":   args.class_weight,
            "n_jobs":         args.n_jobs,
            "val_size":       "external_holdout",
            "train_time_sec": train_time,
            "model_stats":    model_stats,
            "val_summary":    val_summary,
            "test_metrics":   test_metrics,
        },
    }
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    joblib.dump(lr, os.path.join(args.outdir, "lr_model.joblib"))

    coef_df = pd.DataFrame({
        "feature":  cols,
        "coef":     coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coef_df.to_csv(os.path.join(args.outdir, "lr_coefficients.tsv"), sep="\t", index=False)

    print(f"\nWrote {args.outdir}/")
    print(f"  metadata.json")
    print(f"  lr_model.joblib")
    print(f"  lr_coefficients.tsv  ({nonzero:,} non-zero features, sorted by |coef|)")


if __name__ == "__main__":
    main()