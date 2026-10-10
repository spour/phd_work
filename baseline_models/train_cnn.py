#!/usr/bin/env python3
# retrain cpacnn on the same train/val/test fastas as v1
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, log_loss,
    roc_auc_score,
)

from cpa_cnn import CPACNN, CPASequenceDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def labels_of(ds) -> torch.Tensor:
    if isinstance(ds, Subset):
        return ds.dataset.y[ds.indices]
    return ds.y


@torch.no_grad()
def predict_logits_and_probs(model, loader, device):
    model.eval()
    all_y, all_l, all_p = [], [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        all_l.append(logits.detach().cpu().numpy())
        all_p.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(y.numpy())
    return (np.concatenate(all_y),
            np.concatenate(all_l),
            np.concatenate(all_p))


def make_cosine_warmup_schedule(optimizer, warmup_steps, total_steps,
                                  min_lr_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def expected_calibration_error(probs, labels, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi) if i < n_bins - 1 else \
            (probs >= lo) & (probs <= hi)
        if not m.any():
            continue
        conf = float(probs[m].mean())
        acc  = float(labels[m].mean())
        ece += (m.sum() / len(probs)) * abs(conf - acc)
    return float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_train", required=True)
    ap.add_argument("--neg_train", required=True)
    ap.add_argument("--pos_val",   required=True)
    ap.add_argument("--neg_val",   required=True)
    ap.add_argument("--pos_test",  required=True)
    ap.add_argument("--neg_test",  required=True)
    ap.add_argument("--outdir",    required=True)
    ap.add_argument("--seq_len",   type=int, default=500)
    ap.add_argument("--epochs",       type=int,   default=40)
    ap.add_argument("--batch_size",   type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience",     type=int,   default=3)
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--warmup_frac",  type=float, default=0.05,
                    help="Fraction of total steps for linear LR warmup")
    ap.add_argument("--grad_clip",    type=float, default=1.0)
    ap.add_argument("--label_smooth", type=float, default=0.1,
                    help="Binary label smoothing epsilon. 0.1 means targets "
                         "{0.05, 0.95}. 0.0 disables smoothing (matches v1).")
    ap.add_argument("--pos_weight",   type=float, default=1.0,
                    help="BCEWithLogitsLoss pos_weight. v1 used class-ratio "
                         "≈ 30; with label smoothing 1.0 is the honest "
                         "default. Pass --pos_weight 30 to recover v1 weighting.")
    ap.add_argument("--n_filters_1", type=int,   default=128)
    ap.add_argument("--n_filters_2", type=int,   default=256)
    ap.add_argument("--n_hidden",    type=int,   default=128)
    ap.add_argument("--dropout",     type=float, default=0.3)
    ap.add_argument("--mode_label",  default="cnn_v2")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # data
    print("\n[Loading training data]")
    train_ds = CPASequenceDataset(args.pos_train, args.neg_train, args.seq_len)
    print("\n[Loading validation data]")
    val_ds   = CPASequenceDataset(args.pos_val,   args.neg_val,   args.seq_len)
    print("\n[Loading test data]")
    test_ds  = CPASequenceDataset(args.pos_test,  args.neg_test,  args.seq_len)

    train_labels = labels_of(train_ds)
    n_pos = int(train_labels.sum().item())
    n_neg = int(len(train_labels) - n_pos)
    class_ratio = n_neg / max(n_pos, 1)
    print(f"\nTraining class balance: {n_pos:,} pos / {n_neg:,} neg "
          f"(ratio 1:{class_ratio:.1f})")
    print(f"  v1 used pos_weight = {class_ratio:.2f}")
    print(f"  v2 using pos_weight = {args.pos_weight}  (label_smooth = "
          f"{args.label_smooth})")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)

    # model (unchanged)
    arch = {
        "seq_len":     args.seq_len,
        "n_filters_1": args.n_filters_1,
        "n_filters_2": args.n_filters_2,
        "n_hidden":    args.n_hidden,
        "dropout":     args.dropout,
    }
    model = CPACNN(**arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel trainable params: {n_params:,}")

    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler    = make_cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler  = GradScaler("cuda", enabled=use_amp)

    # train
    best_auprc, best_epoch, no_improve = -1.0, -1, 0
    history = []
    ckpt_path = os.path.join(args.outdir, "cnn_model.pt")

    print(f"\n[Training v2] epochs={args.epochs} bs={args.batch_size} "
          f"lr={args.lr} wd={args.weight_decay} amp={use_amp}")
    print(f"  cosine schedule: warmup {warmup_steps} steps / total {total_steps}")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if args.label_smooth > 0:
                y_target = y * (1.0 - args.label_smooth) + 0.5 * args.label_smooth
            else:
                y_target = y

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                logits = model(X)
                loss = criterion(logits, y_target)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                 args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item() * X.size(0)
            n_seen += X.size(0)

        train_loss = epoch_loss / n_seen

        y_v, logits_v, p_v = predict_logits_and_probs(model, val_loader, device)
        val_auroc = roc_auc_score(y_v, p_v)
        val_auprc = average_precision_score(y_v, p_v)
        val_ece   = expected_calibration_error(p_v, y_v)
        val_brier = brier_score_loss(y_v, p_v)
        logit_p95 = float(np.percentile(np.abs(logits_v), 95))
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_auroc={val_auroc:.4f} | val_auprc={val_auprc:.4f} | "
              f"val_ece={val_ece:.4f} | val_brier={val_brier:.4f} | "
              f"|logit|_p95={logit_p95:.2f} | lr={lr_now:.2e}")
        history.append({
            "epoch":      epoch,
            "train_loss": float(train_loss),
            "val_auroc":  float(val_auroc),
            "val_auprc":  float(val_auprc),
            "val_ece":    float(val_ece),
            "val_brier":  float(val_brier),
            "logit_p95":  logit_p95,
            "lr":         lr_now,
        })

        if val_auprc > best_auprc:
            best_auprc, best_epoch, no_improve = float(val_auprc), epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "arch": arch,
                "epoch": epoch,
                "val_auprc": best_auprc,
                "val_auroc": float(val_auroc),
                "val_ece":   float(val_ece),
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    train_time = time.time() - t0
    print(f"\nTraining done in {train_time/60:.1f} min. "
          f"Best val AUPRC = {best_auprc:.4f} @ epoch {best_epoch}")

    # reload best & finalize
    print("\n[Loading best model for final evaluation]")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    y_val_np, logits_val, val_probs = predict_logits_and_probs(model, val_loader, device)
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val_np, val_probs >= t) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])
    val_summary = {
        "calibration":  f"label_smooth_{args.label_smooth}",
        "threshold":    best_threshold,
        "F1":           float(f1s[best_idx]),
        "AUPRC":        float(average_precision_score(y_val_np, val_probs)),
        "AUC":          float(roc_auc_score(y_val_np, val_probs)),
        "ACC":          float(accuracy_score(y_val_np, val_probs >= best_threshold)),
        "ECE":          expected_calibration_error(val_probs, y_val_np),
        "BRIER":        float(brier_score_loss(y_val_np, val_probs)),
        "logit_p50":    float(np.percentile(np.abs(logits_val), 50)),
        "logit_p95":    float(np.percentile(np.abs(logits_val), 95)),
        "logit_p99":    float(np.percentile(np.abs(logits_val), 99)),
    }
    print(f"  Val: AUC={val_summary['AUC']:.4f} AUPRC={val_summary['AUPRC']:.4f} "
          f"F1={val_summary['F1']:.4f} ECE={val_summary['ECE']:.4f} "
          f"|logit|_p99={val_summary['logit_p99']:.2f}")

    print("\n[Evaluating on test set]")
    y_test_np, logits_test, test_probs = predict_logits_and_probs(
        model, test_loader, device)
    test_preds = (test_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test_np, test_preds)
    report = classification_report(y_test_np, test_preds,
                                    output_dict=True, zero_division=0)
    test_metrics = {
        "label":     f"TEST_{args.mode_label}",
        "AUC":       float(roc_auc_score(y_test_np, test_probs)),
        "AUPRC":     float(average_precision_score(y_test_np, test_probs)),
        "ACC":       float(accuracy_score(y_test_np, test_preds)),
        "F1":        float(f1_score(y_test_np, test_preds)),
        "ECE":       expected_calibration_error(test_probs, y_test_np),
        "BRIER":     float(brier_score_loss(y_test_np, test_probs)),
        "CM":        cm.tolist(),
        "report":    report,
        "threshold": best_threshold,
        "logit_p95": float(np.percentile(np.abs(logits_test), 95)),
        "logit_p99": float(np.percentile(np.abs(logits_test), 99)),
    }
    print(f"  Test: AUC={test_metrics['AUC']:.4f}  AUPRC={test_metrics['AUPRC']:.4f}  "
          f"F1={test_metrics['F1']:.4f}  ECE={test_metrics['ECE']:.4f}  "
          f"|logit|_p99={test_metrics['logit_p99']:.2f}")

    metadata = {
        "selected_mode": args.mode_label,
        "winner_meta": {
            "mode":        args.mode_label,
            "selected_by": "best_val_auprc",
            "seed":        args.seed,
            "model_arch":  {**arch, "n_params": n_params},
            "training": {
                "epochs":         args.epochs,
                "best_epoch":     best_epoch,
                "batch_size":     args.batch_size,
                "lr":             args.lr,
                "weight_decay":   args.weight_decay,
                "optimizer":      "AdamW",
                "scheduler":      "CosineWithWarmup",
                "warmup_frac":    args.warmup_frac,
                "criterion":      "BCEWithLogitsLoss + binary label smoothing",
                "label_smooth":   args.label_smooth,
                "pos_weight":     args.pos_weight,
                "class_ratio":    class_ratio,
                "patience":       args.patience,
                "grad_clip":      args.grad_clip,
                "amp":            use_amp,
                "train_time_sec": train_time,
            },
            "val_size":     "external_holdout",
            "val_summary":  val_summary,
            "test_metrics": test_metrics,
            "history":      history,
        },
    }
    meta_path = os.path.join(args.outdir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {args.outdir}/")
    print(f"  cnn_model.pt")
    print(f"  metadata.json")
    print(f"\nKey calibration numbers:")
    print(f"  val ECE: {val_summary['ECE']:.4f}   "
          f"(v1 was probably ≥ 0.1 given the saturation; lower is better, "
          f"target < 0.05)")
    print(f"  test |logit| p99: {test_metrics['logit_p99']:.2f}   "
          f"(v1 saw |logit|>15; target ≤ 4-5 with label_smooth=0.1)")
    print(f"  test F1: {test_metrics['F1']:.4f}   "
          f"(v1 was 0.870; expect a small drop, big calibration gain)")


if __name__ == "__main__":
    main()