#!/usr/bin/env python3
# train the terminal-exon splice model
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, classification_report, confusion_matrix, f1_score,
    roc_auc_score)

from te_model import TESpliceCNN, SeqUint8Dataset

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def labels_of(ds):
    return ds.dataset.y[ds.indices] if isinstance(ds, Subset) else ds.y

@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys, ls, ps = [], [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logit = model(X)
        ls.append(logit.cpu().numpy()); ps.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ls), np.concatenate(ps)

def cosine_warmup(opt, warmup, total, min_ratio=0.05):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)

def ece(probs, labels, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        m = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not m.any(): continue
        e += (m.sum() / len(probs)) * abs(float(probs[m].mean()) - float(labels[m].mean()))
    return float(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_train", required=True); ap.add_argument("--neg_train", required=True)
    ap.add_argument("--pos_val",   required=True); ap.add_argument("--neg_val",   required=True)
    ap.add_argument("--pos_test",  required=True); ap.add_argument("--neg_test",  required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seq_len", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smooth", type=float, default=0.1)
    ap.add_argument("--pos_weight", type=float, default=None,
                    help="default = train neg/pos ratio (class-balanced)")
    ap.add_argument("--n_filters_1", type=int, default=128)
    ap.add_argument("--n_filters_2", type=int, default=256)
    ap.add_argument("--n_hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mode_label", default="te_splice")
    args = ap.parse_args()

    set_seed(args.seed); os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[Loading train]");  train_ds = SeqUint8Dataset(args.pos_train, args.neg_train, args.seq_len)
    print("\n[Loading val]");    val_ds   = SeqUint8Dataset(args.pos_val,   args.neg_val,   args.seq_len)
    print("\n[Loading test]");   test_ds  = SeqUint8Dataset(args.pos_test,  args.neg_test,  args.seq_len)

    pw = args.pos_weight if args.pos_weight is not None else train_ds.pos_weight
    print(f"\npos_weight = {pw:.2f}  label_smooth = {args.label_smooth}")

    pin = device.type == "cuda"
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                    num_workers=args.num_workers, pin_memory=pin)
    tel = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                     num_workers=args.num_workers, pin_memory=pin)

    arch = {"seq_len": args.seq_len, "n_filters_1": args.n_filters_1,
            "n_filters_2": args.n_filters_2, "n_filters_3": args.n_filters_2,
            "n_hidden": args.n_hidden, "dropout": args.dropout}
    model = TESpliceCNN(**arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(tl); warmup = max(1, int(args.warmup_frac * total))
    sched = cosine_warmup(opt, warmup, total)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = GradScaler("cuda", enabled=use_amp)

    best_auprc, best_epoch, no_imp = -1.0, -1, 0
    history = []; ckpt = os.path.join(args.outdir, "te_model.pt")
    print(f"\n[Training] epochs={args.epochs} bs={args.batch_size} lr={args.lr} amp={use_amp}")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); ep_loss = n_seen = 0
        for X, y in tl:
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            yt = y * (1 - args.label_smooth) + 0.5 * args.label_smooth if args.label_smooth > 0 else y
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss = criterion(model(X), yt)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item() * X.size(0); n_seen += X.size(0)
        tr_loss = ep_loss / n_seen
        yv, lv, pv = predict(model, vl, device)
        au = roc_auc_score(yv, pv); ap_ = average_precision_score(yv, pv)
        ec = ece(pv, yv); lp95 = float(np.percentile(np.abs(lv), 95))
        print(f"Epoch {epoch:3d} | loss={tr_loss:.4f} | val_auroc={au:.4f} | "
              f"val_auprc={ap_:.4f} | val_ece={ec:.4f} | |logit|p95={lp95:.2f} | "
              f"lr={opt.param_groups[0]['lr']:.2e}")
        history.append({"epoch": epoch, "train_loss": float(tr_loss), "val_auroc": float(au),
                        "val_auprc": float(ap_), "val_ece": ec, "logit_p95": lp95})
        if ap_ > best_auprc:
            best_auprc, best_epoch, no_imp = float(ap_), epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "arch": arch,
                        "epoch": epoch, "val_auprc": best_auprc}, ckpt)
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"\nEarly stop at epoch {epoch}"); break
    train_time = time.time() - t0
    print(f"\nDone in {train_time/60:.1f} min. Best val AUPRC {best_auprc:.4f} @ epoch {best_epoch}")

    ckpt_d = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_d["model_state_dict"])

    yv, lv, pv = predict(model, vl, device)
    ths = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(yv, pv >= t) for t in ths]; bi = int(np.argmax(f1s)); bt = float(ths[bi])
    val_summary = {"calibration": f"label_smooth_{args.label_smooth}", "threshold": bt,
                   "F1": float(f1s[bi]), "AUPRC": float(average_precision_score(yv, pv)),
                   "AUC": float(roc_auc_score(yv, pv)),
                   "ACC": float(accuracy_score(yv, pv >= bt)), "ECE": ece(pv, yv),
                   "BRIER": float(brier_score_loss(yv, pv))}
    print(f"  Val: AUC={val_summary['AUC']:.4f} AUPRC={val_summary['AUPRC']:.4f} "
          f"F1={val_summary['F1']:.4f} ECE={val_summary['ECE']:.4f} thr={bt:.3f}")

    yt, lt, pt = predict(model, tel, device)
    pred = (pt >= bt).astype(int)
    test_metrics = {"label": f"TEST_{args.mode_label}",
                    "AUC": float(roc_auc_score(yt, pt)),
                    "AUPRC": float(average_precision_score(yt, pt)),
                    "ACC": float(accuracy_score(yt, pred)), "F1": float(f1_score(yt, pred)),
                    "ECE": ece(pt, yt), "BRIER": float(brier_score_loss(yt, pt)),
                    "CM": confusion_matrix(yt, pred).tolist(),
                    "report": classification_report(yt, pred, output_dict=True, zero_division=0),
                    "threshold": bt}
    print(f"  Test: AUC={test_metrics['AUC']:.4f} AUPRC={test_metrics['AUPRC']:.4f} "
          f"F1={test_metrics['F1']:.4f} ECE={test_metrics['ECE']:.4f}")

    metadata = {"selected_mode": args.mode_label, "winner_meta": {
        "mode": args.mode_label, "selected_by": "best_val_auprc", "seed": args.seed,
        "model_arch": {**arch, "n_params": n_params},
        "training": {"epochs": args.epochs, "best_epoch": best_epoch,
                     "batch_size": args.batch_size, "lr": args.lr,
                     "weight_decay": args.weight_decay, "label_smooth": args.label_smooth,
                     "pos_weight": pw, "patience": args.patience, "grad_clip": args.grad_clip,
                     "warmup_frac": args.warmup_frac, "amp": use_amp,
                     "train_time_sec": train_time},
        "val_size": "external_holdout", "val_summary": val_summary,
        "test_metrics": test_metrics, "history": history}}
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote {args.outdir}/te_model.pt and metadata.json")

if __name__ == "__main__":
    main()
