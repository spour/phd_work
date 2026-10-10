#!/usr/bin/env python3
# train puffincpa on labeled sites with on-the-fly sequence fetching from hg38.fa
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pysam
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              f1_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from puffin_cpa import (
    PuffinCPA, PuffinCPAConfig, encode_sequence,
    UP_FLANK, DOWN_FLANK, SEQ_LEN,
)


class SiteSequenceDataset(Dataset):

    def __init__(self, bed_path, tsv_path, genome_fa,
                 oversample_classes=None, seed=42, center_jitter=0):
        self.genome_fa = genome_fa
        self._fa = None
        self.center_jitter = center_jitter
        self._jitter_rng = np.random.default_rng(seed)
        rows = []
        with open(bed_path) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                rows.append({
                    "chrom":  p[0],
                    "start":  int(p[1]),
                    "end":    int(p[2]),
                    "name":   p[3],
                    "strand": p[5],
                })
        sites = pd.DataFrame(rows)
        meta = pd.read_csv(tsv_path, sep="\t")
        if len(meta) != len(sites):
            raise ValueError(
                f"BED and TSV row counts differ: "
                f"BED={len(sites)} TSV={len(meta)}. Files: {bed_path}, {tsv_path}")
        for col in ("site_class", "gene", "source_model"):
            if col in meta.columns:
                sites[col] = meta[col].values
        sites["label"] = (sites["site_class"] == "constitutive").astype(int)
        sites["center"] = (sites["start"] + sites["end"]) // 2

        if oversample_classes:
            rng = np.random.default_rng(seed)
            target_n = max(int(v) for v in oversample_classes.values())
            parts = []
            for cls, _ in oversample_classes.items():
                sub = sites[sites["site_class"] == cls]
                if len(sub) == 0:
                    continue
                if len(sub) < target_n:
                    extra_idx = rng.integers(0, len(sub), size=target_n - len(sub))
                    sub = pd.concat([sub, sub.iloc[extra_idx]], ignore_index=True)
                parts.append(sub)
            sites = pd.concat(parts, ignore_index=True)
            sites = sites.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        self.sites = sites
        print(f"  loaded {len(sites):,} sites from {bed_path}")
        print(f"    class breakdown: {dict(sites['site_class'].value_counts())}")
        print(f"    label balance: positive={int(sites['label'].sum()):,} / "
              f"negative={int((1 - sites['label']).sum()):,}")

    def _ensure_fa(self):
        if self._fa is None:
            self._fa = pysam.FastaFile(self.genome_fa)

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, idx):
        self._ensure_fa()
        row = self.sites.iloc[idx]
        chrom, center, strand = row["chrom"], int(row["center"]), row["strand"]
        if self.center_jitter > 0:
            shift = int(self._jitter_rng.integers(
                -self.center_jitter, self.center_jitter + 1))
            center = center + shift
        if strand == "+":
            s = max(0, center - UP_FLANK)
            e = center + DOWN_FLANK
            seq = self._fa.fetch(chrom, s, e).upper()
            if len(seq) < SEQ_LEN:
                seq = "N" * (SEQ_LEN - len(seq)) + seq
        else:
            s = max(0, center - DOWN_FLANK)
            e = center + UP_FLANK
            seq = self._fa.fetch(chrom, s, e).upper()
            tab = str.maketrans("ACGTNacgtn", "TGCANtgcan")
            seq = seq.translate(tab)[::-1]
            if len(seq) < SEQ_LEN:
                seq = seq + "N" * (SEQ_LEN - len(seq))
        seq = seq[:SEQ_LEN]
        x = encode_sequence(seq, pad_len=SEQ_LEN)
        return torch.from_numpy(x), float(row["label"]), str(row["site_class"])


def collate(batch):
    xs       = torch.stack([b[0] for b in batch], dim=0)
    ys       = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    classes  = [b[2] for b in batch]
    return xs, ys, classes


def evaluate(model, loader, device, pos_weight=None):
    model.eval()
    all_probs    = []
    all_labels   = []
    all_logits   = []
    all_a_pas    = []
    all_a_splice = []
    all_pas_summary    = []
    all_splice_summary = []
    all_classes  = []
    bce_sum = 0.0
    n = 0
    pw_tensor = (torch.tensor(pos_weight, device=device)
                  if pos_weight is not None else None)
    with torch.no_grad():
        for x, y, classes in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x, return_intermediates=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                out["logit"], y, reduction="sum", pos_weight=pw_tensor)
            bce_sum += loss.item()
            n += y.size(0)
            all_probs.append   (out["prob"].cpu().numpy())
            all_logits.append  (out["logit"].cpu().numpy())
            all_a_pas.append   (out["a_pas"].cpu().numpy())
            all_a_splice.append(out["a_splice"].cpu().numpy())
            all_pas_summary.append   (out["pas_summary"].cpu().numpy())
            all_splice_summary.append(out["splice_summary"].cpu().numpy())
            all_labels.append  (y.cpu().numpy())
            all_classes.extend (classes)
    probs           = np.concatenate(all_probs)
    logits          = np.concatenate(all_logits)
    a_pas           = np.concatenate(all_a_pas)
    a_splice        = np.concatenate(all_a_splice)
    pas_summary     = np.concatenate(all_pas_summary)
    splice_summary  = np.concatenate(all_splice_summary)
    labels   = np.concatenate(all_labels)
    classes  = np.array(all_classes)
    metrics = {"bce_loss": bce_sum / max(n, 1), "n": n}
    if 0 < labels.sum() < len(labels):
        metrics["auc"]   = float(roc_auc_score(labels, probs))
        metrics["auprc"] = float(average_precision_score(labels, probs))
        preds = (probs >= 0.5).astype(int)
        metrics["f1"]    = float(f1_score(labels, preds))
        metrics["acc"]   = float((preds == labels).mean())
        cm = confusion_matrix(labels, preds).tolist()
        metrics["cm"] = cm

    stratified = {}
    for cls in np.unique(classes):
        m = classes == cls
        stratified[str(cls)] = {
            "n":              int(m.sum()),
            "mean_a_pas":     float(a_pas[m].mean()),
            "mean_a_splice":  float(a_splice[m].mean()),
            "mean_prob":      float(probs[m].mean()),
            "mean_logit":     float(logits[m].mean()),
        }

    pas_head_w = model.pas_track.summary_head.weight.squeeze(0).detach().cpu().numpy()
    pas_head_w = model.pas_track.summary_head.weight.squeeze(0).detach().cpu().numpy()
    pas_names  = model.pas_track.pool.feature_names
    spl_names  = model.splice_track.pool.feature_names
    spl_motif_names = model.splice_motif_names

    if model.splice_track.per_motif_heads:
        spl_head_w_per_motif = model.splice_track.summary_head.detach().cpu().numpy()
        w_splice_motifs = model.head.weight[0, 1:].detach().cpu().numpy()
        spl_head_w = None
    else:
        spl_head_w = model.splice_track.summary_head.weight.squeeze(0).detach().cpu().numpy()

    feature_contribs = {}
    for cls in np.unique(classes):
        m = classes == cls
        pas_raw    = pas_summary[m].mean(axis=0)
        spl_raw    = splice_summary[m].mean(axis=0)
        entry = {
            "pas": [
                {"feat":   pas_names[k],
                 "raw":    float(pas_raw[k]),
                 "weight": float(pas_head_w[k]),
                 "contrib": float(pas_raw[k] * pas_head_w[k])}
                for k in range(len(pas_names))
            ],
        }
        if model.splice_track.per_motif_heads:
            entry["splice"] = []
            for mi, mn in enumerate(spl_motif_names):
                for k, fn in enumerate(spl_names):
                    raw_mk = float(spl_raw[mi, k])
                    w_mk   = float(spl_head_w_per_motif[mi, k])
                    w_top  = float(w_splice_motifs[mi])
                    entry["splice"].append({
                        "motif":   mn,
                        "feat":    fn,
                        "raw":     raw_mk,
                        "weight":  w_mk,
                        "w_top":   w_top,
                        "contrib": raw_mk * w_mk * w_top,
                    })
        else:
            entry["splice"] = [
                {"feat":   spl_names[k],
                 "raw":    float(spl_raw[k]),
                 "weight": float(spl_head_w[k]),
                 "contrib": float(spl_raw[k] * spl_head_w[k])}
                for k in range(len(spl_names))
            ]
        feature_contribs[str(cls)] = entry

    return metrics, stratified, feature_contribs, probs, logits, labels, classes


def smoothness_penalty(model):
    total = 0.0
    for track in (model.pas_track, model.splice_track):
        w = track.effect_conv.weight
        diff = w[:, :, 1:] - w[:, :, :-1]
        total = total + (diff ** 2).mean()
    return total


def train_one_epoch(model, loader, optimizer, device,
                     pos_weight=None, smoothness_lambda=0.0,
                     log_every=50, scheduler=None):
    model.train()
    total_loss = 0.0
    total_bce  = 0.0
    total_smooth = 0.0
    n = 0
    step = 0
    pw_tensor = (torch.tensor(pos_weight, device=device)
                  if pos_weight is not None and pos_weight != 1.0 else None)
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        bce = nn.functional.binary_cross_entropy_with_logits(
            out["logit"], y, pos_weight=pw_tensor)
        loss = bce
        if smoothness_lambda > 0:
            smooth = smoothness_penalty(model)
            loss = loss + smoothness_lambda * smooth
            total_smooth += float(smooth.item()) * y.size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * y.size(0)
        total_bce  += bce.item() * y.size(0)
        n += y.size(0)
        step += 1
        if step % log_every == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"    step {step:>5d}  running_bce={total_bce/n:.4f}  "
                  f"a_pas={out['a_pas'].mean().item():+.3f}  "
                  f"a_splice={out['a_splice'].mean().item():+.3f}  "
                  f"lr={cur_lr:.2e}",
                  flush=True)
    return {"bce": total_bce / max(n, 1),
            "smooth": total_smooth / max(n, 1) if smoothness_lambda > 0 else 0.0,
            "total": total_loss / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_dir",   required=True)
    ap.add_argument("--genome_fa",   required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--epochs",      type=int, default=50)
    ap.add_argument("--warmup_epochs", type=int, default=3,
                    help="PWM filters stay frozen for this many epochs")
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--lr_min",      type=float, default=5e-5,
                    help="Final LR for cosine annealing")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--smoothness_lambda", type=float, default=1e-3,
                    help="L2 smoothness penalty weight on effect-conv kernels")
    ap.add_argument("--pos_weight",  type=float, default=1.0,
                    help="Per-sample positive class weight in BCE")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--patience",    type=int, default=10,
                    help="Early-stop if val AUPRC doesn't improve for N epochs")
    ap.add_argument("--selection_metric", type=str, default="auprc",
                    choices=["auprc", "auc", "bce"],
                    help="Metric used for best-model checkpointing and "
                         "early stopping. AUPRC recommended for imbalanced "
                         "classification.")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--center_jitter", type=int, default=10,
                    help="At training time, jitter each candidate's center "
                         "by a uniform random shift in [-J, +J] nt. Prevents "
                         "the model from overfitting to the exact annotated "
                         "cleavage position. Val/test always use jitter=0.")
    ap.add_argument("--no_pwm_warm_start", action="store_true",
                    help="Train motif filters from scratch (no PWM init)")
    ap.add_argument("--oversample", action="store_true",
                    help="Oversample minority classes to match the largest. "
                         "Usually NOT needed because 01_build_puffin_dataset.py "
                         "already controls class composition via per-class caps. "
                         "Use only if you've intentionally built an unbalanced "
                         "training set and want runtime rebalancing.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # data
    print(f"\n[data] loading from {args.sites_dir}")
    sites_dir = args.sites_dir
    train_bed = os.path.join(sites_dir, "sites.train.bed")
    train_tsv = os.path.join(sites_dir, "sites.train.tsv")
    val_bed   = os.path.join(sites_dir, "sites.val.bed")
    val_tsv   = os.path.join(sites_dir, "sites.val.tsv")
    test_bed  = os.path.join(sites_dir, "sites.test.bed")
    test_tsv  = os.path.join(sites_dir, "sites.test.tsv")

    if args.oversample:
        meta_train = pd.read_csv(train_tsv, sep="\t")
        class_counts = meta_train["site_class"].value_counts().to_dict()
        oversample = {k: int(v) for k, v in class_counts.items()}
        print(f"  oversample target counts: {oversample}")
    else:
        oversample = None

    print(f"\n[train set]")
    train_ds = SiteSequenceDataset(train_bed, train_tsv, args.genome_fa,
                                    oversample_classes=oversample,
                                    seed=args.seed,
                                    center_jitter=args.center_jitter)
    print(f"\n[val set]")
    val_ds   = SiteSequenceDataset(val_bed, val_tsv, args.genome_fa)
    print(f"\n[test set]")
    test_ds  = SiteSequenceDataset(test_bed, test_tsv, args.genome_fa)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               collate_fn=collate, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               collate_fn=collate, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               collate_fn=collate, pin_memory=(device.type == "cuda"))

    # model
    print(f"\n[model] building PuffinCPA "
          f"(pwm_warm_start={not args.no_pwm_warm_start})")
    cfg = PuffinCPAConfig(pwm_warm_start=not args.no_pwm_warm_start)
    model = PuffinCPA(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    # optimizer
    optimizer = torch.optim.AdamW(
        model.pwm_param_groups(base_lr=args.lr,
                                weight_decay=args.weight_decay))
    if cfg.pwm_warm_start and args.warmup_epochs > 0:
        model.freeze_pwms()
        print(f"  PWMs FROZEN for first {args.warmup_epochs} epoch(s)")
    else:
        print(f"  PWMs UNFROZEN from epoch 1")

    # training
    log_path = os.path.join(args.out_dir, "training_log.tsv")
    with open(log_path, "w") as f:
        f.write("epoch\tphase\ttrain_bce\ttrain_smooth\tval_bce\t"
                "val_auc\tval_auprc\tval_f1\tlr\twall_s\n")
    strat_log_path = os.path.join(args.out_dir, "stratified_log.tsv")
    with open(strat_log_path, "w") as f:
        f.write("epoch\tsite_class\tn\tmean_a_pas\tmean_a_splice\tmean_prob\n")
    feat_log_path = os.path.join(args.out_dir, "feature_contributions_log.tsv")
    with open(feat_log_path, "w") as f:
        f.write("epoch\tsite_class\ttrack\tfeat\traw\tweight\tcontrib\n")

    n_train_steps_per_epoch = max(1, len(train_loader))
    total_steps = n_train_steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr_min)
    print(f"  cosine LR: {args.lr:.2e} -> {args.lr_min:.2e} over {total_steps:,} steps")

    higher_is_better = args.selection_metric in ("auprc", "auc")
    best_score = float("-inf") if higher_is_better else float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    t_start = time.time()

    def selection_score(metrics):
        if args.selection_metric == "auprc":   return metrics.get("auprc", 0.0)
        if args.selection_metric == "auc":     return metrics.get("auc",   0.0)
        if args.selection_metric == "bce":     return metrics.get("bce_loss",
                                                                  float("inf"))

    def is_better(new, old):
        return (new > old) if higher_is_better else (new < old)

    print(f"\n[training] {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr} -> {args.lr_min} (cosine), "
          f"selection_metric={args.selection_metric}")
    for epoch in range(1, args.epochs + 1):
        if (cfg.pwm_warm_start and epoch == args.warmup_epochs + 1):
            print(f"\n  *** unfreezing PWMs at start of epoch {epoch} ***")
            model.unfreeze_pwms()
        phase = "warmup" if (cfg.pwm_warm_start and epoch <= args.warmup_epochs) else "main"

        t0 = time.time()
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device,
            pos_weight=args.pos_weight,
            smoothness_lambda=args.smoothness_lambda,
            scheduler=scheduler)
        val_metrics, val_stratified, val_feature_contribs, _, _, _, _ = evaluate(
            model, val_loader, device, pos_weight=args.pos_weight)
        wall = time.time() - t0
        val_bce   = val_metrics["bce_loss"]
        val_auc   = val_metrics.get("auc",   float("nan"))
        val_auprc = val_metrics.get("auprc", float("nan"))
        val_f1    = val_metrics.get("f1",    float("nan"))
        cur_lr    = optimizer.param_groups[0]["lr"]

        print(f"\nepoch {epoch:>3d} [{phase:>6s}]  "
              f"train_bce={train_losses['bce']:.4f}  "
              f"smooth={train_losses['smooth']:.4f}  "
              f"val_bce={val_bce:.4f}  "
              f"val_auc={val_auc:.4f}  val_auprc={val_auprc:.4f}  "
              f"val_f1={val_f1:.4f}  lr={cur_lr:.2e}  ({wall:.1f}s)",
              flush=True)
        print(f"  class-stratified track outputs:")
        for cls in ("constitutive", "hard_neg", "medium_neg", "easy_neg"):
            if cls in val_stratified:
                s = val_stratified[cls]
                print(f"    {cls:13s}  n={s['n']:>5,}  "
                      f"a_pas={s['mean_a_pas']:+.3f}  "
                      f"a_splice={s['mean_a_splice']:+.3f}  "
                      f"prob={s['mean_prob']:.3f}")

        if ("constitutive" in val_feature_contribs and
                "hard_neg" in val_feature_contribs):
            print(f"  per-feature mean contributions to a_pas "
                  f"(constit vs hard_neg, sorted by |diff|):")
            const_fc = val_feature_contribs["constitutive"]["pas"]
            hard_fc  = val_feature_contribs["hard_neg"]["pas"]
            diffs = [(f["feat"], f["contrib"], hf["contrib"],
                      f["contrib"] - hf["contrib"], f["weight"])
                     for f, hf in zip(const_fc, hard_fc)]
            diffs.sort(key=lambda x: -abs(x[3]))
            for feat, c_contrib, h_contrib, diff, weight in diffs:
                print(f"    {feat:25s}  const={c_contrib:+.3f}  "
                      f"hard_neg={h_contrib:+.3f}  diff={diff:+.3f}  "
                      f"(w={weight:+.3f})")
            print(f"  per-feature mean contributions to a_splice "
                  f"(constit vs hard_neg, sorted by |diff|):")
            const_fc = val_feature_contribs["constitutive"]["splice"]
            hard_fc  = val_feature_contribs["hard_neg"]["splice"]
            per_motif = model.splice_track.per_motif_heads
            if per_motif:
                diffs = [(f"{f['motif']}:{f['feat']}",
                          f["contrib"], hf["contrib"],
                          f["contrib"] - hf["contrib"], f["weight"])
                         for f, hf in zip(const_fc, hard_fc)]
            else:
                diffs = [(f["feat"], f["contrib"], hf["contrib"],
                          f["contrib"] - hf["contrib"], f["weight"])
                         for f, hf in zip(const_fc, hard_fc)]
            diffs.sort(key=lambda x: -abs(x[3]))
            top_n = 15 if per_motif else len(diffs)
            for feat, c_contrib, h_contrib, diff, weight in diffs[:top_n]:
                print(f"    {feat:30s}  const={c_contrib:+.3f}  "
                      f"hard_neg={h_contrib:+.3f}  diff={diff:+.3f}  "
                      f"(w={weight:+.3f})")
            if per_motif:
                print(f"  per-motif aggregated contributions to a_splice:")
                from collections import defaultdict
                const_by_motif = defaultdict(float)
                hard_by_motif  = defaultdict(float)
                w_top_by_motif = {}
                for fc, hfc in zip(const_fc, hard_fc):
                    mn = fc["motif"]
                    const_by_motif[mn] += fc["contrib"]
                    hard_by_motif[mn]  += hfc["contrib"]
                    w_top_by_motif[mn]  = fc["w_top"]
                motif_diffs = [(mn,
                                const_by_motif[mn],
                                hard_by_motif[mn],
                                const_by_motif[mn] - hard_by_motif[mn],
                                w_top_by_motif[mn])
                               for mn in const_by_motif]
                motif_diffs.sort(key=lambda x: -abs(x[3]))
                for mn, c, h, d, w in motif_diffs:
                    print(f"    {mn:20s}  const={c:+.3f}  hard_neg={h:+.3f}  "
                          f"diff={d:+.3f}  (w_final={w:+.3f})")

        with open(log_path, "a") as f:
            f.write(f"{epoch}\t{phase}\t{train_losses['bce']:.5f}\t"
                    f"{train_losses['smooth']:.5f}\t{val_bce:.5f}\t"
                    f"{val_auc:.5f}\t{val_auprc:.5f}\t{val_f1:.5f}\t"
                    f"{cur_lr:.6f}\t{wall:.1f}\n")
        with open(strat_log_path, "a") as f:
            for cls, s in val_stratified.items():
                f.write(f"{epoch}\t{cls}\t{s['n']}\t{s['mean_a_pas']:.5f}\t"
                        f"{s['mean_a_splice']:.5f}\t{s['mean_prob']:.5f}\n")
        with open(feat_log_path, "a") as f:
            for cls, blocks in val_feature_contribs.items():
                for track_name, feat_list in blocks.items():
                    for fc in feat_list:
                        feat_label = fc.get("motif")
                        if feat_label is not None:
                            feat_label = f"{feat_label}:{fc['feat']}"
                        else:
                            feat_label = fc["feat"]
                        f.write(f"{epoch}\t{cls}\t{track_name}\t{feat_label}\t"
                                f"{fc['raw']:.5f}\t{fc['weight']:.5f}\t"
                                f"{fc['contrib']:.5f}\n")

        score = selection_score(val_metrics)
        if is_better(score, best_score):
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config":           cfg.__dict__,
                "epoch":            epoch,
                "val_metrics":      val_metrics,
                "val_stratified":   val_stratified,
                "selection_metric": args.selection_metric,
                "best_score":       float(score),
            }, os.path.join(args.out_dir, "best_model.pt"))
            print(f"    -> new best ({args.selection_metric}={score:.4f}), saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"    -> {args.patience} epochs without improvement on "
                      f"{args.selection_metric}, stopping")
                break

    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           cfg.__dict__,
        "epoch":            epoch,
    }, os.path.join(args.out_dir, "last_model.pt"))

    print(f"\n[test eval] loading best_model.pt (val epoch {best_epoch})")
    ckpt = torch.load(os.path.join(args.out_dir, "best_model.pt"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_stratified, test_feature_contribs, test_probs, test_logits, test_labels, test_classes = evaluate(
        model, test_loader, device, pos_weight=args.pos_weight)
    print(f"  test results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    print(f"  test class-stratified track outputs:")
    for cls in ("constitutive", "hard_neg", "medium_neg", "easy_neg"):
        if cls in test_stratified:
            s = test_stratified[cls]
            print(f"    {cls:13s}  n={s['n']:>5,}  "
                  f"a_pas={s['mean_a_pas']:+.3f}  "
                  f"a_splice={s['mean_a_splice']:+.3f}  "
                  f"prob={s['mean_prob']:.3f}")

    test_feat_path = os.path.join(args.out_dir, "test_feature_contributions.tsv")
    with open(test_feat_path, "w") as f:
        f.write("site_class\ttrack\tfeat\traw\tweight\tcontrib\n")
        for cls, blocks in test_feature_contribs.items():
            for track_name, feat_list in blocks.items():
                for fc in feat_list:
                    f.write(f"{cls}\t{track_name}\t{fc['feat']}\t"
                            f"{fc['raw']:.5f}\t{fc['weight']:.5f}\t"
                            f"{fc['contrib']:.5f}\n")
    print(f"       test feature contributions -> {test_feat_path}")

    metadata = {
        "model_name":     "PuffinCPA",
        "run_dir":        args.out_dir,
        "timestamp":      datetime.now().isoformat(),
        "config":         cfg.__dict__,
        "hyperparams": {
            "batch_size":          args.batch_size,
            "epochs_trained":      epoch,
            "warmup_epochs":       args.warmup_epochs,
            "lr":                  args.lr,
            "lr_min":              args.lr_min,
            "weight_decay":        args.weight_decay,
            "smoothness_lambda":   args.smoothness_lambda,
            "pos_weight":          args.pos_weight,
            "pwm_warm_start":      not args.no_pwm_warm_start,
            "oversample":          oversample is not None,
            "selection_metric":    args.selection_metric,
        },
        "winner_meta": {
            "best_epoch":      best_epoch,
            "best_score":      float(best_score),
            "selection_metric": args.selection_metric,
            "test_metrics": {
                "AUC":       test_metrics.get("auc"),
                "AUPRC":     test_metrics.get("auprc"),
                "ACC":       test_metrics.get("acc"),
                "F1":        test_metrics.get("f1"),
                "CM":        test_metrics.get("cm"),
                "threshold": 0.5,
                "label":     "test",
            },
            "test_stratified": test_stratified,
        },
        "n_parameters":   n_params,
        "wall_time_s":    time.time() - t_start,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\n[done] metadata -> {os.path.join(args.out_dir, 'metadata.json')}")

    np.savez(os.path.join(args.out_dir, "test_predictions.npz"),
              probs=test_probs, logits=test_logits, labels=test_labels,
              classes=test_classes)
    print(f"       predictions -> {os.path.join(args.out_dir, 'test_predictions.npz')}")


if __name__ == "__main__":
    main()