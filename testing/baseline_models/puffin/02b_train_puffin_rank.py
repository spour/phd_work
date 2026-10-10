#!/usr/bin/env python3
# train puffincpa-rank with pairwise hinge loss
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from puffin_cpa import (
    PuffinCPA, PuffinCPAConfig, encode_sequence,
    UP_FLANK, DOWN_FLANK, SEQ_LEN,
)


class GeneGroupDataset(Dataset):

    def __init__(self, tsv_path, genome_fa):
        self.genome_fa = genome_fa
        self._fa = None
        meta = pd.read_csv(tsv_path, sep="\t")
        meta["center"] = (meta["start"] + meta["end"]) // 2
        self.genes = []
        for gene, gdf in meta.groupby("gene"):
            n_pos = int((gdf["label"] == 1).sum())
            if n_pos != 1:
                continue
            if len(gdf) < 2:
                continue
            self.genes.append((gene, gdf.reset_index(drop=True)))
        n_pos_total = sum(int((g[1]["label"] == 1).sum()) for g in self.genes)
        n_neg_total = sum(int((g[1]["label"] == 0).sum()) for g in self.genes)
        print(f"  loaded {len(self.genes):,} genes from {tsv_path}")
        print(f"    {n_pos_total:,} positives, {n_neg_total:,} negatives")
        per_gene_n = [len(g[1]) for g in self.genes]
        print(f"    per-gene candidates: median={int(np.median(per_gene_n))}, "
              f"min={min(per_gene_n)}, max={max(per_gene_n)}")

    def _ensure_fa(self):
        if self._fa is None:
            self._fa = pysam.FastaFile(self.genome_fa)

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        self._ensure_fa()
        gene, gdf = self.genes[idx]
        xs = []
        for _, row in gdf.iterrows():
            chrom  = row["chrom"]
            center = int(row["center"])
            strand = row["strand"]
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
            xs.append(encode_sequence(seq, pad_len=SEQ_LEN))
        x = torch.from_numpy(np.stack(xs))
        labels  = torch.tensor(gdf["label"].values, dtype=torch.float32)
        classes = list(gdf["site_class"].values)
        return x, labels, classes, gene


def collate_gene_groups(batch):
    xs       = torch.cat([b[0] for b in batch], dim=0)
    labels   = torch.cat([b[1] for b in batch], dim=0)
    classes  = []
    for b in batch: classes.extend(b[2])
    genes    = [b[3] for b in batch]
    sizes    = [b[0].shape[0] for b in batch]
    return xs, labels, classes, genes, sizes


def pairwise_hinge_loss(logits, labels, sizes, margin=1.0, center_logits=True):
    losses = []
    start = 0
    for n in sizes:
        glogits = logits[start:start + n]
        glabels = labels[start:start + n]
        start += n
        pos_idx = (glabels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (glabels == 0).nonzero(as_tuple=True)[0]
        if len(pos_idx) != 1 or len(neg_idx) == 0:
            continue
        if center_logits:
            glogits = glogits - glogits.mean()
        pos_logit  = glogits[pos_idx[0]]
        neg_logits = glogits[neg_idx]
        diffs = neg_logits - pos_logit + margin
        gene_loss = F.softplus(diffs).mean()
        losses.append(gene_loss)
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def evaluate_ranking(model, loader, device, margin=1.0, center_logits=True):
    model.eval()
    total_loss = 0.0
    n_genes = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    pos_ranks = []
    within_gene_auprcs = []

    all_a_pas    = []
    all_a_splice = []
    all_probs    = []
    all_classes  = []
    all_pas_summary    = []
    all_splice_summary = []

    with torch.no_grad():
        for xs, labels, classes, genes, sizes in loader:
            xs = xs.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)
            out = model(xs, return_intermediates=True)
            loss = pairwise_hinge_loss(out["logit"], labels_dev, sizes,
                                        margin=margin, center_logits=center_logits)
            total_loss += loss.item() * len(sizes)
            n_genes += len(sizes)
            logits   = out["logit"].cpu().numpy()
            probs    = out["prob"].cpu().numpy()
            a_pas    = out["a_pas"].cpu().numpy()
            a_splice = out["a_splice"].cpu().numpy()
            pas_summ = out["pas_summary"].cpu().numpy()
            spl_summ = out["splice_summary"].cpu().numpy()
            all_a_pas.append(a_pas)
            all_a_splice.append(a_splice)
            all_probs.append(probs)
            all_pas_summary.append(pas_summ)
            all_splice_summary.append(spl_summ)
            all_classes.extend(classes)
            labels_np = labels.numpy()
            start = 0
            for n in sizes:
                glogits = logits[start:start + n]
                glabels = labels_np[start:start + n]
                start += n
                pos_idx = np.where(glabels == 1)[0]
                if len(pos_idx) != 1:
                    continue
                pos = pos_idx[0]
                order = np.argsort(-glogits)
                rank_of_pos = int(np.where(order == pos)[0][0]) + 1
                pos_ranks.append(rank_of_pos)
                if rank_of_pos == 1: top1_hits += 1
                if rank_of_pos <= 3: top3_hits += 1
                if rank_of_pos <= 5: top5_hits += 1
                if 0 < glabels.sum() < len(glabels):
                    auprc = average_precision_score(glabels, glogits)
                    within_gene_auprcs.append(auprc)

    metrics = {
        "loss":                   total_loss / max(n_genes, 1),
        "n_genes":                n_genes,
        "top1_acc":               top1_hits / max(n_genes, 1),
        "top3_acc":               top3_hits / max(n_genes, 1),
        "top5_acc":               top5_hits / max(n_genes, 1),
        "mean_positive_rank":     float(np.mean(pos_ranks)) if pos_ranks else None,
        "median_positive_rank":   float(np.median(pos_ranks)) if pos_ranks else None,
        "mean_within_gene_auprc": float(np.mean(within_gene_auprcs))
                                   if within_gene_auprcs else None,
    }

    a_pas    = np.concatenate(all_a_pas)
    a_splice = np.concatenate(all_a_splice)
    probs    = np.concatenate(all_probs)
    pas_summary    = np.concatenate(all_pas_summary)
    splice_summary = np.concatenate(all_splice_summary)
    classes = np.array(all_classes)
    stratified = {}
    for cls in np.unique(classes):
        m = classes == cls
        stratified[str(cls)] = {
            "n":              int(m.sum()),
            "mean_a_pas":     float(a_pas[m].mean()),
            "mean_a_splice":  float(a_splice[m].mean()),
            "mean_prob":      float(probs[m].mean()),
        }

    pas_head_w = model.pas_track.summary_head.weight.squeeze(0).detach().cpu().numpy()
    spl_head_w = model.splice_track.summary_head.weight.squeeze(0).detach().cpu().numpy()
    pas_names  = model.pas_track.pool.feature_names
    spl_names  = model.splice_track.pool.feature_names
    feature_contribs = {}
    for cls in np.unique(classes):
        m = classes == cls
        pas_raw    = pas_summary[m].mean(axis=0)
        spl_raw    = splice_summary[m].mean(axis=0)
        feature_contribs[str(cls)] = {
            "pas": [
                {"feat":   pas_names[k],
                 "raw":    float(pas_raw[k]),
                 "weight": float(pas_head_w[k]),
                 "contrib": float(pas_raw[k] * pas_head_w[k])}
                for k in range(len(pas_names))
            ],
            "splice": [
                {"feat":   spl_names[k],
                 "raw":    float(spl_raw[k]),
                 "weight": float(spl_head_w[k]),
                 "contrib": float(spl_raw[k] * spl_head_w[k])}
                for k in range(len(spl_names))
            ],
        }

    motif_contribs = compute_motif_attributions(model, loader, device, classes)

    return metrics, stratified, feature_contribs, motif_contribs


def compute_motif_attributions(model, loader, device, classes_all,
                                max_per_class=500):
    classes_all = np.array(classes_all)
    unique_classes = np.unique(classes_all)
    pas_motif_names    = model.pas_motif_names
    splice_motif_names = model.splice_motif_names
    n_pas    = len(pas_motif_names)
    n_splice = len(splice_motif_names)

    pas_sum    = {c: np.zeros(n_pas)    for c in unique_classes}
    splice_sum = {c: np.zeros(n_splice) for c in unique_classes}
    counts     = {c: 0                  for c in unique_classes}

    idx_global = 0
    model.eval()
    with torch.no_grad():
        for xs, labels, classes_batch, _, _ in loader:
            xs = xs.to(device, non_blocking=True)
            classes_batch_arr = np.array(classes_batch)
            uncapped = np.array([counts[c] < max_per_class for c in classes_batch_arr])
            if not uncapped.any():
                idx_global += len(classes_batch)
                continue
            attr = model.attribution(xs)
            pas_contribs    = attr["pas_contribs"].cpu().numpy()
            splice_contribs = attr["splice_contribs"].cpu().numpy()
            for i, c in enumerate(classes_batch_arr):
                if counts[c] >= max_per_class:
                    continue
                pas_sum[c]    += pas_contribs[i]
                splice_sum[c] += splice_contribs[i]
                counts[c] += 1
            idx_global += len(classes_batch)
            if all(counts[c] >= max_per_class for c in unique_classes
                    if counts[c] < max_per_class or True):
                if all(counts[c] >= max_per_class for c in unique_classes):
                    break

    out = {}
    for c in unique_classes:
        n = max(counts[c], 1)
        out[str(c)] = {
            "n_used":                int(counts[c]),
            "pas_motif_names":       pas_motif_names,
            "splice_motif_names":    splice_motif_names,
            "pas_motif_contribs":    (pas_sum[c]    / n).tolist(),
            "splice_motif_contribs": (splice_sum[c] / n).tolist(),
        }
    return out


def smoothness_penalty(model):
    total = 0.0
    for track in (model.pas_track, model.splice_track):
        w = track.effect_conv.weight
        diff = w[:, :, 1:] - w[:, :, :-1]
        total = total + (diff ** 2).mean()
    return total


def train_one_epoch(model, loader, optimizer, device, margin=1.0,
                     smoothness_lambda=0.0, spread_lambda=0.0,
                     scheduler=None, log_every=20,
                     center_logits=True):
    model.train()
    total_rank_loss = 0.0
    total_smooth    = 0.0
    total_spread    = 0.0
    n_genes = 0
    step = 0
    for xs, labels, _, _, sizes in loader:
        xs = xs.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(xs)
        loss = pairwise_hinge_loss(out["logit"], labels_dev, sizes,
                                    margin=margin, center_logits=center_logits)
        loss_total = loss
        if smoothness_lambda > 0:
            smooth = smoothness_penalty(model)
            loss_total = loss_total + smoothness_lambda * smooth
            total_smooth += float(smooth.item()) * len(sizes)
        if spread_lambda > 0:
            spread_terms = []
            start = 0
            for n in sizes:
                glogits = out["logit"][start:start + n]
                start += n
                centered = glogits - glogits.mean()
                spread_terms.append((centered ** 2).mean())
            spread = torch.stack(spread_terms).mean()
            loss_total = loss_total + spread_lambda * spread
            total_spread += float(spread.item()) * len(sizes)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_rank_loss += loss.item() * len(sizes)
        n_genes += len(sizes)
        step += 1
        if step % log_every == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"    step {step:>5d}  running_loss={total_rank_loss/n_genes:.4f}  "
                  f"lr={cur_lr:.2e}", flush=True)
    return {
        "rank_loss":   total_rank_loss / max(n_genes, 1),
        "smooth_loss": total_smooth / max(n_genes, 1) if smoothness_lambda > 0 else 0.0,
        "spread_loss": total_spread / max(n_genes, 1) if spread_lambda > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",      required=True,
                    help="Directory with gene_groups.{train,val,test}.tsv")
    ap.add_argument("--genome_fa",     required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--margin",        type=float, default=1.0,
                    help="Hinge margin for pairwise ranking loss")
    ap.add_argument("--no_center_logits", action="store_true",
                    help="Disable per-gene logit mean-centering in the loss. "
                         "Without centering, the model can solve ranking by "
                         "inflating absolute logit magnitudes, which makes "
                         "per-feature contributions uninterpretable across "
                         "epochs. Default ON.")
    ap.add_argument("--batch_size",    type=int, default=4,
                    help="Number of GENES per batch (each gene is variable-sized)")
    ap.add_argument("--epochs",        type=int, default=50)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--lr",            type=float, default=5e-4)
    ap.add_argument("--lr_min",        type=float, default=5e-5)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--smoothness_lambda", type=float, default=1e-3)
    ap.add_argument("--spread_lambda",     type=float, default=1e-4,
                    help="Per-gene centered spread penalty on logits. "
                         "Keeps the model from drifting toward huge absolute "
                         "logit magnitudes (the pairwise loss alone is "
                         "scale-invariant). Set to 0 to disable.")
    ap.add_argument("--num_workers",   type=int, default=1)
    ap.add_argument("--patience",      type=int, default=10)
    ap.add_argument("--selection_metric", type=str, default="top1_acc",
                    choices=["top1_acc", "top3_acc",
                              "mean_within_gene_auprc", "loss"])
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--no_pwm_warm_start", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # data
    print(f"\n[data] loading from {args.data_dir}")
    train_tsv = os.path.join(args.data_dir, "gene_groups.train.tsv")
    val_tsv   = os.path.join(args.data_dir, "gene_groups.val.tsv")
    test_tsv  = os.path.join(args.data_dir, "gene_groups.test.tsv")
    print(f"\n[train set]")
    train_ds = GeneGroupDataset(train_tsv, args.genome_fa)
    print(f"\n[val set]")
    val_ds   = GeneGroupDataset(val_tsv,   args.genome_fa)
    print(f"\n[test set]")
    test_ds  = GeneGroupDataset(test_tsv,  args.genome_fa)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               collate_fn=collate_gene_groups,
                               pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               collate_fn=collate_gene_groups,
                               pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               collate_fn=collate_gene_groups,
                               pin_memory=(device.type == "cuda"))

    # model
    print(f"\n[model] building PuffinCPA "
          f"(pwm_warm_start={not args.no_pwm_warm_start})")
    cfg = PuffinCPAConfig(pwm_warm_start=not args.no_pwm_warm_start)
    model = PuffinCPA(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.pwm_param_groups(base_lr=args.lr,
                                weight_decay=args.weight_decay))
    if cfg.pwm_warm_start and args.warmup_epochs > 0:
        model.freeze_pwms()
        print(f"  PWMs FROZEN for first {args.warmup_epochs} epoch(s)")

    n_train_steps_per_epoch = max(1, len(train_loader))
    total_steps = n_train_steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr_min)
    print(f"  cosine LR: {args.lr:.2e} -> {args.lr_min:.2e} over {total_steps:,} steps")

    # log files
    log_path = os.path.join(args.out_dir, "training_log.tsv")
    strat_path = os.path.join(args.out_dir, "stratified_log.tsv")
    feat_path = os.path.join(args.out_dir, "feature_contributions_log.tsv")
    motif_path = os.path.join(args.out_dir, "motif_contributions_log.tsv")
    with open(log_path, "w") as f:
        f.write("epoch\tphase\ttrain_loss\ttrain_smooth\ttrain_spread\tval_loss\t"
                "val_top1\tval_top3\tval_top5\tval_mwg_auprc\t"
                "val_mean_rank\tlr\twall_s\n")
    with open(strat_path, "w") as f:
        f.write("epoch\tsite_class\tn\tmean_a_pas\tmean_a_splice\tmean_prob\n")
    with open(feat_path, "w") as f:
        f.write("epoch\tsite_class\ttrack\tfeat\traw\tweight\tcontrib\n")
    with open(motif_path, "w") as f:
        f.write("epoch\tsite_class\ttrack\tmotif\tcontrib\n")

    # training loop
    higher_is_better = args.selection_metric != "loss"
    best_score = float("-inf") if higher_is_better else float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    t_start = time.time()

    def is_better(new, old):
        return (new > old) if higher_is_better else (new < old)

    print(f"\n[training] {args.epochs} epochs, batch_size={args.batch_size} genes, "
          f"margin={args.margin}, selection={args.selection_metric}")
    center_logits = not args.no_center_logits
    print(f"  loss: pairwise_hinge(margin={args.margin}, "
          f"center_logits={center_logits})")
    for epoch in range(1, args.epochs + 1):
        if cfg.pwm_warm_start and epoch == args.warmup_epochs + 1:
            print(f"\n  *** unfreezing PWMs at start of epoch {epoch} ***")
            model.unfreeze_pwms()
        phase = "warmup" if (cfg.pwm_warm_start and epoch <= args.warmup_epochs) else "main"

        t0 = time.time()
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device,
            margin=args.margin,
            smoothness_lambda=args.smoothness_lambda,
            spread_lambda=args.spread_lambda,
            scheduler=scheduler,
            center_logits=center_logits)
        val_metrics, val_stratified, val_feature_contribs, val_motif_contribs = evaluate_ranking(
            model, val_loader, device, margin=args.margin,
            center_logits=center_logits)
        wall = time.time() - t0

        cur_lr = optimizer.param_groups[0]["lr"]
        val_top1 = val_metrics["top1_acc"]
        val_top3 = val_metrics["top3_acc"]
        val_top5 = val_metrics["top5_acc"]
        val_mwg  = val_metrics["mean_within_gene_auprc"]
        val_mrk  = val_metrics["mean_positive_rank"]
        print(f"\nepoch {epoch:>3d} [{phase:>6s}]  "
              f"train_loss={train_losses['rank_loss']:.4f}  "
              f"smooth={train_losses['smooth_loss']:.4f}  "
              f"spread={train_losses['spread_loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_top1={val_top1:.4f}  val_top3={val_top3:.4f}  "
              f"val_top5={val_top5:.4f}  "
              f"val_mwg_auprc={val_mwg:.4f}  "
              f"val_mean_rank={val_mrk:.2f}  "
              f"lr={cur_lr:.2e}  ({wall:.1f}s)",
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
            print(f"  per-feature contribs to a_pas (constit vs hard_neg):")
            const_fc = val_feature_contribs["constitutive"]["pas"]
            hard_fc  = val_feature_contribs["hard_neg"]["pas"]
            diffs = [(f["feat"], f["contrib"], hf["contrib"],
                      f["contrib"] - hf["contrib"], f["weight"])
                     for f, hf in zip(const_fc, hard_fc)]
            diffs.sort(key=lambda x: -abs(x[3]))
            for feat, c, h, d, w in diffs:
                print(f"    {feat:25s}  const={c:+.3f}  hard={h:+.3f}  "
                      f"diff={d:+.3f}  (w={w:+.3f})")
            print(f"  per-feature contribs to a_splice (constit vs hard_neg):")
            const_fc = val_feature_contribs["constitutive"]["splice"]
            hard_fc  = val_feature_contribs["hard_neg"]["splice"]
            diffs = [(f["feat"], f["contrib"], hf["contrib"],
                      f["contrib"] - hf["contrib"], f["weight"])
                     for f, hf in zip(const_fc, hard_fc)]
            diffs.sort(key=lambda x: -abs(x[3]))
            for feat, c, h, d, w in diffs:
                print(f"    {feat:25s}  const={c:+.3f}  hard={h:+.3f}  "
                      f"diff={d:+.3f}  (w={w:+.3f})")

        if ("constitutive" in val_motif_contribs and
                "hard_neg" in val_motif_contribs):
            const_mc = val_motif_contribs["constitutive"]
            hard_mc  = val_motif_contribs["hard_neg"]
            print(f"  per-MOTIF contribs to a_pas (top 5 by |diff|):")
            pas_diffs = [
                (const_mc["pas_motif_names"][k],
                 const_mc["pas_motif_contribs"][k],
                 hard_mc["pas_motif_contribs"][k],
                 const_mc["pas_motif_contribs"][k] - hard_mc["pas_motif_contribs"][k])
                for k in range(len(const_mc["pas_motif_names"]))
            ]
            pas_diffs.sort(key=lambda x: -abs(x[3]))
            for nm, c, h, d in pas_diffs[:5]:
                print(f"    {nm:25s}  const={c:+.3f}  hard={h:+.3f}  diff={d:+.3f}")
            print(f"  per-MOTIF contribs to a_splice (all motifs, sorted by |diff|):")
            spl_diffs = [
                (const_mc["splice_motif_names"][k],
                 const_mc["splice_motif_contribs"][k],
                 hard_mc["splice_motif_contribs"][k],
                 const_mc["splice_motif_contribs"][k] - hard_mc["splice_motif_contribs"][k])
                for k in range(len(const_mc["splice_motif_names"]))
            ]
            spl_diffs.sort(key=lambda x: -abs(x[3]))
            for nm, c, h, d in spl_diffs[:8]:
                print(f"    {nm:25s}  const={c:+.3f}  hard={h:+.3f}  diff={d:+.3f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch}\t{phase}\t{train_losses['rank_loss']:.5f}\t"
                    f"{train_losses['smooth_loss']:.5f}\t"
                    f"{train_losses['spread_loss']:.5f}\t"
                    f"{val_metrics['loss']:.5f}\t"
                    f"{val_top1:.5f}\t{val_top3:.5f}\t{val_top5:.5f}\t"
                    f"{val_mwg:.5f}\t{val_mrk:.3f}\t{cur_lr:.6f}\t{wall:.1f}\n")
        with open(strat_path, "a") as f:
            for cls, s in val_stratified.items():
                f.write(f"{epoch}\t{cls}\t{s['n']}\t{s['mean_a_pas']:.5f}\t"
                        f"{s['mean_a_splice']:.5f}\t{s['mean_prob']:.5f}\n")
        with open(feat_path, "a") as f:
            for cls, blocks in val_feature_contribs.items():
                for track_name, feat_list in blocks.items():
                    for fc in feat_list:
                        f.write(f"{epoch}\t{cls}\t{track_name}\t{fc['feat']}\t"
                                f"{fc['raw']:.5f}\t{fc['weight']:.5f}\t"
                                f"{fc['contrib']:.5f}\n")
        with open(motif_path, "a") as f:
            for cls, blocks in val_motif_contribs.items():
                for nm, c in zip(blocks["pas_motif_names"],
                                  blocks["pas_motif_contribs"]):
                    f.write(f"{epoch}\t{cls}\tpas\t{nm}\t{c:.5f}\n")
                for nm, c in zip(blocks["splice_motif_names"],
                                  blocks["splice_motif_contribs"]):
                    f.write(f"{epoch}\t{cls}\tsplice\t{nm}\t{c:.5f}\n")

        score = val_metrics.get(args.selection_metric, float("nan"))
        if is_better(score, best_score):
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
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
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"    -> {args.patience} epochs without improvement, stopping")
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
    test_metrics, test_stratified, test_feature_contribs, test_motif_contribs = evaluate_ranking(
        model, test_loader, device, margin=args.margin,
        center_logits=center_logits)
    print(f"  test results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    print(f"  test class-stratified:")
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

    metadata = {
        "model_name":    "PuffinCPA-Rank",
        "run_dir":       args.out_dir,
        "timestamp":     datetime.now().isoformat(),
        "config":        cfg.__dict__,
        "hyperparams": {
            "batch_size":          args.batch_size,
            "epochs_trained":      epoch,
            "warmup_epochs":       args.warmup_epochs,
            "margin":              args.margin,
            "center_logits":       center_logits,
            "lr":                  args.lr,
            "lr_min":              args.lr_min,
            "weight_decay":        args.weight_decay,
            "smoothness_lambda":   args.smoothness_lambda,
            "spread_lambda":       args.spread_lambda,
            "selection_metric":    args.selection_metric,
        },
        "winner_meta": {
            "best_epoch":  best_epoch,
            "best_score":  float(best_score),
            "test_metrics": {
                "loss":                   test_metrics["loss"],
                "top1_acc":               test_metrics["top1_acc"],
                "top3_acc":               test_metrics["top3_acc"],
                "top5_acc":               test_metrics["top5_acc"],
                "mean_positive_rank":     test_metrics["mean_positive_rank"],
                "median_positive_rank":   test_metrics["median_positive_rank"],
                "mean_within_gene_auprc": test_metrics["mean_within_gene_auprc"],
                "n_genes_evaluated":      test_metrics["n_genes"],
                "label":                  "test",
            },
            "test_stratified": test_stratified,
        },
        "n_parameters":  n_params,
        "wall_time_s":   time.time() - t_start,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\n[done] metadata -> {os.path.join(args.out_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()