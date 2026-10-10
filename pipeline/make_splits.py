#!/usr/bin/env python3
# build train/val/test splits of positives and negatives by chromosome
import argparse
import os
import random
import sys


def read_chroms(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def split_bed(bed_path, chroms):
    rows = []
    with open(bed_path) as f:
        for line in f:
            if not line.strip():
                continue
            chrom = line.split("\t", 1)[0]
            if chrom in chroms:
                rows.append(line.rstrip("\n"))
    return rows


def write_bed(path, rows, sort_by_pos=False):
    if sort_by_pos:
        rows = sorted(rows, key=lambda r: (r.split("\t")[0], int(r.split("\t")[1])))
    with open(path, "w") as f:
        for r in rows:
            f.write(r + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_bed",          default="Dominant_CPA_baseline_total.bed")
    ap.add_argument("--neg_decoy_bed",    default="negatives_w_pa_clean.bed")
    ap.add_argument("--neg_nondecoy_bed", default="negatives_no_pa_clean.bed")
    ap.add_argument("--train_chroms",     default="train_chroms.txt")
    ap.add_argument("--val_chroms",       default="val_chroms.txt")
    ap.add_argument("--test_chroms",      default="test_chroms.txt")
    ap.add_argument("--fold",             type=int, default=40,
                    help="target negatives per positive")
    ap.add_argument("--decoy_per_total",  type=int, default=7,
                    help="1 decoy per N total negs (7 -> decoy:nondecoy = 1:6)")
    ap.add_argument("--seed",             type=int, default=42)
    ap.add_argument("--outdir",           default="splits")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    splits = {
        "train": read_chroms(args.train_chroms),
        "val":   read_chroms(args.val_chroms),
        "test":  read_chroms(args.test_chroms),
    }

    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = splits[names[i]] & splits[names[j]]
            if overlap:
                sys.exit(f"ERROR: chroms in both {names[i]} and {names[j]}: {sorted(overlap)}")

    print("Chrom split sizes:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} chroms  ({', '.join(sorted(v))})")
    print()

    summary = []
    for split, chroms in splits.items():
        pos       = split_bed(args.pos_bed,          chroms)
        decoy     = split_bed(args.neg_decoy_bed,    chroms)
        nondecoy  = split_bed(args.neg_nondecoy_bed, chroms)

        n_pos = len(pos)
        target_neg      = n_pos * args.fold
        target_decoy    = target_neg // args.decoy_per_total
        target_nondecoy = target_neg - target_decoy

        use_decoy    = min(target_decoy,    len(decoy))
        use_nondecoy = min(target_nondecoy, len(nondecoy))
        short = (use_decoy < target_decoy) or (use_nondecoy < target_nondecoy)

        print(f"[{split}]")
        print(f"  positives:        {n_pos:,}")
        print(f"  target negs:      {target_neg:,}  "
              f"(decoy {target_decoy:,} + nondecoy {target_nondecoy:,})")
        print(f"  available:        decoy {len(decoy):,}, nondecoy {len(nondecoy):,}")
        print(f"  using:            decoy {use_decoy:,}, nondecoy {use_nondecoy:,}")
        if short:
            print(f"  ** WARNING: insufficient negatives, actual fold will be lower **")

        decoy_sample    = rng.sample(decoy,    use_decoy)
        nondecoy_sample = rng.sample(nondecoy, use_nondecoy)

        write_bed(f"{args.outdir}/pos_{split}.bed", pos, sort_by_pos=True)
        write_bed(f"{args.outdir}/neg_{split}_decoy.bed",    decoy_sample,    sort_by_pos=True)
        write_bed(f"{args.outdir}/neg_{split}_nondecoy.bed", nondecoy_sample, sort_by_pos=True)
        write_bed(f"{args.outdir}/neg_{split}.bed",
                  decoy_sample + nondecoy_sample, sort_by_pos=True)

        actual_fold = (use_decoy + use_nondecoy) / max(n_pos, 1)
        actual_decoy_pct = use_decoy / max(use_decoy + use_nondecoy, 1) * 100
        summary.append({
            "split": split,
            "n_pos": n_pos,
            "n_neg": use_decoy + use_nondecoy,
            "n_decoy": use_decoy,
            "n_nondecoy": use_nondecoy,
            "fold": actual_fold,
            "decoy_pct": actual_decoy_pct,
        })
        print(f"  -> wrote pos_{split}.bed, neg_{split}.bed "
              f"(decoy + nondecoy variants saved separately)")
        print()

    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'split':<7} {'pos':>8} {'neg':>10} {'fold':>7} "
          f"{'decoy':>9} {'nondecoy':>10} {'decoy%':>8}")
    for s in summary:
        print(f"{s['split']:<7} {s['n_pos']:>8,} {s['n_neg']:>10,} "
              f"{s['fold']:>6.1f}x {s['n_decoy']:>9,} {s['n_nondecoy']:>10,} "
              f"{s['decoy_pct']:>7.1f}%")

    print()
    print("Next: extract FASTAs (strand-aware) for the CNN")
    print(f"  GENOME=hg38.fa   # path to your hg38 fasta")
    print(f"  for split in train val test; do")
    print(f"      bedtools getfasta -s -name -fi $GENOME \\")
    print(f"          -bed {args.outdir}/pos_${{split}}.bed > {args.outdir}/pos_${{split}}.fa")
    print(f"      bedtools getfasta -s -name -fi $GENOME \\")
    print(f"          -bed {args.outdir}/neg_${{split}}.bed > {args.outdir}/neg_${{split}}.fa")
    print(f"  done")


if __name__ == "__main__":
    main()