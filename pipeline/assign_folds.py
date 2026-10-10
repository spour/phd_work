#!/usr/bin/env python3
# (step 1 of the cross-fit rebuild) partition whole chromosomes into k folds balanced
import argparse, os, sys
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_bed", required=True,
                    help="positives BED (chrom in col1); counts sites per chrom")
    ap.add_argument("-k", "--folds", type=int, default=5)
    ap.add_argument("--neg_decoy_bed",    default="data/negatives_w_pa_clean.bed")
    ap.add_argument("--neg_nondecoy_bed", default="data/negatives_no_pa_clean.bed")
    ap.add_argument("--outdir", default="folds")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    cnt = defaultdict(int)
    with open(a.pos_bed) as f:
        for line in f:
            if not line.strip(): continue
            cnt[line.split("\t", 1)[0]] += 1
    if not cnt: sys.exit("[err] no positives read")

    chroms = sorted(cnt, key=lambda c: cnt[c], reverse=True)
    fold_chroms = [[] for _ in range(a.folds)]
    fold_load   = [0]*a.folds
    for c in chroms:
        k = min(range(a.folds), key=lambda i: fold_load[i])
        fold_chroms[k].append(c); fold_load[k] += cnt[c]

    print(f"[folds] {len(chroms)} chromosomes, {sum(cnt.values()):,} positives, K={a.folds}",
          file=sys.stderr)
    with open(f"{a.outdir}/fold_assignments.tsv","w") as fo:
        fo.write("chrom\tn_pos\tfold\n")
        for k in range(a.folds):
            chs = sorted(fold_chroms[k], key=lambda c:(len(c),c))
            with open(f"{a.outdir}/fold_{k}_chroms.txt","w") as cf:
                for c in chs: cf.write(c+"\n")
            for c in chs: fo.write(f"{c}\t{cnt[c]}\t{k}\n")
            print(f"  fold {k}: {fold_load[k]:>6,} pos  ({len(chs)} chr: {', '.join(chs)})",
                  file=sys.stderr)
    spread = (max(fold_load)-min(fold_load))/ (sum(fold_load)/a.folds)
    print(f"[balance] fold-size spread = {spread:.1%} of mean "
          f"(min {min(fold_load):,}, max {max(fold_load):,})", file=sys.stderr)

    with open(f"{a.outdir}/run_make_splits.sh","w") as sh:
        sh.write("#!/bin/bash\nset -euo pipefail\n")
        sh.write("# per-fold splits: test=fold k (scoring target), val=next fold, train=rest\n")
        for k in range(a.folds):
            vk = (k+1) % a.folds
            train_folds = [j for j in range(a.folds) if j not in (k, vk)]
            sh.write(f"\n# ---- fold {k}: test=fold{k} val=fold{vk} train=folds{train_folds} ----\n")
            sh.write(f"cat {a.outdir}/fold_{k}_chroms.txt > {a.outdir}/test_chroms_f{k}.txt\n")
            sh.write(f"cat {a.outdir}/fold_{vk}_chroms.txt > {a.outdir}/val_chroms_f{k}.txt\n")
            sh.write("cat " + " ".join(f"{a.outdir}/fold_{j}_chroms.txt" for j in train_folds)
                     + f" > {a.outdir}/train_chroms_f{k}.txt\n")
            sh.write(
                f"python3 scripts/make_splits.py "
                f"--pos_bed {a.pos_bed} "
                f"--neg_decoy_bed {a.neg_decoy_bed} --neg_nondecoy_bed {a.neg_nondecoy_bed} "
                f"--train_chroms {a.outdir}/train_chroms_f{k}.txt "
                f"--val_chroms {a.outdir}/val_chroms_f{k}.txt "
                f"--test_chroms {a.outdir}/test_chroms_f{k}.txt "
                f"--outdir splits/fold{k}\n")
    print(f"[out] {a.outdir}/fold_*_chroms.txt, fold_assignments.tsv, run_make_splits.sh",
          file=sys.stderr)
    print("[next] review the balance above, then: bash folds/run_make_splits.sh", file=sys.stderr)

if __name__ == "__main__":
    main()