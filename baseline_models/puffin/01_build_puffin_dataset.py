#!/usr/bin/env python3
# construct positive + 3-tier negative site sets for puffincpa training
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd


def read_bed(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            try:
                score = float(p[4])
            except (ValueError, IndexError):
                score = 0.0
            rows.append({
                "chrom":  p[0],
                "start":  int(p[1]),
                "end":    int(p[2]),
                "name":   p[3],
                "score":  score,
                "strand": p[5],
            })
    return pd.DataFrame(rows)


def write_bed(df, path, cols=("chrom","start","end","name","score","strand")):
    df[list(cols)].to_csv(path, sep="\t", header=False, index=False)


def read_chroms(path):
    return set(open(path).read().split())


def bedtools_intersect(a_path, b_path, *flags):
    cmd = ["bedtools", "intersect"] + list(flags) + ["-a", a_path, "-b", b_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if not result.stdout.strip():
        return pd.DataFrame(columns=["chrom","start","end","name","score","strand"])
    rows = []
    for line in result.stdout.strip().split("\n"):
        p = line.split("\t")
        rows.append({
            "chrom":  p[0],
            "start":  int(p[1]),
            "end":    int(p[2]),
            "name":   p[3] if len(p) > 3 else "",
            "score":  float(p[4]) if len(p) > 4 and p[4] not in (".", "") else 0.0,
            "strand": p[5] if len(p) > 5 else "+",
        })
    return pd.DataFrame(rows)


def temp_bed(df, suffix=".bed"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    df[["chrom","start","end","name","score","strand"]].to_csv(
        f.name, sep="\t", header=False, index=False)
    f.close()
    return f.name


def build_atlas_union(atlas_beds, out_dir):
    print(f"\n[atlas] unioning {len(atlas_beds)} atlas BED(s)")
    cat_path = os.path.join(out_dir, "_atlas_concat.bed")
    with open(cat_path, "w") as fout:
        for bp in atlas_beds:
            with open(bp) as fin:
                for line in fin:
                    if not line.strip() or line.startswith("#"):
                        continue
                    p = line.rstrip("\n").split("\t")
                    if len(p) < 3:
                        continue
                    strand = p[5] if len(p) >= 6 and p[5] in ("+", "-") else "+"
                    fout.write(f"{p[0]}\t{p[1]}\t{p[2]}\t.\t0\t{strand}\n")
    sorted_path = os.path.join(out_dir, "_atlas_sorted.bed")
    subprocess.run(["sort", "-k1,1", "-k2,2n", cat_path],
                   stdout=open(sorted_path, "w"), check=True)
    union_path = os.path.join(out_dir, "atlas_union.bed")
    raw = subprocess.run(
        ["bedtools", "merge", "-s", "-c", "6", "-o", "distinct",
         "-i", sorted_path],
        capture_output=True, text=True, check=True).stdout
    with open(union_path, "w") as fout:
        for line in raw.strip().split("\n"):
            p = line.split("\t")
            if len(p) < 4:
                continue
            fout.write(f"{p[0]}\t{p[1]}\t{p[2]}\t.\t0\t{p[3]}\n")
    os.remove(cat_path); os.remove(sorted_path)
    n = sum(1 for _ in open(union_path))
    print(f"  -> {union_path} ({n:,} intervals)")
    return union_path


def atlas_filter(rep_bed_path, atlas_union_path, label, out_dir):
    out_path = os.path.join(out_dir, f"_{label}_no_atlas.bed")
    subprocess.run(
        ["bedtools", "intersect", "-v", "-s",
         "-a", rep_bed_path, "-b", atlas_union_path],
        stdout=open(out_path, "w"), check=True,
    )
    n_in  = sum(1 for _ in open(rep_bed_path))
    n_out = sum(1 for _ in open(out_path))
    print(f"  {label}: {n_in:,} peaks -> {n_out:,} after atlas filter "
          f"({100*(1 - n_out/max(n_in,1)):.1f}% removed)")
    return out_path


def compute_intersections(rf_path, lr_path, cnn_path, out_dir,
                           overlap_dist=250):
    print(f"\n[intersections] computing 3-way overlap (tol={overlap_dist} nt)...")

    bf = {}
    for m, p in [("rf", rf_path), ("lr", lr_path), ("cnn", cnn_path)]:
        bf[m] = read_bed(p)
        bf[m]["source"] = m
        print(f"  {m}: {len(bf[m]):,} atlas-filtered peaks")

    slopped = {}
    for m in ("rf", "lr", "cnn"):
        df = bf[m].copy()
        df["start"] = (df["start"] - overlap_dist).clip(lower=0)
        df["end"]   = df["end"]   + overlap_dist
        slopped[m] = df

    presence = {}
    for m in ("rf", "lr", "cnn"):
        a_path = temp_bed(bf[m])
        flags = {}
        for n in ("rf", "lr", "cnn"):
            if n == m:
                continue
            b_path = temp_bed(slopped[n])
            res = subprocess.run(
                ["bedtools", "intersect", "-c", "-s",
                 "-a", a_path, "-b", b_path],
                capture_output=True, text=True, check=True)
            counts = []
            for line in res.stdout.strip().split("\n"):
                if not line: continue
                counts.append(int(line.split("\t")[-1]) > 0)
            flags[n] = np.array(counts)
            os.remove(b_path)
        presence[m] = flags
        os.remove(a_path)
        print(f"  {m}: overlap with others computed")

    rf_in_lr  = presence["rf"]["lr"]
    rf_in_cnn = presence["rf"]["cnn"]
    hard_mask = rf_in_lr & rf_in_cnn
    hard_neg_df = bf["rf"][hard_mask].copy()
    hard_neg_df["site_class"]   = "hard_neg"
    hard_neg_df["source_model"] = "all3"
    print(f"  HARD negatives (in all 3 models): {len(hard_neg_df):,}")

    med_dfs = []
    for m in ("rf", "lr", "cnn"):
        others = [x for x in ("rf","lr","cnn") if x != m]
        not_in_others = ~presence[m][others[0]] & ~presence[m][others[1]]
        d = bf[m][not_in_others].copy()
        d["site_class"]   = "medium_neg"
        d["source_model"] = m
        med_dfs.append(d)
        print(f"  MEDIUM (only in {m}): {len(d):,}")
    medium_neg_df = pd.concat(med_dfs, ignore_index=True)
    print(f"  MEDIUM total: {len(medium_neg_df):,}")

    hard_path   = os.path.join(out_dir, "hard_neg_all3.bed")
    medium_path = os.path.join(out_dir, "medium_neg_1model.bed")
    write_bed(hard_neg_df,   hard_path)
    write_bed(medium_neg_df, medium_path)
    print(f"  -> {hard_path}")
    print(f"  -> {medium_path}")
    return hard_neg_df, medium_neg_df


def sample_easy_negatives(pc_genes_bed, atlas_union_path,
                           canonical_bed, n_total, out_dir, seed=42,
                           window_len=500, min_dist_to_atlas=500):
    print(f"\n[easy_neg] sampling {n_total:,} random in-gene windows...")
    rng = np.random.default_rng(seed)
    pc = read_bed(pc_genes_bed)
    pc["len"] = pc["end"] - pc["start"]
    pc = pc[pc["len"] > window_len + 2 * min_dist_to_atlas]

    weights = pc["len"].values.astype(np.float64)
    weights /= weights.sum()
    n_sample = n_total * 5
    gene_idx = rng.choice(len(pc), size=n_sample, replace=True, p=weights)

    rows = []
    for i, gi in enumerate(gene_idx):
        g = pc.iloc[gi]
        max_start = g["end"] - window_len
        s = int(rng.integers(g["start"], max_start))
        rows.append({
            "chrom":  g["chrom"],
            "start":  s,
            "end":    s + window_len,
            "name":   f"{g['name']}_easyneg_{i}",
            "score":  0.0,
            "strand": g["strand"],
        })
    cand = pd.DataFrame(rows)
    print(f"  drew {len(cand):,} candidates (5x oversample)")

    slop_atlas = os.path.join(out_dir, "_atlas_slop.bed")
    rows_atlas = []
    with open(atlas_union_path) as fin:
        for line in fin:
            p = line.rstrip("\n").split("\t")
            rows_atlas.append({
                "chrom":  p[0],
                "start":  max(0, int(p[1]) - min_dist_to_atlas),
                "end":    int(p[2]) + min_dist_to_atlas,
                "name":   ".",
                "score":  0,
                "strand": p[5],
            })
    pd.DataFrame(rows_atlas).to_csv(slop_atlas, sep="\t",
                                     header=False, index=False)
    cand_bed = temp_bed(cand)
    res = subprocess.run(
        ["bedtools", "intersect", "-v", "-s",
         "-a", cand_bed, "-b", slop_atlas],
        capture_output=True, text=True, check=True)
    os.remove(cand_bed); os.remove(slop_atlas)
    keep = []
    for line in res.stdout.strip().split("\n"):
        if not line: continue
        p = line.split("\t")
        keep.append({"chrom":p[0], "start":int(p[1]), "end":int(p[2]),
                     "name":p[3], "score":0.0, "strand":p[5]})
    cand_keep = pd.DataFrame(keep)
    print(f"  after atlas filter: {len(cand_keep):,}")
    if len(cand_keep) > n_total:
        cand_keep = cand_keep.sample(n=n_total, random_state=seed).reset_index(drop=True)
    cand_keep["site_class"]   = "easy_neg"
    cand_keep["source_model"] = "random"
    print(f"  kept: {len(cand_keep):,}")
    return cand_keep


def split_by_chrom(df, train_chr, val_chr, test_chr):
    train = df[df["chrom"].isin(train_chr)].copy()
    val   = df[df["chrom"].isin(val_chr)].copy()
    test  = df[df["chrom"].isin(test_chr)].copy()
    return train, val, test


def assemble_and_split(constitutive_df, hard_df, medium_df, easy_df,
                        train_chr, val_chr, test_chr,
                        out_dir, cap_table=None):
    print(f"\n[assemble] splitting by chromosome...")
    splits = {"train": {}, "val": {}, "test": {}}
    for label, df in [("constitutive", constitutive_df),
                       ("hard_neg",    hard_df),
                       ("medium_neg",  medium_df),
                       ("easy_neg",    easy_df)]:
        tr, va, te = split_by_chrom(df, train_chr, val_chr, test_chr)
        for split, sdf in [("train", tr), ("val", va), ("test", te)]:
            cap = (cap_table or {}).get(split, {}).get(label)
            if cap is not None and len(sdf) > cap:
                sdf = sdf.sample(n=cap, random_state=42).reset_index(drop=True)
            splits[split][label] = sdf
            print(f"  {split:5s}  {label:13s}  {len(sdf):>7,}")

    manifest = {"splits": {}, "site_classes":
                ["constitutive","hard_neg","medium_neg","easy_neg"]}
    for split in ("train", "val", "test"):
        combined = pd.concat(list(splits[split].values()),
                              ignore_index=True)
        if len(combined) == 0:
            print(f"  WARN: {split} is empty"); continue
        combined["gene"] = (combined["name"].str.split("::").str[0]
                            .str.replace(r"_(cryptic|easyneg|peak|rep).*$",
                                          "", regex=True))
        bed_path = os.path.join(out_dir, f"sites.{split}.bed")
        tsv_path = os.path.join(out_dir, f"sites.{split}.tsv")
        write_bed(combined, bed_path)
        combined[["name", "site_class", "gene", "source_model"]].to_csv(
            tsv_path, sep="\t", index=False)
        counts = combined["site_class"].value_counts().to_dict()
        n_pos = int(counts.get("constitutive", 0))
        n_neg = int(len(combined) - n_pos)
        manifest["splits"][split] = {
            "n_total":    int(len(combined)),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "ratio":      f"1:{n_neg/max(n_pos,1):.2f}",
            "n_by_class": {k: int(v) for k, v in counts.items()},
            "bed_path":   bed_path,
            "tsv_path":   tsv_path,
        }
        print(f"  {split}: total={len(combined):,}  pos={n_pos:,}  neg={n_neg:,}  "
              f"ratio=1:{n_neg/max(n_pos,1):.2f}")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] manifest -> {os.path.join(out_dir, 'manifest.json')}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical_bed", required=True)
    ap.add_argument("--pc_genes_bed",  required=True)
    ap.add_argument("--atlas_beds",    required=True, nargs="+")
    ap.add_argument("--rep_bed_rf",    required=True)
    ap.add_argument("--rep_bed_lr",    required=True)
    ap.add_argument("--rep_bed_cnn",   required=True)
    ap.add_argument("--train_chroms",  required=True)
    ap.add_argument("--val_chroms",    required=True)
    ap.add_argument("--test_chroms",   required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--overlap_dist",  type=int, default=250,
                    help="Tolerance for 3-way model intersection (nt)")
    ap.add_argument("--n_easy_neg_per_split", type=int, default=8000,
                    help="Cap on easy negatives PER SPLIT (train/val/test)")
    ap.add_argument("--train_cap_constitutive", type=int, default=10000)
    ap.add_argument("--train_cap_hard_neg",     type=int, default=20000)
    ap.add_argument("--train_cap_medium_neg",   type=int, default=10000)
    ap.add_argument("--train_cap_easy_neg",     type=int, default=2500)
    ap.add_argument("--val_cap_constitutive",   type=int, default=None)
    ap.add_argument("--val_cap_hard_neg",       type=int, default=5000)
    ap.add_argument("--val_cap_medium_neg",     type=int, default=5000)
    ap.add_argument("--val_cap_easy_neg",       type=int, default=1250)
    ap.add_argument("--test_cap_constitutive",  type=int, default=None)
    ap.add_argument("--test_cap_hard_neg",      type=int, default=10000)
    ap.add_argument("--test_cap_medium_neg",    type=int, default=10000)
    ap.add_argument("--test_cap_easy_neg",      type=int, default=2500)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_chr = read_chroms(args.train_chroms)
    val_chr   = read_chroms(args.val_chroms)
    test_chr  = read_chroms(args.test_chroms)
    print(f"chroms: train={sorted(train_chr)}")
    print(f"        val  ={sorted(val_chr)}")
    print(f"        test ={sorted(test_chr)}")

    # a: atlas union
    atlas_union = build_atlas_union(args.atlas_beds, args.out_dir)

    # b: atlas-filter each cryptic bed
    print(f"\n[filter] atlas-filtering cryptic peaks...")
    rf_clean  = atlas_filter(args.rep_bed_rf,  atlas_union, "rf",  args.out_dir)
    lr_clean  = atlas_filter(args.rep_bed_lr,  atlas_union, "lr",  args.out_dir)
    cnn_clean = atlas_filter(args.rep_bed_cnn, atlas_union, "cnn", args.out_dir)

    # c: intersections -> hard + medium negatives
    hard_df, medium_df = compute_intersections(
        rf_clean, lr_clean, cnn_clean, args.out_dir,
        overlap_dist=args.overlap_dist)

    # d: easy negatives (3x because we'll split across train/val/test)
    easy_df = sample_easy_negatives(
        args.pc_genes_bed, atlas_union, args.canonical_bed,
        n_total=args.n_easy_neg_per_split * 5,
        out_dir=args.out_dir)

    const_df = read_bed(args.canonical_bed)
    const_df["site_class"]   = "constitutive"
    const_df["source_model"] = "canonical"
    print(f"\n[positives] constitutive sites: {len(const_df):,}")

    cap_table = {
        "train": {"constitutive": args.train_cap_constitutive,
                   "hard_neg":    args.train_cap_hard_neg,
                   "medium_neg":  args.train_cap_medium_neg,
                   "easy_neg":    args.train_cap_easy_neg},
        "val":   {"constitutive": args.val_cap_constitutive,
                   "hard_neg":    args.val_cap_hard_neg,
                   "medium_neg":  args.val_cap_medium_neg,
                   "easy_neg":    args.val_cap_easy_neg},
        "test":  {"constitutive": args.test_cap_constitutive,
                   "hard_neg":    args.test_cap_hard_neg,
                   "medium_neg":  args.test_cap_medium_neg,
                   "easy_neg":    args.test_cap_easy_neg},
    }
    print(f"\n[caps] per-class per-split caps:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {cap_table[split]}")
    manifest = assemble_and_split(
        const_df, hard_df, medium_df, easy_df,
        train_chr, val_chr, test_chr, args.out_dir,
        cap_table=cap_table)

    for p in [rf_clean, lr_clean, cnn_clean]:
        if os.path.exists(p):
            os.remove(p)

    print(f"\nNext: run extract_features_all.py to compute teacher features.")


if __name__ == "__main__":
    main()