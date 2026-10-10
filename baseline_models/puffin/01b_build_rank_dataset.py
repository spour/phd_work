#!/usr/bin/env python3
# build a per-gene ranking dataset for puffincpa-rank
import argparse
import json
import os
import subprocess
import sys
import tempfile

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


def write_bed(df, path):
    df[["chrom","start","end","name","score","strand"]].to_csv(
        path, sep="\t", header=False, index=False)


def temp_bed(df):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False)
    df[["chrom","start","end","name","score","strand"]].to_csv(
        f.name, sep="\t", header=False, index=False)
    f.close()
    return f.name


def read_chroms(path):
    return set(open(path).read().split())


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


def pool_cryptics(rep_rf, rep_lr, rep_cnn, atlas_union, out_dir):
    print(f"\n[cryptics] pooling and labeling cryptic peaks...")
    dfs = {}
    for name, path in [("rf", rep_rf), ("lr", rep_lr), ("cnn", rep_cnn)]:
        d = read_bed(path)
        d["source_model"] = name
        ap = temp_bed(d)
        res = subprocess.run(
            ["bedtools", "intersect", "-v", "-s",
             "-a", ap, "-b", atlas_union],
            capture_output=True, text=True, check=True)
        os.remove(ap)
        if res.stdout.strip():
            rows = []
            for line in res.stdout.strip().split("\n"):
                p = line.split("\t")
                rows.append({
                    "chrom":  p[0], "start": int(p[1]), "end": int(p[2]),
                    "name":   p[3], "score": float(p[4]) if p[4] != "." else 0.0,
                    "strand": p[5], "source_model": name,
                })
            dfs[name] = pd.DataFrame(rows)
        else:
            dfs[name] = pd.DataFrame()
        print(f"  {name}: {len(dfs[name]):,} peaks after atlas filter")

    overlap_dist = 250
    slopped = {}
    for m, d in dfs.items():
        s = d.copy()
        s["start"] = (s["start"] - overlap_dist).clip(lower=0)
        s["end"]   = s["end"]   + overlap_dist
        slopped[m] = s

    presence = {}
    for m in ("rf", "lr", "cnn"):
        if len(dfs[m]) == 0:
            continue
        a_path = temp_bed(dfs[m])
        flags = {}
        for n in ("rf", "lr", "cnn"):
            if n == m or len(slopped[n]) == 0:
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

    hard_dfs = []
    medium_dfs = []
    for m in ("rf", "lr", "cnn"):
        if m not in presence:
            continue
        others = [x for x in ("rf","lr","cnn") if x != m and x in presence[m]]
        if len(others) < 2:
            continue
        in_other1 = presence[m][others[0]]
        in_other2 = presence[m][others[1]]
        hard_mask = in_other1 & in_other2
        only_mask = ~in_other1 & ~in_other2
        h = dfs[m][hard_mask].copy()
        h["site_class"] = "hard_neg"
        hard_dfs.append(h)
        o = dfs[m][only_mask].copy()
        o["site_class"] = "medium_neg"
        medium_dfs.append(o)

    hard_df = pd.concat(hard_dfs, ignore_index=True) if hard_dfs else pd.DataFrame()
    if len(hard_df) > 0:
        hp = temp_bed(hard_df)
        subprocess.run(["sort", "-k1,1", "-k2,2n", "-o", hp, hp], check=True)
        res = subprocess.run(
            ["bedtools", "merge", "-s", "-c", "4,5,6", "-o",
             "first,max,distinct", "-d", "100", "-i", hp],
            capture_output=True, text=True, check=True)
        os.remove(hp)
        rows = []
        for line in res.stdout.strip().split("\n"):
            p = line.split("\t")
            rows.append({
                "chrom":  p[0], "start": int(p[1]), "end": int(p[2]),
                "name":   p[3], "score": float(p[4]) if p[4] != "." else 0.0,
                "strand": p[5],
                "source_model": "all3",
                "site_class":   "hard_neg",
            })
        hard_df = pd.DataFrame(rows)

    medium_df = pd.concat(medium_dfs, ignore_index=True) if medium_dfs else pd.DataFrame()
    print(f"  hard_neg (in all 3 models, deduped): {len(hard_df):,}")
    print(f"  medium_neg (only 1 model):           {len(medium_df):,}")
    return pd.concat([hard_df, medium_df], ignore_index=True)


def assign_cryptics_to_genes(cryptic_df, pc_genes_bed, out_dir):
    print(f"\n[assign] mapping cryptics to genes (strand-aware)...")
    pc = read_bed(pc_genes_bed)
    pc_path = temp_bed(pc)
    cp_path = temp_bed(cryptic_df)
    res = subprocess.run(
        ["bedtools", "intersect", "-s", "-wa", "-wb",
         "-a", cp_path, "-b", pc_path],
        capture_output=True, text=True, check=True)
    os.remove(pc_path); os.remove(cp_path)

    if not res.stdout.strip():
        cryptic_df["gene"] = None
        return cryptic_df

    rows = []
    for line in res.stdout.strip().split("\n"):
        p = line.split("\t")
        rows.append({
            "chrom":        p[0],
            "start":        int(p[1]),
            "end":          int(p[2]),
            "name":         p[3],
            "score":        float(p[4]) if p[4] != "." else 0.0,
            "strand":       p[5],
            "gene":         p[9],
        })
    crypt_in_gene = pd.DataFrame(rows)
    crypt_in_gene = crypt_in_gene.drop_duplicates(
        subset=["chrom","start","end","name"], keep="first")
    cryptic_df = cryptic_df.merge(
        crypt_in_gene[["chrom","start","end","name","gene"]],
        on=["chrom","start","end","name"], how="left")
    n_assigned = int(cryptic_df["gene"].notna().sum())
    print(f"  cryptics assigned to a gene: {n_assigned:,} / {len(cryptic_df):,} "
          f"({100*n_assigned/max(len(cryptic_df),1):.1f}%)")
    return cryptic_df


def build_gene_groups(canonical_df, cryptic_df, min_negs, max_negs,
                      pad_pool, seed=42):
    print(f"\n[groups] building gene groups (min_negs={min_negs}, "
          f"max_negs={max_negs}, pad pool size={len(pad_pool):,})...")
    rng = np.random.default_rng(seed)

    canonical_df = canonical_df.copy()
    canonical_df["gene"] = canonical_df["name"].str.replace(
        r"_canon.*$", "", regex=True)
    canonical_df["label"]         = 1
    canonical_df["site_class"]    = "constitutive"
    canonical_df["source_origin"] = "in_gene"
    canonical_df["source_model"]  = "canonical"

    cryptic_df = cryptic_df[cryptic_df["gene"].notna()].copy()
    cryptic_df["label"]         = 0
    cryptic_df["source_origin"] = "in_gene"
    in_gene_by_gene = dict(tuple(cryptic_df.groupby("gene")))

    pad_pool_df = pad_pool.copy()
    pad_pool_df["label"]         = 0
    pad_pool_df["source_origin"] = "external"

    group_rows = []
    n_genes_skipped_no_canon  = 0
    n_genes_with_padding      = 0
    n_genes_capped            = 0
    n_padded_rows             = 0
    for _, canon_row in canonical_df.iterrows():
        gene = canon_row["gene"]
        group = [canon_row.to_dict()]
        in_gene = in_gene_by_gene.get(gene, pd.DataFrame())
        n_in = len(in_gene)
        if n_in > 0:
            negs = in_gene.to_dict("records")
        else:
            negs = []
        n_have = len(negs)
        if n_have < min_negs and len(pad_pool_df) > 0:
            n_need = min_negs - n_have
            pad = pad_pool_df.sample(
                n=min(n_need, len(pad_pool_df)),
                random_state=int(rng.integers(0, 2**31))
            ).to_dict("records")
            for p in pad:
                p["gene"] = gene
            negs.extend(pad)
            n_padded_rows += len(pad)
            n_genes_with_padding += 1
        if len(negs) > max_negs:
            negs = list(rng.choice(negs, size=max_negs, replace=False))
            n_genes_capped += 1
        group.extend(negs)
        group_rows.extend(group)

    out = pd.DataFrame(group_rows)
    cols = ["chrom","start","end","name","score","strand",
            "gene","label","site_class","source_origin","source_model"]
    out = out[[c for c in cols if c in out.columns]]
    print(f"  total rows: {len(out):,}")
    print(f"  genes with padding (in-gene < min): {n_genes_with_padding:,}")
    print(f"  genes capped (in-gene > max):       {n_genes_capped:,}")
    print(f"  padding rows added:                 {n_padded_rows:,}")
    return out


def split_by_chrom(groups_df, train_chr, val_chr, test_chr):
    print(f"\n[split] splitting by chromosome...")
    splits = {}
    for split, chrs in [("train", train_chr), ("val", val_chr), ("test", test_chr)]:
        sub = groups_df[groups_df["chrom"].isin(chrs)].copy()
        splits[split] = sub
        n_genes = sub["gene"].nunique()
        n_rows  = len(sub)
        n_pos   = int((sub["label"] == 1).sum())
        n_neg   = int((sub["label"] == 0).sum())
        per_gene_counts = sub.groupby("gene").size()
        print(f"  {split}: {n_genes:,} genes, {n_rows:,} rows "
              f"(pos={n_pos:,}, neg={n_neg:,}), "
              f"per-gene candidates median={int(per_gene_counts.median())}, "
              f"max={int(per_gene_counts.max()) if len(per_gene_counts) else 0}")
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical_bed",     required=True)
    ap.add_argument("--pc_genes_bed",      required=True)
    ap.add_argument("--rep_bed_rf",        required=True)
    ap.add_argument("--rep_bed_lr",        required=True)
    ap.add_argument("--rep_bed_cnn",       required=True)
    ap.add_argument("--atlas_beds",        required=True, nargs="+")
    ap.add_argument("--train_chroms",      required=True)
    ap.add_argument("--val_chroms",        required=True)
    ap.add_argument("--test_chroms",       required=True)
    ap.add_argument("--out_dir",           required=True)
    ap.add_argument("--min_negs_per_gene", type=int, default=5)
    ap.add_argument("--max_negs_per_gene", type=int, default=50)
    ap.add_argument("--seed",              type=int, default=42)
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

    # b: pool cryptics (labeled hard/medium)
    cryptic_df = pool_cryptics(
        args.rep_bed_rf, args.rep_bed_lr, args.rep_bed_cnn,
        atlas_union, args.out_dir)

    # c: assign cryptics to protein-coding genes
    cryptic_df = assign_cryptics_to_genes(
        cryptic_df, args.pc_genes_bed, args.out_dir)

    in_gene_cryptics = cryptic_df[cryptic_df["gene"].notna()].copy()
    pad_pool = cryptic_df[cryptic_df["gene"].isna()].copy()
    pad_pool["gene"] = None
    if len(pad_pool) == 0:
        print(f"  pad pool is empty; using in-gene cryptics as fallback pad")
        pad_pool = in_gene_cryptics.copy()

    # d: build gene groups
    canonical_df = read_bed(args.canonical_bed)
    groups_df = build_gene_groups(
        canonical_df, in_gene_cryptics,
        min_negs=args.min_negs_per_gene,
        max_negs=args.max_negs_per_gene,
        pad_pool=pad_pool, seed=args.seed)

    splits = split_by_chrom(groups_df, train_chr, val_chr, test_chr)

    manifest = {"splits": {}, "config": {
        "min_negs_per_gene":  args.min_negs_per_gene,
        "max_negs_per_gene":  args.max_negs_per_gene,
    }}
    for split, sdf in splits.items():
        bed_path = os.path.join(args.out_dir, f"gene_groups.{split}.bed")
        tsv_path = os.path.join(args.out_dir, f"gene_groups.{split}.tsv")
        sdf["score_int"] = 0
        sdf_bed = sdf.rename(columns={"score_int": "score_bed"})
        sdf_bed[["chrom","start","end","name","score_bed","strand"]].to_csv(
            bed_path, sep="\t", header=False, index=False)
        sdf[["gene","name","label","site_class","source_origin",
             "source_model","chrom","start","end","strand"]].to_csv(
            tsv_path, sep="\t", index=False)
        n_genes = int(sdf["gene"].nunique())
        manifest["splits"][split] = {
            "n_genes":      n_genes,
            "n_rows":       int(len(sdf)),
            "n_positive":   int((sdf["label"] == 1).sum()),
            "n_negative":   int((sdf["label"] == 0).sum()),
            "bed_path":     bed_path,
            "tsv_path":     tsv_path,
        }
        print(f"  -> {tsv_path}")
        print(f"  -> {bed_path}")
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] manifest -> {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()