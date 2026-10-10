#!/usr/bin/env python3
# confirm the splicing model boosts 3'-end accuracy on rf candidates
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-6
TRAIN = {f"chr{c}" for c in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]}
VAL = {"chr3", "chr13"}
TEST = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}
FEATS = ["upstream_max_me3__wg", "prox_u1_sum_t200__wg"]
GBT = dict(max_iter=800, max_depth=4, min_samples_leaf=50, learning_rate=0.05,
           class_weight="balanced", early_stopping=True, validation_fraction=0.15,
           n_iter_no_change=30, random_state=42)
PROX_NT, PAD, U1_TAU = 1000, 12, 200


def norm(c):
    c = str(c).strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def slogit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def p_at_1_and_recall(cand):
    hits = n = recall_hits = 0
    for _, g in cand.groupby("gene", sort=False):
        if g["in_zone"].sum() == 0:
            n += 1
            continue
        n += 1
        recall_hits += 1
        top = g.iloc[int(np.argmax(g["score"].to_numpy()))]
        hits += int(top["in_zone"] == 1)
    return (hits / n if n else np.nan), (recall_hits / n if n else np.nan), n


def precision_at_recall(cand, recall_target=0.7):
    y = cand["in_zone"].to_numpy()
    s = cand["score"].to_numpy()
    order = np.argsort(s)[::-1]
    tp = np.cumsum(y[order]); fp = np.cumsum(1 - y[order])
    P = y.sum()
    if P == 0:
        return np.nan
    rec = tp / P
    k = np.searchsorted(rec, recall_target)
    k = min(k, len(rec) - 1)
    return float(tp[k] / (tp[k] + fp[k] + EPS))


# part b feature extraction (imports your tested extractor)
def burden_prox_u1(fa, u1_lookup, chrom, center, strand):
    if strand == "+":
        gs, ge = center - PROX_NT, center + PAD
    else:
        gs, ge = center - PAD, center + PROX_NT
    seq = fa.fetch(chrom, gs, ge, strand=strand)
    if seq is None or len(seq) < PROX_NT:
        return np.nan
    from extract_splicing_features import scan_u1
    u1 = scan_u1(seq, u1_lookup)
    if u1.size == 0:
        return np.nan
    p = np.arange(u1.size)
    up = p < PROX_NT
    d = (PROX_NT - p[up]).astype(float)
    vals = u1[up]
    fin = np.isfinite(vals)
    if not fin.any():
        return np.nan
    d, vals = d[fin], vals[fin]
    bg = np.percentile(vals, 10)
    excess = np.maximum(vals - bg, 0.0)
    return float(np.sum(excess * np.exp(-d / float(U1_TAU))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--rf_dir", required=True, help="per-gene RF parquets (window_start_genomic, rf_prob, is_constitutive_cpa)")
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--gene_baselines", required=True, help="gene_baselines.tsv (baseline_upstream_max_me3)")
    ap.add_argument("--burden_baseline", required=True, help="baseline_prox_u1_sum_t200.tsv (gene, mean)")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--tolerance", type=int, default=250)
    ap.add_argument("--min_peak_dist", type=int, default=250)
    ap.add_argument("--max_candidates", type=int, default=50)
    ap.add_argument("--center_offset", type=int, default=0)
    ap.add_argument("--limit_genes", type=int, default=0, help="cap genes per split for a fast first check")
    ap.add_argument("--upstream_nt", type=int, default=3000)
    ap.add_argument("--downstream_nt", type=int, default=1000)
    args = ap.parse_args()

    # part a: refit + confirm
    df = pd.read_csv(args.features_tsv, sep="\t")
    df["c"] = df["chrom"].apply(norm)
    tr = df["c"].isin(TRAIN)
    clf = HistGradientBoostingClassifier(**GBT).fit(df.loc[tr, FEATS].to_numpy(),
                                                    df.loc[tr, "y"].to_numpy().astype(int))
    print("[A] fresh refit, labeled-site performance:", file=sys.stderr)
    print(f"  {'split':<6s}{'AUPRC':>8s}{'P@1':>8s}{'AUROC':>8s}", file=sys.stderr)
    for nm, S in [("train", TRAIN), ("val", VAL), ("test", TEST)]:
        d = df[df["c"].isin(S)]
        p = clf.predict_proba(d[FEATS].to_numpy())[:, 1]
        y = d["y"].to_numpy().astype(int)
        dd = d.copy(); dd["p"] = p; hh = nn = 0
        for _, g in dd.groupby("gene", sort=False):
            yy = g["y"].to_numpy().astype(int)
            if yy.sum() and (yy == 0).sum():
                nn += 1; hh += int(yy[int(np.argmax(g["p"].to_numpy()))] == 1)
        print(f"  {nm:<6s}{average_precision_score(y,p):>8.4f}{(hh/nn if nn else np.nan):>8.4f}{roc_auc_score(y,p):>8.4f}",
              file=sys.stderr)

    # part b: joint scoring on rf candidates
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.pc_genes_bed)) + "/../scripts")
    sys.path.insert(0, "scripts")
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup,
                                           init_maxentscan, extract_one_site)
    fa = GenomeFasta(args.reference_fasta)
    u1 = load_u1_lookup(args.u1_lookup)
    me5 = load_me5_lookup(args.me5_lookup)
    _, score3, _, m3 = init_maxentscan()

    bl1 = pd.read_csv(args.gene_baselines, sep="\t").set_index("gene")["baseline_upstream_max_me3"]
    bl2 = pd.read_csv(args.burden_baseline, sep="\t").set_index("gene").iloc[:, 0]
    bed = pd.read_csv(args.pc_genes_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    bed["gene"] = bed["name"].astype(str).str.split("_").str[-1]
    meta = bed.drop_duplicates("gene").set_index("gene")[["chrom", "strand"]]

    def candidates_for(split_set, limit):
        rows, used = [], 0
        for p in sorted(glob.glob(os.path.join(args.rf_dir, "*.parquet"))):
            g = os.path.basename(p).replace(".parquet", "")
            if g not in meta.index:
                continue
            chrom = norm(meta.loc[g, "chrom"]); strand = meta.loc[g, "strand"]
            if chrom not in split_set or g not in bl1.index or g not in bl2.index:
                continue
            rf = pd.read_parquet(p, columns=["window_start_genomic", "rf_prob", "is_constitutive_cpa"])
            canon = rf.loc[rf["is_constitutive_cpa"].astype(bool), "window_start_genomic"].to_numpy()
            if canon.size == 0:
                continue
            pos = rf["window_start_genomic"].to_numpy(); pr = rf["rf_prob"].to_numpy()
            order = np.argsort(pr)[::-1]; keep, kp = [], []
            for i in order:
                if all(abs(pos[i] - q) > args.min_peak_dist for q in kp):
                    keep.append(i); kp.append(pos[i])
                if len(keep) >= args.max_candidates:
                    break
            for i in keep:
                c = int(pos[i]) + args.center_offset
                if strand == "+":
                    ws, we = c - args.upstream_nt, c + args.downstream_nt
                else:
                    ws, we = c - args.downstream_nt, c + args.upstream_nt
                seq = fa.fetch(chrom, ws, we, strand=strand)
                if seq is None:
                    continue
                feats = extract_one_site(seq, args.upstream_nt, args.downstream_nt, 8, me5, score3, m3, u1)
                me3_max = np.nanmax([feats.get(f"bin{b}_me3_max", np.nan) for b in range(1, 7)])
                burden = burden_prox_u1(fa, u1, chrom, c, strand)
                rows.append(dict(gene=g, pos=int(pos[i]), rf_prob=float(pr[i]),
                                 upstream_max_me3__wg=me3_max - float(bl1.loc[g]),
                                 prox_u1_sum_t200__wg=burden - float(bl2.loc[g]),
                                 in_zone=int(np.abs(canon - int(pos[i])).min() <= args.tolerance)))
            used += 1
            if limit and used >= limit:
                break
        return pd.DataFrame(rows)

    print("\n[B] extracting candidate features (val, then test)...", file=sys.stderr)
    cv = candidates_for(VAL, args.limit_genes)
    ct = candidates_for(TEST, args.limit_genes)
    if cv.empty or ct.empty:
        sys.exit("no candidates; check rf_dir, baselines coverage, and pc_genes_bed gene names")

    cv["splice"] = clf.predict_proba(cv[FEATS].to_numpy())[:, 1]
    ct["splice"] = clf.predict_proba(ct[FEATS].to_numpy())[:, 1]

    comb = LogisticRegression(max_iter=1000).fit(
        np.column_stack([slogit(cv["rf_prob"]), slogit(cv["splice"])]), cv["in_zone"])
    print(f"[B] combiner coef rf={comb.coef_[0,0]:+.3f} splice={comb.coef_[0,1]:+.3f}", file=sys.stderr)
    ct["combined"] = comb.predict_proba(np.column_stack([slogit(ct["rf_prob"]), slogit(ct["splice"])]))[:, 1]

    print("\n[B] TEST, RF-alone vs RF+splicing on candidates:", file=sys.stderr)
    print(f"  {'scorer':<12s}{'P@1':>8s}{'prec@0.7':>10s}", file=sys.stderr)
    for nm, col in [("cpa_only", "rf_prob"), ("splice_only", "splice"), ("combined", "combined")]:
        ct["score"] = ct[col]
        p1, recall, n = p_at_1_and_recall(ct)
        prec = precision_at_recall(ct, 0.7)
        print(f"  {nm:<12s}{p1:>8.4f}{prec:>10.4f}", file=sys.stderr)
    ct["score"] = ct["rf_prob"]
    _, recall, n = p_at_1_and_recall(ct)
    print(f"\n[B] candidate recall (ceiling on P@1) = {recall:.4f} over {n} test genes", file=sys.stderr)
    ct.to_csv(f"{args.out_prefix}_test_candidates.tsv", sep="\t", index=False)
    cv.to_csv(f"{args.out_prefix}_val_candidates.tsv", sep="\t", index=False)
    print(f"[B] wrote {args.out_prefix}_test_candidates.tsv and _val_candidates.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()