#!/usr/bin/env python3
# train both the 2-feature and 55-feature splicing gbts on a union(tier2
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from extract_splicing_features import (
    GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan,
    extract_one_site,
)
from add_proximal_u1_burden import (
    fetch_upstream, burden_features, gene_baseline as compute_burden_baseline,
    PROX_NT, TAUS,
)


EPS = 1e-9
TRAIN_CHROMS = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
                "chr10", "chr11", "chr12", "chr14", "chr15", "chr16", "chr17"}
VAL_CHROMS = {"chr3", "chr13"}
TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}


def normalize_chrom(c):
    c = str(c).strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def read_ccpcpa_bed(path):
    df = pd.read_csv(path, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    df["center"] = (df["start"] + df["end"]) // 2
    parts = df["name"].astype(str).str.split("_", expand=True)
    if parts.shape[1] < 5:
        raise ValueError(
            f"BED name field has fewer than 5 underscore-separated parts: "
            f"{df['name'].iloc[0]!r} in {path}")
    df["gene"] = parts[4]
    return df[["chrom", "center", "strand", "gene"]].copy()


def per_gene_metrics(df, prob):
    d = df.copy(); d["p"] = prob
    n_paired = 0
    sum_top1 = 0.0
    sum_rr = 0.0
    sum_wgauc = 0.0
    for _, g in d.groupby("gene", sort=False):
        y = g["y"].to_numpy().astype(int)
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        n_paired += 1
        p = g["p"].to_numpy()
        sum_top1 += int(y[int(np.argmax(p))] == 1)
        cs = float(np.max(p[y == 1]))
        rank = 1 + int((p[y == 0] > cs).sum())
        sum_rr += 1.0 / rank
        try:
            sum_wgauc += float(roc_auc_score(y, p))
        except ValueError:
            pass
    if n_paired == 0:
        return {"p_at_1": np.nan, "mrr": np.nan, "wg_auc": np.nan, "n_paired": 0}
    return {"p_at_1": sum_top1 / n_paired,
            "mrr": sum_rr / n_paired,
            "wg_auc": sum_wgauc / n_paired,
            "n_paired": n_paired}


def fit_eval(df_train, df_test, feature_cols, seed=42, label="model"):
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["y"].to_numpy().astype(int)
    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test["y"].to_numpy().astype(int)

    clf = HistGradientBoostingClassifier(
        max_iter=800, max_depth=4, min_samples_leaf=50, learning_rate=0.05,
        class_weight="balanced", early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=seed)
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    p_test = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, p_test)
    auprc = average_precision_score(y_test, p_test)
    gm = per_gene_metrics(df_test, p_test)

    return {
        "label": label,
        "n_features": len(feature_cols),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "p_at_1": gm["p_at_1"],
        "mrr": gm["mrr"],
        "within_gene_auc": gm["wg_auc"],
        "n_paired_genes": gm["n_paired"],
        "fit_time_s": fit_time,
        "clf": clf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier2_bed", required=True,
                     help="cpccpa_tier2_exactly_two_no_3utr.bed")
    ap.add_argument("--tier3_bed", required=True,
                     help="cpccpa_tier3_exactly_three_no_3utr.bed")
    ap.add_argument("--canonical_features_tsv", required=True,
                     help="features_with_burden.tsv — canonical positions taken "
                          "from where y=1; burden baselines reverse-engineered")
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--gene_baselines_tsv", required=True,
                     help="data/gene_baselines.tsv (for the 7 axes)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_baseline_samples_new", type=int, default=40,
                     help="Random baseline samples for NEW genes only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*70}")
    print(f"STEP 1: Build union training set")
    print(f"{'='*70}")

    print(f"[step1] reading canonical features TSV")
    existing = pd.read_csv(args.canonical_features_tsv, sep="\t")
    canonical = existing[existing["y"] == 1][
        ["gene", "chrom", "center", "strand"]].copy()
    canonical["y"] = 1
    canonical["source"] = "canonical_inherited"
    print(f"[step1]   canonical positions (y=1): {len(canonical):,}")

    print(f"[step1] reading tier2 and tier3 BED files")
    tier2 = read_ccpcpa_bed(args.tier2_bed)
    tier3 = read_ccpcpa_bed(args.tier3_bed)
    print(f"[step1]   tier2 (exactly 2-way): {len(tier2):,} sites")
    print(f"[step1]   tier3 (exactly 3-way): {len(tier3):,} sites")

    cryptic_union = pd.concat([tier2, tier3], ignore_index=True)
    cryptic_union = cryptic_union.drop_duplicates(
        subset=["chrom", "center", "strand"], keep="first")
    cryptic_union["y"] = 0
    cryptic_union["source"] = "ccpcpa_union_tier2_tier3"
    print(f"[step1]   union (cryptic, y=0): {len(cryptic_union):,} unique sites "
           f"({len(tier2) + len(tier3) - len(cryptic_union):,} duplicates removed)")

    all_positions = pd.concat([canonical, cryptic_union], ignore_index=True)
    dup_mask = all_positions.duplicated(subset=["chrom", "center"], keep=False)
    n_conflict = int(dup_mask.sum())
    if n_conflict > 0:
        print(f"[step1]   WARNING: {n_conflict} canonical-cryptic conflicts removed")
        all_positions = all_positions[~dup_mask].reset_index(drop=True)

    all_positions["chrom_norm"] = all_positions["chrom"].apply(normalize_chrom)
    print(f"[step1] final position table: {len(all_positions):,} positions, "
           f"{all_positions['gene'].nunique():,} genes")
    print(f"[step1]   y=1 (canonical): {int((all_positions['y']==1).sum()):,}")
    print(f"[step1]   y=0 (cryptic):   {int((all_positions['y']==0).sum()):,}")
    print(f"[step1]   class balance:   {float(all_positions['y'].mean()):.4f}")

    all_positions.to_csv(os.path.join(args.out_dir, "union_position_table.tsv"),
                          sep="\t", index=False)

    print(f"\n{'='*70}")
    print(f"STEP 2: Load and prepare per-gene baselines")
    print(f"{'='*70}")

    axes_baselines = pd.read_csv(args.gene_baselines_tsv, sep="\t")
    axes_baselines = axes_baselines.drop_duplicates(subset="gene", keep="first")
    baseline_cols_axes = [c for c in axes_baselines.columns
                            if c.startswith("baseline_")]
    print(f"[step2] axes baselines: {len(axes_baselines):,} genes, "
           f"{len(baseline_cols_axes)} axis baseline columns")

    print(f"[step2] reverse-engineering burden baselines from existing TSV")
    burden_raw = "prox_u1_sum_t200"
    burden_wg = "prox_u1_sum_t200__wg"
    if burden_raw not in existing.columns or burden_wg not in existing.columns:
        print(f"[step2] ERROR: existing TSV missing {burden_raw} or {burden_wg}; "
               f"cannot reverse-engineer baselines")
        return
    existing["__burden_baseline"] = existing[burden_raw] - existing[burden_wg]
    burden_baselines = (existing.dropna(subset=["__burden_baseline"])
                                .groupby("gene")["__burden_baseline"]
                                .first()
                                .to_dict())
    print(f"[step2]   reverse-engineered burden baselines: "
           f"{len(burden_baselines):,} genes")

    genes_needed = set(all_positions["gene"].unique())
    genes_with_burden_baseline = set(burden_baselines.keys())
    genes_missing_burden = genes_needed - genes_with_burden_baseline
    print(f"[step2]   genes needing fresh burden baselines: "
           f"{len(genes_missing_burden):,}")

    print(f"\n{'='*70}")
    print(f"STEP 3: Extract features at {len(all_positions):,} positions")
    print(f"{'='*70}")

    print(f"[step3] loading FASTA + lookups + MaxEntScan")
    fa = GenomeFasta(args.reference_fasta)
    u1_lookup = load_u1_lookup(args.u1_lookup)
    me5_lookup = load_me5_lookup(args.me5_lookup)
    _, score3, _, m3 = init_maxentscan()

    UP_NT, DN_NT, N_BINS = 3000, 1000, 8

    pc = pd.read_csv(args.pc_genes_bed, sep="\t", header=None,
                      names=["chrom", "start", "end", "name", "score", "strand"])
    pc["gene"] = pc["name"].astype(str)
    if pc["gene"].str.contains("_").any():
        pc["gene"] = pc["gene"].str.split("_").str[-1]
    pc = pc.drop_duplicates(subset="gene", keep="first").set_index("gene")

    if genes_missing_burden:
        print(f"[step3] computing fresh burden baselines for "
               f"{len(genes_missing_burden):,} new genes...")
        t0 = time.time()
        for i, g in enumerate(sorted(genes_missing_burden)):
            if g in pc.index:
                fresh = compute_burden_baseline(fa, u1_lookup, pc.loc[g],
                                                  args.n_baseline_samples_new, rng)
                if fresh and np.isfinite(fresh.get(burden_raw, np.nan)):
                    burden_baselines[g] = fresh[burden_raw]
            if (i + 1) % 500 == 0:
                print(f"[step3]   fresh baselines: {i+1:,}/{len(genes_missing_burden):,}"
                       f"  ({(time.time()-t0)/60:.1f}m)")

    print(f"[step3] extracting features at all positions...")
    t0 = time.time()
    feature_rows = []
    n_failed = 0
    for i, row in enumerate(all_positions.itertuples(index=False)):
        if row.strand == "+":
            win_start = int(row.center) - UP_NT
            win_end = int(row.center) + DN_NT
        else:
            win_start = int(row.center) - DN_NT
            win_end = int(row.center) + UP_NT
        seq = fa.fetch(row.chrom, win_start, win_end, strand=row.strand)
        if seq is None or len(seq) < (UP_NT + DN_NT - 100):
            n_failed += 1
            feature_rows.append({})
            continue
        try:
            site_feats = extract_one_site(seq, UP_NT, DN_NT, N_BINS,
                                            me5_lookup, score3, m3, u1_lookup)
        except Exception:
            n_failed += 1
            feature_rows.append({})
            continue
        upseq = fetch_upstream(fa, row.chrom, int(row.center), row.strand)
        bf = burden_features(upseq, u1_lookup)
        site_feats.update(bf)
        feature_rows.append(site_feats)
        if (i + 1) % args.log_every == 0:
            elapsed = (time.time() - t0) / 60
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            eta = (len(all_positions) - i - 1) / max(rate, 1e-9) / 60
            print(f"[step3]   {i+1:,}/{len(all_positions):,}  "
                   f"({elapsed:.1f}m elapsed, {eta:.1f}m eta, "
                   f"failed={n_failed:,})")

    feat_df = pd.DataFrame(feature_rows)
    feat_df = feat_df.reset_index(drop=True)
    all_positions = all_positions.reset_index(drop=True)
    df = pd.concat([all_positions, feat_df], axis=1)
    df = df.dropna(subset=["bin1_u1_mean"]).reset_index(drop=True)
    print(f"[step3] extracted features for {len(df):,} positions "
           f"(failed: {n_failed:,})")

    print(f"\n{'='*70}")
    print(f"STEP 4: Apply gene baselines to create __wg columns")
    print(f"{'='*70}")

    AXES = ["upstream_mean_u1", "upstream_max_me3", "term_3ss", "local_me5",
            "pas_prox_u1", "dn5ss_has_strong", "interv_me5_max"]
    baselines_axes_map = axes_baselines.set_index("gene")[
        baseline_cols_axes].to_dict("index")

    n_axes_centered = 0
    for ax in AXES:
        bcol = f"baseline_{ax}"
        if bcol not in baseline_cols_axes:
            print(f"[step4] WARNING: no baseline column {bcol}; skipping")
            df[f"{ax}__wg"] = np.nan
            continue
        df[f"{ax}__wg"] = df.apply(
            lambda r: r.get(ax, np.nan) - (baselines_axes_map.get(r["gene"], {})
                                              .get(bcol, np.nan)),
            axis=1)
        n_axes_centered += 1
    print(f"[step4] centered {n_axes_centered} axes via gene_baselines.tsv")

    df[f"{burden_raw}__wg"] = df[burden_raw] - df["gene"].map(burden_baselines)
    n_burden_centered = int(df[f"{burden_raw}__wg"].notna().sum())
    print(f"[step4] centered burden for {n_burden_centered:,}/{len(df):,} positions")

    out_tsv = os.path.join(args.out_dir, "union_features.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[step4] wrote {out_tsv}")

    print(f"\n{'='*70}")
    print(f"STEP 5: Train 55-feature and 2-feature models, benchmark on test")
    print(f"{'='*70}")

    df["split"] = "other"
    df.loc[df["chrom_norm"].isin(TRAIN_CHROMS), "split"] = "train"
    df.loc[df["chrom_norm"].isin(VAL_CHROMS), "split"] = "val"
    df.loc[df["chrom_norm"].isin(TEST_CHROMS), "split"] = "test"

    print(f"[step5] split sizes:")
    for s in ["train", "val", "test", "other"]:
        sub = df[df["split"] == s]
        n_genes = sub["gene"].nunique()
        pos_rate = float((sub["y"] == 1).mean()) if len(sub) > 0 else 0
        print(f"  {s:>5s}: {len(sub):>7,} sites, {n_genes:>6,} genes, "
               f"pos_rate={pos_rate:.4f}")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    bin_cols = sorted([c for c in df.columns
                          if c.startswith("bin") and
                          (c.endswith("_mean") or c.endswith("_max"))])
    axes_wg_cols = [f"{ax}__wg" for ax in AXES]
    features_55 = bin_cols + axes_wg_cols
    features_2 = [f"{burden_raw}__wg", "upstream_max_me3__wg"]
    print(f"\n[step5] feature sets:")
    print(f"  55-feature: {len(features_55)} features "
           f"(48 per-bin + {len(axes_wg_cols)} axes __wg)")
    print(f"  2-feature:  {len(features_2)} features ({features_2})")

    missing_55 = [f for f in features_55 if f not in df.columns]
    missing_2 = [f for f in features_2 if f not in df.columns]
    if missing_55:
        print(f"[step5] ERROR: 55-feature model missing: {missing_55}")
        return
    if missing_2:
        print(f"[step5] ERROR: 2-feature model missing: {missing_2}")
        return

    print(f"\n[step5] training 55-feature model...")
    r55 = fit_eval(df_train, df_test, features_55, args.seed,
                     label="55-feature (union training)")

    print(f"[step5] training 2-feature model...")
    r2 = fit_eval(df_train, df_test, features_2, args.seed,
                    label="2-feature (union training)")

    joblib.dump({"model": r55["clf"], "feature_cols": features_55,
                  "training_set": "union(tier2,tier3)+canonical"},
                 os.path.join(args.out_dir, "gbt_55feat_union.joblib"))
    joblib.dump({"model": r2["clf"], "feature_cols": features_2,
                  "training_set": "union(tier2,tier3)+canonical"},
                 os.path.join(args.out_dir, "gbt_2feat_union.joblib"))

    print(f"\n{'='*70}")
    print(f"TEST-SET COMPARISON (chr18-22, chrX)")
    print(f"{'='*70}")
    print(f"  {'model':<28s} {'AUROC':>7s} {'AUPRC':>7s} {'P@1':>7s} "
           f"{'MRR':>7s} {'WG-AUC':>7s} {'n_paired':>9s}")
    rows = []
    for r in [r55, r2]:
        print(f"  {r['label']:<28s} {r['auroc']:>7.4f} {r['auprc']:>7.4f} "
               f"{r['p_at_1']:>7.4f} {r['mrr']:>7.4f} "
               f"{r['within_gene_auc']:>7.4f} {r['n_paired_genes']:>9d}")
        rows.append({k: v for k, v in r.items() if k != "clf"})
    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "benchmark_metrics.tsv"),
        sep="\t", index=False)

    if rows[0]["auroc"] and rows[1]["auroc"]:
        d_auroc = rows[0]["auroc"] - rows[1]["auroc"]
        d_auprc = rows[0]["auprc"] - rows[1]["auprc"]
        d_p1 = rows[0]["p_at_1"] - rows[1]["p_at_1"]
        print(f"\n  Gap (55-feature - 2-feature):")
        print(f"    Δ AUROC = {d_auroc:+.4f}")
        print(f"    Δ AUPRC = {d_auprc:+.4f}")
        print(f"    Δ P@1   = {d_p1:+.4f}")

    print(f"\n  Reference: existing 3-way-only 55-feature model "
           f"(chrsplit/gbt_combined_chrsplit.joblib)")
    print(f"    AUROC=0.830, AUPRC=0.752, P@1=0.580 on 1,137 paired test genes")
    print(f"    [trained on smaller 3-way intersection cryptic set]")

    print(f"\n[step5] wrote benchmark_metrics.tsv, gbt_55feat_union.joblib, "
           f"gbt_2feat_union.joblib")


if __name__ == "__main__":
    main()