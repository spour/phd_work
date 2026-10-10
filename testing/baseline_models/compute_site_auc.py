#!/usr/bin/env python3
# auroc/auprc for model scores on canonical/cryptic sites
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
)


KNOWN_PROB_COLS = {
    "rf":       "rf_prob",
    "lr":       "lr_prob",
    "cnn":      "cnn_prob",
    "aparent2": "aparent2_prob",
}


def read_bed(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            rows.append({
                "chrom":  p[0],
                "start":  int(p[1]),
                "end":    int(p[2]),
                "name":   p[3] if len(p) > 3 else f"{p[0]}:{p[1]}-{p[2]}",
                "strand": p[5] if len(p) > 5 else "+",
            })
    return rows


def build_pc_lookup(pc_genes_bed):
    pc = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    n_before = len(pc)
    pc = (pc.assign(span=pc["end"] - pc["start"])
            .sort_values("span", ascending=False)
            .drop_duplicates("gene", keep="first")
            .drop(columns=["span"]))
    if len(pc) < n_before:
        print(f"  deduped pc_genes: {n_before:,} -> {len(pc):,}")
    return pc.set_index("gene")[["chrom", "start", "end", "strand"]].to_dict("index")


def site_to_tx_offset(site_mid_genomic, gene_info):
    if gene_info["strand"] == "+":
        return site_mid_genomic - gene_info["start"]
    else:
        return gene_info["end"] - site_mid_genomic


def score_sites(sites_df, scan_dir, models, vicinity_nt, pc_lookup, label_str):
    parquets = {os.path.basename(p).replace(".parquet", ""): p
                for p in glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet"))}
    prob_cols = [KNOWN_PROB_COLS[m] for m in models]
    needed = prob_cols + ["window_offset"]

    n_no_parquet = 0
    n_no_pc = 0
    n_no_window = 0
    n_ok = 0
    rows = []
    for _, site in sites_df.iterrows():
        gene = site["gene"]
        pq = parquets.get(gene)
        if pq is None:
            n_no_parquet += 1
            continue
        gi = pc_lookup.get(gene)
        if gi is None:
            n_no_pc += 1
            continue
        try:
            df = pd.read_parquet(pq, columns=needed)
        except Exception:
            continue

        site_mid_genomic = (site["start"] + site["end"]) // 2
        site_tx = site_to_tx_offset(site_mid_genomic, gi)
        offsets = df["window_offset"].values
        win_centers = offsets + 250
        near = np.abs(win_centers - site_tx) <= vicinity_nt
        if not near.any():
            n_no_window += 1
            continue

        row = {"gene": gene, "site_mid": int(site_mid_genomic),
               "label": label_str}
        for m in models:
            col = KNOWN_PROB_COLS[m]
            row[m] = float(df[col].values[near].max())
        rows.append(row)
        n_ok += 1

    print(f"  scored {n_ok:,} sites  "
          f"(skipped: {n_no_parquet:,} no parquet, "
          f"{n_no_pc:,} not in pc_genes, "
          f"{n_no_window:,} no window in vicinity)")
    return pd.DataFrame(rows)


def metrics_for_pair(scores_pos, scores_neg, name, model):
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return None
    y_true = np.concatenate([np.ones(len(scores_pos)),
                              np.zeros(len(scores_neg))])
    y_score = np.concatenate([scores_pos, scores_neg])
    valid = ~np.isnan(y_score)
    y_true = y_true[valid]; y_score = y_score[valid]
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return None
    try:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
    except Exception as e:
        return {"name": name, "model": model, "error": str(e)}
    pos_rate = float((y_true == 1).mean())
    return {
        "name":     name,
        "model":    model,
        "n_pos":    int((y_true == 1).sum()),
        "n_neg":    int((y_true == 0).sum()),
        "pos_rate": pos_rate,
        "auroc":    auroc,
        "auprc":    auprc,
        "auprc_baseline": pos_rate,
        "auprc_lift":    auprc / pos_rate if pos_rate > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical_bed", required=True)
    ap.add_argument("--cryptic_bed",   required=True)
    ap.add_argument("--pc_genes_bed",  required=True)
    ap.add_argument("--scan_dir",      required=True)
    ap.add_argument("--models",        nargs="+",
                    default=["aparent2", "rf", "lr", "cnn"])
    ap.add_argument("--vicinity_nt",   type=int, default=500)
    ap.add_argument("--test_chroms",   default=None,
                    help="Optional file; if given, restrict all sites to "
                         "these chroms (one per line).")
    ap.add_argument("--n_random",      type=int, default=20000,
                    help="Number of random in-gene windows to sample for the "
                         "random-baseline negative set (default 20000).")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--out",           default=None,
                    help="Optional JSON output path with full metrics.")
    args = ap.parse_args()

    # validate models against parquet columns
    sample_pqs = sorted(glob.glob(os.path.join(args.scan_dir,
                                                  "per_gene", "*.parquet")))[:3]
    if not sample_pqs:
        sys.exit(f"No parquets under {args.scan_dir}/per_gene")
    import pyarrow.parquet as pq_mod
    cols_present = set(pq_mod.ParquetFile(sample_pqs[0]).schema_arrow.names)
    eff_models = [m for m in args.models if KNOWN_PROB_COLS[m] in cols_present]
    if eff_models != args.models:
        missing = [m for m in args.models if m not in eff_models]
        print(f"WARN: requested models {missing} have no prob column; "
              f"using {eff_models}")
    if not eff_models:
        sys.exit("No usable models")

    # load pc_genes lookup
    print(f"[Loading pc_genes lookup]")
    pc_lookup = build_pc_lookup(args.pc_genes_bed)
    print(f"  {len(pc_lookup):,} genes in pc_genes")

    # load sites
    test_chroms = None
    if args.test_chroms:
        test_chroms = set(open(args.test_chroms).read().split())
        print(f"restricting to test chroms: {sorted(test_chroms)}")
    canon = pd.DataFrame(read_bed(args.canonical_bed))
    canon["gene"] = canon["name"].str.split("::").str[0]
    cryp = pd.DataFrame(read_bed(args.cryptic_bed))
    cryp["gene"] = cryp["name"].str.split("::").str[0]
    if test_chroms is not None:
        canon = canon[canon["chrom"].isin(test_chroms)]
        cryp  = cryp[cryp["chrom"].isin(test_chroms)]
    print(f"canonical sites: {len(canon):,}")
    print(f"cryptic sites:   {len(cryp):,}")

    # score sites
    print(f"\n[Scoring canonical sites with max-in-+/-{args.vicinity_nt}-nt]")
    canon_scores = score_sites(canon, args.scan_dir, eff_models,
                                args.vicinity_nt, pc_lookup,
                                label_str="canonical")
    print(f"\n[Scoring cryptic sites with max-in-+/-{args.vicinity_nt}-nt]")
    cryp_scores = score_sites(cryp, args.scan_dir, eff_models,
                               args.vicinity_nt, pc_lookup,
                               label_str="cryptic")

    print(f"\n[Sampling random non-CPA windows for baseline comparisons]")
    rng = np.random.default_rng(args.seed)
    parquets = sorted(glob.glob(os.path.join(args.scan_dir,
                                                "per_gene", "*.parquet")))
    if test_chroms is not None:
        keep = []
        for p in parquets:
            gene = os.path.basename(p).replace(".parquet", "")
            gi = pc_lookup.get(gene)
            if gi and gi["chrom"] in test_chroms:
                keep.append(p)
        print(f"  parquets after test-chrom filter: "
              f"{len(parquets):,} -> {len(keep):,}")
        parquets = keep
    if not parquets:
        sys.exit("no parquets available for random sampling")
    sample_size = min(args.n_random, len(parquets) * 5)
    sampled = rng.choice(parquets, size=sample_size, replace=True)
    prob_cols = [KNOWN_PROB_COLS[m] for m in eff_models]
    rand_records = []
    for pq in sampled:
        try:
            pf = pd.read_parquet(pq, columns=prob_cols + ["is_constitutive_cpa"])
        except Exception:
            continue
        non_canon = pf[~pf["is_constitutive_cpa"]]
        if len(non_canon) == 0:
            continue
        r = non_canon.sample(n=1,
                              random_state=int(rng.integers(0, 2**31))).iloc[0]
        rec = {"label": "random"}
        for m in eff_models:
            rec[m] = float(r[KNOWN_PROB_COLS[m]])
        rand_records.append(rec)
    rand_df = pd.DataFrame(rand_records)
    print(f"  sampled {len(rand_df):,} random windows")

    print(f"\n[Computing metrics]")
    all_metrics = []
    print(f"{'Model':<10} {'Comparison':<28} {'n_pos':>8} {'n_neg':>8} "
          f"{'AUROC':>8} {'AUPRC':>8} {'base':>8} {'lift':>6}")
    print("-" * 90)
    for m in eff_models:
        if m not in canon_scores.columns or m not in cryp_scores.columns:
            print(f"  skip {m}: missing scored sites")
            continue
        canon_s = canon_scores[m].values.astype(float)
        cryp_s  = cryp_scores[m].values.astype(float)
        rand_s  = rand_df[m].values.astype(float) if m in rand_df.columns else np.array([])

        for name, pos, neg in [
            ("A. canonical vs cryptic", canon_s, cryp_s),
            ("B. canonical vs random",  canon_s, rand_s),
            ("C. cryptic   vs random",  cryp_s,  rand_s),
        ]:
            mres = metrics_for_pair(pos, neg, name, m)
            if mres is None:
                continue
            all_metrics.append(mres)
            print(f"{m:<10} {name:<28} {mres['n_pos']:>8,} {mres['n_neg']:>8,} "
                  f"{mres['auroc']:>8.4f} {mres['auprc']:>8.4f} "
                  f"{mres['auprc_baseline']:>8.4f} "
                  f"{mres['auprc_lift']:>6.2f}x")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                     exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "config": {
                    "canonical_bed":  args.canonical_bed,
                    "cryptic_bed":    args.cryptic_bed,
                    "pc_genes_bed":   args.pc_genes_bed,
                    "scan_dir":       args.scan_dir,
                    "models":         eff_models,
                    "vicinity_nt":    args.vicinity_nt,
                    "test_chroms":    sorted(list(test_chroms)) if test_chroms else None,
                    "n_random":       args.n_random,
                    "seed":           args.seed,
                },
                "metrics": all_metrics,
            }, f, indent=2)
        print(f"\nwrote {args.out}")

    print("""
Reading the table:
  AUROC    = area under ROC curve  (1.0 = perfect, 0.5 = random)
  AUPRC    = area under precision-recall curve  (1.0 = perfect)
  base     = AUPRC baseline = positive rate (random predictor's AUPRC)
  lift     = AUPRC / baseline (how many times better than random)

For comparison A (canonical vs cryptic), positive rate ~ n_canon/(n_canon+n_cryp).
  AUROC > 0.5 = model gives higher scores to canonicals on average.
  AUROC = 0.5 = no discrimination.
  AUROC < 0.5 = the model prefers cryptic to canonical (unusual).
""")


if __name__ == "__main__":
    main()