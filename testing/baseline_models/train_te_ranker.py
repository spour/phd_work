#!/usr/bin/env python3
# lightgbm lambdarank reranker on the peak table
import argparse, json
import numpy as np, pandas as pd
import lightgbm as lgb

RF_TRAIN = {"chr1","chr2","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
            "chr11","chr12","chr14","chr15","chr16","chr17"}
RF_VAL  = {"chr3","chr13"}
RF_TEST = {"chr18","chr19","chr20","chr21","chr22","chrX"}

DEFAULT_FEATS = ["upstream_mean_u1__wg","upstream_max_me3__wg","term_3ss__wg","term_ppt__wg",
                 "local_me5__wg","pas_prox_u1__wg","dn5ss_has_strong__wg","interv_me5_max__wg",
                 "bp_strength__wg","u2af65_ppt__wg","u2af35_ag__wg","u1_suppression_down__wg",
                 "u1_duplex_down__wg","u2_duplex_bp__wg"]

def groups_and_labels(df, feats):
    df = df.sort_values("gene")
    sizes = df.groupby("gene", sort=True).size().to_numpy()
    X = df[feats].fillna(0.0).to_numpy()
    y = df["in_zone"].astype(int).to_numpy()
    return X, y, sizes, df

def contested_recovery(df, score_col):
    rec_s = rec_r = n = 0
    for _, g in df.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            continue
        crf = float(g.rf_prob[iz == 1].max())
        if not (g.rf_prob[iz == 0] >= crf).any():
            continue
        s = g[score_col].to_numpy()
        rec_s += int(iz[int(np.argmax(s))] == 1)
        rec_r += iz.sum() / len(g)
        n += 1
    return (rec_s / n if n else float("nan")), (rec_r / n if n else float("nan")), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True, help="peak table WITH all 14 __wg columns")
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATS)
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--min_child", type=int, default=50)
    ap.add_argument("--out_prefix", default="te_ranker")
    a = ap.parse_args()

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr" + df["chrom"])
    miss = [f for f in a.features if f not in df.columns]
    if miss:
        raise SystemExit(f"[err] peak table missing __wg columns: {miss}\n"
                         f"run add_wg_features.py first.")
    tr = df[df.chrom.isin(RF_TRAIN)]; va = df[df.chrom.isin(RF_VAL)]; te = df[df.chrom.isin(RF_TEST)]
    print(f"[data] train {tr.gene.nunique():,}g  val {va.gene.nunique():,}g  test {te.gene.nunique():,}g")

    Xtr, ytr, gtr, _ = groups_and_labels(tr, a.features)
    Xva, yva, gva, _ = groups_and_labels(va, a.features)

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=a.n_estimators,
        num_leaves=a.num_leaves, learning_rate=a.lr, min_child_samples=a.min_child,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    ranker.fit(Xtr, ytr, group=gtr,
               eval_set=[(Xva, yva)], eval_group=[gva], eval_at=[1],
               callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)])

    df["rank_score"] = ranker.predict(df[a.features].fillna(0.0).to_numpy())

    print("\n[importances (gain)]")
    imp = sorted(zip(a.features, ranker.feature_importances_), key=lambda x: -x[1])
    for f, v in imp:
        print(f"  {f:<26}{v:>8d}")

    for name, sub in [("TEST", te), ("VAL", va)]:
        sub = df[df.chrom.isin(RF_TEST if name == "TEST" else RF_VAL)]
        rs, rr, n = contested_recovery(sub, "rank_score")
        print(f"\n[{name}] contested recovery: ranker={rs:.3f}  random={rr:.3f}  "
              f"delta={rs-rr:+.3f}  (n={n:,})  vs MaxEntScan logistic +0.051")

    out = df[["gene","chrom","position","in_zone","rf_prob","rank_score"]].copy()
    out.to_parquet(f"{a.out_prefix}_scored.parquet", index=False)
    ranker.booster_.save_model(f"{a.out_prefix}_model.txt")
    print(f"\n[out] {a.out_prefix}_scored.parquet, {a.out_prefix}_model.txt")
    print(f"      then: genomewide_splice_recovery.py --features rank_score --peak_table {a.out_prefix}_scored.parquet")

if __name__ == "__main__":
    main()