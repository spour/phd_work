#!/usr/bin/env python3
# the definitive leak test for the lambdarank reranker
import argparse
import numpy as np, pandas as pd
import lightgbm as lgb

RF_TRAIN = {"chr1","chr2","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
            "chr11","chr12","chr14","chr15","chr16","chr17"}
RF_TEST = {"chr18","chr19","chr20","chr21","chr22","chrX"}
DEFAULT_FEATS = ["upstream_mean_u1__wg","upstream_max_me3__wg","term_3ss__wg","term_ppt__wg",
                 "local_me5__wg","pas_prox_u1__wg","dn5ss_has_strong__wg","interv_me5_max__wg"]

def contested_recovery(df, score_col):
    rs = rr = n = 0
    for _, g in df.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            continue
        crf = float(g.rf_prob[iz == 1].max())
        if not (g.rf_prob[iz == 0] >= crf).any():
            continue
        s = g[score_col].to_numpy()
        rs += int(iz[int(np.argmax(s))] == 1)
        rr += iz.sum() / len(g)
        n += 1
    return (rs / n if n else float("nan")), (rr / n if n else float("nan")), n

def shuffle_within_gene(df, seed):
    rng = np.random.default_rng(seed)
    n = len(df)
    r = rng.random(n)
    gcodes = pd.factorize(df["gene"].to_numpy())[0]
    order = np.lexsort((r, gcodes))
    sorted_g = gcodes[order]
    within_rank = np.empty(n, dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_g)) + 1]
    counts = np.diff(np.r_[boundaries, n])
    within_rank[order] = np.concatenate([np.arange(c) for c in counts])
    pos_per_gene = pd.Series(df["in_zone"].to_numpy()).groupby(gcodes).sum().to_numpy()
    npos_row = pos_per_gene[gcodes]
    new = (within_rank < npos_row).astype(int)
    out = df.copy(); out["in_zone"] = new
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min_child", type=int, default=50)
    ap.add_argument("--out_prefix", default="ranker_shuffle")
    a = ap.parse_args()

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr" + df["chrom"])

    print("[ref] real-label contested recovery is your prior run (~0.406 vs 0.251).\n")
    for seed in a.seeds:
        sh = shuffle_within_gene(df, seed)
        tr = sh[sh.chrom.isin(RF_TRAIN)].sort_values("gene")
        gtr = tr.groupby("gene", sort=True).size().to_numpy()
        Xtr = tr[a.features].fillna(0.0).to_numpy(); ytr = tr["in_zone"].astype(int).to_numpy()
        r = lgb.LGBMRanker(objective="lambdarank", n_estimators=a.n_estimators,
                           num_leaves=a.num_leaves, learning_rate=a.lr,
                           min_child_samples=a.min_child, subsample=0.8,
                           colsample_bytree=0.8, random_state=seed, n_jobs=-1)
        r.fit(Xtr, ytr, group=gtr,
              callbacks=[lgb.log_evaluation(50)])
        sh["rank_score"] = r.predict(sh[a.features].fillna(0.0).to_numpy())
        te = sh[sh.chrom.isin(RF_TEST)]
        rs, rr, n = contested_recovery(te, "rank_score")
        flag = "OK (near random)" if abs(rs - rr) < 0.03 else "LEAK SUSPECT"
        print(f"[seed {seed}] shuffled-label contested recovery = {rs:.3f}  "
              f"random = {rr:.3f}  delta = {rs-rr:+.3f}   {flag}")
    print("\nAll deltas near 0 => the real +0.154 is signal, not leakage.")
    print("Any delta >> 0 => features or eval leak the label; do not trust the real number.")

if __name__ == "__main__":
    main()