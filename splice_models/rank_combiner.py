#!/usr/bin/env python3
# evaluate rf+splice as a tiebreaker over rf candidates
import argparse, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def slogit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def overall_p_at_1(cand, col):
    hits = n = rec = 0
    for _, g in cand.groupby("gene", sort=False):
        n += 1
        if g["in_zone"].sum() == 0:
            continue
        rec += 1
        hits += int(g.iloc[int(np.argmax(g[col].to_numpy()))]["in_zone"] == 1)
    return (hits / n if n else np.nan), (rec / n if n else np.nan), n


def precision_at_recall(cand, col, target=0.7):
    y = cand["in_zone"].to_numpy(); s = cand[col].to_numpy()
    o = np.argsort(s)[::-1]; tp = np.cumsum(y[o]); fp = np.cumsum(1 - y[o]); P = y.sum()
    if P == 0:
        return np.nan
    rec = tp / P; k = min(int(np.searchsorted(rec, target)), len(rec) - 1)
    return float(tp[k] / (tp[k] + fp[k] + EPS))


def rescue(cand, base_col, score_col):
    n_total = easy = hard = easy_ret = hard_resc = 0
    for _, g in cand.groupby("gene", sort=False):
        n_total += 1
        if g["in_zone"].sum() == 0:
            continue
        base_first = g.iloc[int(np.argmax(g[base_col].to_numpy()))]["in_zone"] == 1
        score_first = g.iloc[int(np.argmax(g[score_col].to_numpy()))]["in_zone"] == 1
        if base_first:
            easy += 1; easy_ret += int(score_first)
        else:
            hard += 1; hard_resc += int(score_first)
    return dict(n_total=n_total, n_easy=easy, n_hard=hard,
                easy_retained=(easy_ret / easy if easy else np.nan),
                hard_rescued=(hard_resc / hard if hard else np.nan),
                net_p_at_1=((easy_ret + hard_resc) / n_total if n_total else np.nan))


def fit_rank(cv, max_neg=40, seed=0):
    rng = np.random.default_rng(seed); diffs = []
    for _, g in cv.groupby("gene", sort=False):
        pos = g[g["in_zone"] == 1]; neg = g[g["in_zone"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        if len(neg) > max_neg:
            neg = neg.iloc[rng.choice(len(neg), max_neg, replace=False)]
        fp = np.column_stack([slogit(pos["rf_prob"]), slogit(pos["splice"])])
        fn = np.column_stack([slogit(neg["rf_prob"]), slogit(neg["splice"])])
        for a in fp:
            for b in fn:
                diffs.append(a - b)
    if not diffs:
        sys.exit("no within-gene pos/neg pairs in val")
    X = np.array(diffs); X2 = np.vstack([X, -X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    return LogisticRegression(fit_intercept=False, max_iter=3000).fit(X2, y2).coef_[0], len(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val_candidates", required=True)
    ap.add_argument("--test_candidates", required=True)
    ap.add_argument("--max_neg", type=int, default=40)
    args = ap.parse_args()

    cv = pd.read_csv(args.val_candidates, sep="\t")
    ct = pd.read_csv(args.test_candidates, sep="\t")

    Xp = np.column_stack([slogit(cv["rf_prob"]), slogit(cv["splice"])])
    pooled = LogisticRegression(max_iter=1000).fit(Xp, cv["in_zone"])
    ct["pooled"] = pooled.predict_proba(np.column_stack([slogit(ct["rf_prob"]), slogit(ct["splice"])]))[:, 1]
    w, npairs = fit_rank(cv, args.max_neg)
    ct["rank"] = w[0] * slogit(ct["rf_prob"]) + w[1] * slogit(ct["splice"])
    print(f"[rank] pooled coef rf={pooled.coef_[0,0]:+.3f} splice={pooled.coef_[0,1]:+.3f}", file=sys.stderr)
    print(f"[rank] ranker coef rf={w[0]:+.3f} splice={w[1]:+.3f}  ({npairs} pairs)", file=sys.stderr)

    cpa_p1, recall, n = overall_p_at_1(ct, "rf_prob")
    sp_p1, _, _ = overall_p_at_1(ct, "splice")
    print(f"\n[rank] candidate recall (ceiling) = {recall:.4f} over {n} test genes", file=sys.stderr)
    print(f"[rank] cpa_only net P@1 = {cpa_p1:.4f}   (splice_only diagnostic only = {sp_p1:.4f})", file=sys.stderr)

    print("\n[rank] TIEBREAKER view (partition by whether RF already ranks the true end first):", file=sys.stderr)
    print(f"  {'combiner':<10s}{'n_easy':>7s}{'n_hard':>7s}{'easy_kept':>11s}{'hard_rescued':>14s}"
          f"{'net_P@1':>9s}{'prec@0.7':>10s}", file=sys.stderr)
    for nm, col in [("pooled", "pooled"), ("rank", "rank")]:
        r = rescue(ct, "rf_prob", col)
        prec = precision_at_recall(ct, col, 0.7)
        print(f"  {nm:<10s}{r['n_easy']:>7d}{r['n_hard']:>7d}{r['easy_retained']:>11.3f}"
              f"{r['hard_rescued']:>14.3f}{r['net_p_at_1']:>9.3f}{prec:>10.3f}", file=sys.stderr)
    print("\n[rank] read: hard_rescued is the refiner payoff; easy_kept near 1.0 means it doesn't "
          "break RF's existing wins. net_P@1 above cpa_only means the tiebreaker helps on balance.",
          file=sys.stderr)


if __name__ == "__main__":
    main()