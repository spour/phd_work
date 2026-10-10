#!/usr/bin/env python3
# splicing arbitrates only when the rf is ambiguous
import argparse, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def slogit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def rf_top2(g):
    pr = np.sort(g["rf_prob"].to_numpy())[::-1]
    return (float(pr[0]), float(pr[1]) if len(pr) > 1 else 0.0)


def is_contested(g, hi, margin):
    top, run = rf_top2(g)
    return (run >= hi) and (top - run <= margin)


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
    X = np.array(diffs); X2 = np.vstack([X, -X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    return LogisticRegression(fit_intercept=False, max_iter=3000).fit(X2, y2).coef_[0]


def gated_scores(cand, w, hi, margin):
    touched = 0; pieces = []
    for _, g in cand.groupby("gene", sort=False):
        if is_contested(g, hi, margin):
            s = w[0] * slogit(g["rf_prob"]) + w[1] * slogit(g["splice"])
            touched += 1
        else:
            s = slogit(g["rf_prob"])
        pieces.append(pd.Series(s, index=g.index))
    return pd.concat(pieces).reindex(cand.index).to_numpy(), touched


def metrics(cand, col):
    n_total = easy = hard = easy_ret = hard_resc = hits = 0
    for _, g in cand.groupby("gene", sort=False):
        n_total += 1
        if g["in_zone"].sum() == 0:
            continue
        base_first = g.iloc[int(np.argmax(g["rf_prob"].to_numpy()))]["in_zone"] == 1
        score_first = g.iloc[int(np.argmax(g[col].to_numpy()))]["in_zone"] == 1
        hits += int(score_first)
        if base_first:
            easy += 1; easy_ret += int(score_first)
        else:
            hard += 1; hard_resc += int(score_first)
    return dict(n_easy=easy, n_hard=hard,
                easy_kept=(easy_ret / easy if easy else np.nan),
                hard_rescued=(hard_resc / hard if hard else np.nan),
                net=(hits / n_total if n_total else np.nan))


def prec_at_recall(cand, col, t=0.7):
    y = cand["in_zone"].to_numpy(); s = cand[col].to_numpy()
    o = np.argsort(s)[::-1]; tp = np.cumsum(y[o]); fp = np.cumsum(1 - y[o]); P = y.sum()
    if P == 0:
        return np.nan
    rec = tp / P; k = min(int(np.searchsorted(rec, t)), len(rec) - 1)
    return float(tp[k] / (tp[k] + fp[k] + EPS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val_candidates", required=True)
    ap.add_argument("--test_candidates", required=True)
    args = ap.parse_args()
    cv = pd.read_csv(args.val_candidates, sep="\t")
    ct = pd.read_csv(args.test_candidates, sep="\t")

    w = fit_rank(cv)
    print(f"[gate] ranker coef rf={w[0]:+.3f} splice={w[1]:+.3f}", file=sys.stderr)

    MAX_CONTESTED_FRAC = 0.20
    n_val_genes = cv["gene"].nunique()
    hi_grid = np.round(np.linspace(0.3, 0.85, 12), 3)
    margin_grid = np.round(np.linspace(0.02, 0.20, 10), 3)
    best, best_obj, best_diag = (0.6, 0.05), -1e9, None
    for hi in hi_grid:
        for margin in margin_grid:
            rescued = broken = contested_n = 0
            for _, g in cv.groupby("gene", sort=False):
                if not is_contested(g, hi, margin):
                    continue
                if g["in_zone"].sum() == 0:
                    continue
                contested_n += 1
                base_first = g.iloc[int(np.argmax(g["rf_prob"].to_numpy()))]["in_zone"] == 1
                s = w[0] * slogit(g["rf_prob"]) + w[1] * slogit(g["splice"])
                gated_first = g.iloc[int(np.argmax(s.to_numpy() if hasattr(s, "to_numpy") else s))]["in_zone"] == 1
                if base_first and not gated_first:
                    broken += 1
                elif (not base_first) and gated_first:
                    rescued += 1
            if contested_n > MAX_CONTESTED_FRAC * n_val_genes or contested_n == 0:
                continue
            obj = rescued - broken
            if obj > best_obj:
                best_obj, best, best_diag = obj, (hi, margin), (contested_n, rescued, broken)
    hi, margin = best
    if best_diag:
        cn, rs, bk = best_diag
        print(f"[gate] chosen on val: runner-up>={hi}, top-runnerup<={margin}  "
              f"(val contested={cn}, rescued={rs}, broken={bk}, net={rs-bk:+d})", file=sys.stderr)
    else:
        print(f"[gate] WARNING: no threshold kept contested fraction under {MAX_CONTESTED_FRAC} "
              f"with any rescue; gate is off. splicing does not arbitrate cleanly on val.", file=sys.stderr)

    contested = sum(is_contested(g, hi, margin) for _, g in ct.groupby("gene", sort=False))

    ct["ungated"] = w[0] * slogit(ct["rf_prob"]) + w[1] * slogit(ct["splice"])
    gs, touched = gated_scores(ct, w, hi, margin)
    ct["gated"] = gs
    n_test_genes = ct["gene"].nunique()

    print(f"\n[gate] TEST ({n_test_genes} genes; {contested} are contested = two high, close RF "
          f"candidates; splicing arbitrated those, left {n_test_genes - touched} untouched):", file=sys.stderr)
    print(f"  {'scorer':<12s}{'n_easy':>7s}{'n_hard':>7s}{'easy_kept':>11s}{'hard_rescued':>14s}"
          f"{'net_P@1':>9s}{'prec@0.7':>10s}", file=sys.stderr)
    for nm, col in [("cpa_only", "rf_prob"), ("ungated", "ungated"), ("gated", "gated")]:
        m = metrics(ct, col); pr = prec_at_recall(ct, col)
        print(f"  {nm:<12s}{m['n_easy']:>7d}{m['n_hard']:>7d}{m['easy_kept']:>11.3f}"
              f"{m['hard_rescued']:>14.3f}{m['net']:>9.3f}{pr:>10.3f}", file=sys.stderr)

    cmask = ct.groupby("gene")["gene"].transform(lambda s: is_contested(ct.loc[s.index], hi, margin))
    cg = ct[cmask]
    if cg["gene"].nunique() > 0:
        rf_hits = comb_hits = ng = 0
        for _, g in cg.groupby("gene", sort=False):
            if g["in_zone"].sum() == 0:
                continue
            ng += 1
            rf_hits += int(g.iloc[int(np.argmax(g["rf_prob"].to_numpy()))]["in_zone"] == 1)
            s = w[0] * slogit(g["rf_prob"].to_numpy()) + w[1] * slogit(g["splice"].to_numpy())
            comb_hits += int(g.iloc[int(np.argmax(s))]["in_zone"] == 1)
        print(f"\n[gate] CONTESTED-ONLY ({ng} test genes with a true end among two close high candidates):",
              file=sys.stderr)
        print(f"    rf picks true end : {rf_hits}/{ng} = {rf_hits/ng:.3f}", file=sys.stderr)
        print(f"    rf+splice picks it: {comb_hits}/{ng} = {comb_hits/ng:.3f}", file=sys.stderr)
        print(f"    -> splicing changes {comb_hits - rf_hits:+d} contested calls", file=sys.stderr)
    print("\n[gate] this contested-only line is the claim. if rf+splice > rf here with a nonzero "
          "splice coef, splicing resolves competing high sites. if not, it does not, and you report "
          "the precision gain only.", file=sys.stderr)


if __name__ == "__main__":
    main()