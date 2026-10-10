#!/usr/bin/env python3
# leak-free selector for the compact interpretable model
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-9
TRAIN_CHROMS = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
                "chr10", "chr11", "chr12", "chr14", "chr15", "chr16", "chr17"}
VAL_CHROMS = {"chr3", "chr13"}
TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}

POOL = {
    "upstream_max_me3__wg":  (+1, "upstream 3'SS / terminal-exon definition", False),
    "term_3ss__wg":          (+1, "PAS-proximal upstream 3'SS / terminal exon", False),
    "prox_u1_sum_t200__wg":  (-1, "proximal upstream U1 burden tau200 / telescripting", False),
}
GBT = dict(max_iter=800, max_depth=4, min_samples_leaf=50, learning_rate=0.05,
           class_weight="balanced", early_stopping=True, validation_fraction=0.15,
           n_iter_no_change=30, random_state=42)


def norm_chrom(c):
    c = str(c).strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def p_at_1(df, prob):
    d = df.copy(); d["p"] = prob
    hits = n = 0
    for _, g in d.groupby("gene", sort=False):
        y = g["y"].to_numpy().astype(int)
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        n += 1
        hits += int(y[int(np.argmax(g["p"].to_numpy()))] == 1)
    return (hits / n if n else np.nan)


def fit(X, y):
    return HistGradientBoostingClassifier(**GBT).fit(X, y)


def evalm(clf, X, d):
    p = clf.predict_proba(X)[:, 1]
    y = d["y"].to_numpy().astype(int)
    return average_precision_score(y, p), p_at_1(d, p), roc_auc_score(y, p)


def slope(y, x):
    ok = np.isfinite(x)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or x[ok].std() < EPS:
        return np.nan
    xs = (x[ok] - x[ok].mean()) / (x[ok].std() + EPS)
    return float(LogisticRegression(C=1e6, max_iter=2000).fit(xs.reshape(-1, 1), y[ok]).coef_[0, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--objective", choices=["p_at_1", "auprc"], default="p_at_1")
    ap.add_argument("--epsilon", type=float, default=0.004)
    ap.add_argument("--drop_compartment", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.features_tsv, sep="\t")
    df["c"] = df["chrom"].apply(norm_chrom)
    pool = [c for c in POOL if c in df.columns and not (args.drop_compartment and POOL[c][2])]
    missing = [c for c in POOL if c not in df.columns]
    if missing:
        print(f"[compact] not in file, skipped: {missing}", file=sys.stderr)

    tr = df["c"].isin(TRAIN_CHROMS)
    dva = df[df["c"].isin(VAL_CHROMS)].reset_index(drop=True)
    dte = df[df["c"].isin(TEST_CHROMS)].reset_index(drop=True)
    ytr = df.loc[tr, "y"].to_numpy().astype(int)

    print("\n[compact] sign filter (slope on train, __wg columns):", file=sys.stderr)
    kept = []
    for c in pool:
        exp = POOL[c][0]
        s = slope(ytr, df.loc[tr, c].to_numpy(float))
        ok = np.isfinite(s) and np.sign(s) == exp
        flag = " COMPARTMENT" if POOL[c][2] else ""
        print(f"  {c:<24s} expect {exp:+d}  slope {s:+.3f}  {'keep' if ok else 'DROP'}{flag}", file=sys.stderr)
        if ok:
            kept.append(c)
    if not kept:
        sys.exit("no sign-consistent __wg features found")

    obj_idx = 0 if args.objective == "auprc" else 1
    print(f"\n[compact] forward selection on val {args.objective}:", file=sys.stderr)
    chosen, remaining, best, path = [], list(kept), -1.0, []
    while remaining:
        scored = []
        for c in remaining:
            cols = chosen + [c]
            clf = fit(df.loc[tr, cols].to_numpy(), ytr)
            ap_v, p1_v, _ = evalm(clf, dva[cols].to_numpy(), dva)
            scored.append(((ap_v, p1_v)[obj_idx], ap_v, p1_v, c))
        scored.sort(reverse=True)
        score, ap_v, p1_v, c = scored[0]
        gain = score - best
        chosen.append(c); remaining.remove(c)
        path.append(dict(k=len(chosen), added=c, val_AUPRC=round(ap_v, 4), val_P1=round(p1_v, 4), gain=round(gain, 4)))
        print(f"  +{c:<24s} val_AUPRC={ap_v:.4f} val_P@1={p1_v:.4f} (gain {gain:+.4f})", file=sys.stderr)
        if gain < args.epsilon:
            break
        best = score
    rec_k = max(1, len(chosen) - 1) if path[-1]["gain"] < args.epsilon and len(chosen) > 1 else len(chosen)

    print("\n[compact] TEST metrics by number of features:", file=sys.stderr)
    print(f"  {'k':>2s} {'AUPRC':>7s} {'P@1':>7s} {'AUROC':>7s}  feature", file=sys.stderr)
    rows = []
    for k in range(1, len(chosen) + 1):
        cols = chosen[:k]
        clf = fit(df.loc[tr, cols].to_numpy(), ytr)
        ap_t, p1_t, au_t = evalm(clf, dte[cols].to_numpy(), dte)
        mark = " <-- recommended" if k == rec_k else ""
        print(f"  {k:>2d} {ap_t:>7.4f} {p1_t:>7.4f} {au_t:>7.4f}  {chosen[k-1]}{mark}", file=sys.stderr)
        rows.append(dict(k=k, feature=chosen[k-1], test_AUPRC=round(ap_t, 4),
                         test_P1=round(p1_t, 4), test_AUROC=round(au_t, 4), recommended=k == rec_k))

    bins = [c for c in df.columns if c.startswith("bin") and (c.endswith("_mean") or c.endswith("_max"))]
    allwg = [c for c in df.columns if c.endswith("__wg")]
    full = bins + allwg
    clf = fit(df.loc[tr, full].to_numpy(), ytr)
    ap_f, p1_f, au_f = evalm(clf, dte[full].to_numpy(), dte)
    print(f"\n[compact] ceiling, full {len(full)} (bins + all __wg): "
          f"AUPRC={ap_f:.4f} P@1={p1_f:.4f} AUROC={au_f:.4f}", file=sys.stderr)

    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_test_by_k.tsv", sep="\t", index=False)
    with open(f"{args.out_prefix}_final_features.txt", "w") as fh:
        fh.write("\n".join(chosen[:rec_k]) + "\n")
    r = next(x for x in rows if x["k"] == rec_k)
    print(f"\n[compact] RECOMMENDED ({args.objective}): {rec_k} features", file=sys.stderr)
    for c in chosen[:rec_k]:
        print(f"    {c:<24s} {POOL[c][1]}", file=sys.stderr)
    print(f"  test AUPRC={r['test_AUPRC']} P@1={r['test_P1']} AUROC={r['test_AUROC']}  "
          f"vs full {len(full)}: AUPRC={ap_f:.4f} P@1={p1_f:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()