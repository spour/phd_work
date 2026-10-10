#!/usr/bin/env python3
# compact, signed, mechanism-aligned splice model
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

TRAIN_CHROMS = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
                "chr10", "chr11", "chr12", "chr14", "chr15", "chr16", "chr17"}
VAL_CHROMS = {"chr3", "chr13"}
TEST_CHROMS = {"chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}

COMPACT_FEATURES = {
    "pas_prox_u1":              (-1, "telescripting: U1 within 500nt upstream suppresses CPA"),
    "pas_prox_u1__wg":          (-1, "telescripting: upstream U1 above this gene's background"),
    "local_me5":                (-1, "telescripting: 5'SS donor within 500nt upstream"),
    "interv_me5_max_500":       (-1, "intervening upstream 5'SS = unspliced intron donor -> cryptic"),
    "interv_me5_max__wg":       (-1, "intervening upstream 5'SS above gene background -> cryptic"),
    "term_me3_top3mean_500_25": (+1, "strong upstream 3'SS defines terminal exon -> canonical"),
    "upstream_max_me3__wg":     (+1, "upstream 3'SS above gene background -> canonical"),
    "dn5ss_best_score":         (-1, "downstream 5'SS = gene continues, non-terminal (architecture)"),
}


def normalize_chrom(c):
    if not isinstance(c, str):
        return c
    c = c.strip()
    return "chr" + c[3:].upper() if c.lower().startswith("chr") else "chr" + c.upper()


def p_at_1(df, prob):
    d = df.copy()
    d["p"] = prob
    hits = n = 0
    for _, g in d.groupby("gene", sort=False):
        y = g["y"].to_numpy().astype(int)
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        n += 1
        hits += int(y[int(np.argmax(g["p"].to_numpy()))] == 1)
    return (hits / n if n else np.nan), n


def fit_native(X, y, seed=42):
    clf = HistGradientBoostingClassifier(
        max_iter=800, max_depth=4, min_samples_leaf=50, learning_rate=0.05,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        random_state=seed)
    return clf.fit(X, y)


def evaluate(clf, Xte, dte):
    p = clf.predict_proba(Xte)[:, 1]
    y = dte["y"].to_numpy().astype(int)
    pa, npaired = p_at_1(dte, p)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "p_at_1": float(pa), "n_paired": int(npaired)}, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.features_tsv, sep="\t")
    df["chrom_norm"] = df["chrom"].astype(str).apply(normalize_chrom)
    df["split"] = "other"
    df.loc[df["chrom_norm"].isin(TRAIN_CHROMS), "split"] = "train"
    df.loc[df["chrom_norm"].isin(VAL_CHROMS), "split"] = "val"
    df.loc[df["chrom_norm"].isin(TEST_CHROMS), "split"] = "test"

    bin_cols = [c for c in df.columns if c.startswith("bin") and (c.endswith("_mean") or c.endswith("_max"))]
    wg_cols = [c for c in df.columns if c.endswith("__wg")]
    full_feats = bin_cols + wg_cols
    compact = [c for c in COMPACT_FEATURES if c in df.columns]
    missing = [c for c in COMPACT_FEATURES if c not in df.columns]
    if missing:
        print(f"[interp] compact features not found, skipping: {missing}", file=sys.stderr)
    print(f"[interp] full set: {len(full_feats)}  compact set: {len(compact)}", file=sys.stderr)

    tr, te = df["split"] == "train", df["split"] == "test"
    ytr = df.loc[tr, "y"].to_numpy().astype(int)
    dte = df.loc[te].reset_index(drop=True)

    results = {}
    full_clf = fit_native(df.loc[tr, full_feats].to_numpy(), ytr, args.seed)
    results["full_55"], _ = evaluate(full_clf, dte[full_feats].to_numpy(), dte)

    comp_clf = fit_native(df.loc[tr, compact].to_numpy(), ytr, args.seed)
    results["compact_mech"], p_comp = evaluate(comp_clf, dte[compact].to_numpy(), dte)

    print("\n[interp] test-chromosome performance:", file=sys.stderr)
    print(f"  {'model':<14s} {'AUROC':>7s} {'AUPRC':>7s} {'P@1':>7s} {'n_paired':>9s}", file=sys.stderr)
    for k, m in results.items():
        print(f"  {k:<14s} {m['auroc']:>7.4f} {m['auprc']:>7.4f} {m['p_at_1']:>7.4f} {m['n_paired']:>9d}",
              file=sys.stderr)
    kept = results["compact_mech"]["auroc"] / results["full_55"]["auroc"]
    print(f"  compact keeps {kept:.1%} of full AUROC", file=sys.stderr)

    realized = {}
    if args.shap:
        try:
            import shap
            expl = shap.TreeExplainer(comp_clf)
            Xs = dte[compact].sample(min(3000, len(dte)), random_state=args.seed)
            sv = expl.shap_values(Xs)
            sv = sv[1] if isinstance(sv, list) else sv
            for j, f in enumerate(compact):
                col = Xs[f].to_numpy()
                ok = np.isfinite(col)
                realized[f] = float(np.sign(spearmanr(col[ok], sv[ok, j])[0])) if ok.sum() > 10 else 0.0
        except Exception as e:
            print(f"[interp] SHAP unavailable ({e}); using prob-correlation direction", file=sys.stderr)
    if not realized:
        for f in compact:
            col = dte[f].to_numpy()
            ok = np.isfinite(col)
            realized[f] = float(np.sign(spearmanr(col[ok], p_comp[ok])[0])) if ok.sum() > 10 else 0.0

    sign_rows = []
    for f in compact:
        exp_sign, mech = COMPACT_FEATURES[f]
        match = (realized[f] == exp_sign) or realized[f] == 0.0
        sign_rows.append({"feature": f, "expected_sign": exp_sign,
                          "realized_sign": int(realized[f]),
                          "consistent": bool(realized[f] == exp_sign),
                          "mechanism": mech})
    sign_df = pd.DataFrame(sign_rows)
    print("\n[interp] sign-consistency (expected vs realized direction):", file=sys.stderr)
    print(sign_df[["feature", "expected_sign", "realized_sign", "consistent"]].to_string(index=False),
          file=sys.stderr)
    bad = sign_df[~sign_df["consistent"]]
    if not bad.empty:
        print(f"\n[interp] {len(bad)} feature(s) used OPPOSITE to mechanism, investigate:", file=sys.stderr)
        print(bad[["feature", "mechanism"]].to_string(index=False), file=sys.stderr)

    loo_rows = []
    base_auroc = results["compact_mech"]["auroc"]
    for f in compact:
        sub = [c for c in compact if c != f]
        clf = fit_native(df.loc[tr, sub].to_numpy(), ytr, args.seed)
        m, _ = evaluate(clf, dte[sub].to_numpy(), dte)
        loo_rows.append({"dropped": f, "auroc_without": round(m["auroc"], 4),
                         "auroc_drop": round(base_auroc - m["auroc"], 4)})
    loo_df = pd.DataFrame(loo_rows).sort_values("auroc_drop", ascending=False)
    print("\n[interp] leave-one-out (AUROC drop when feature removed):", file=sys.stderr)
    print(loo_df.to_string(index=False), file=sys.stderr)

    sign_df.merge(loo_df.rename(columns={"dropped": "feature"}), on="feature", how="left") \
        .to_csv(os.path.join(args.out_dir, "interpretability.tsv"), sep="\t", index=False)
    with open(os.path.join(args.out_dir, "model_comparison.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    joblib.dump({"model": comp_clf, "features": compact, "expected_signs": COMPACT_FEATURES},
                os.path.join(args.out_dir, "compact_mech_model.joblib"))
    print(f"\n[interp] wrote interpretability.tsv, model_comparison.json, compact_mech_model.joblib",
          file=sys.stderr)


if __name__ == "__main__":
    main()