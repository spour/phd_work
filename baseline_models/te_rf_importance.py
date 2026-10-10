#!/usr/bin/env python3
# what the terminal-exon rf used, and in which direction
import argparse, json
import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

PRIOR = {
    "term_3ss": +1, "upstream_max_me3": +1, "term_ppt": +1, "upstream_mean_u1": +1,
    "u2af65_ppt": +1, "u2af35_ag": +1, "bp_strength": +1, "pas_prox_u1": +1,
    "dn5ss_has_strong": -1,
    "u1_suppression_down": -1,
    "u1_duplex_down": +1,
    "interv_me5_max": -1,
    "local_me5": -1, "u2_duplex_bp": +1,
}

def load_tsv(pos, neg, cols):
    P = pd.read_csv(pos, sep="\t", index_col=0); N = pd.read_csv(neg, sep="\t", index_col=0)
    P["_y"] = 1; N["_y"] = 0
    df = pd.concat([P, N]).fillna(0.0)
    y = df["_y"].to_numpy(); X = df.reindex(columns=cols).fillna(0.0)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_uncalibrated", required=True, help="rf_uncalibrated.joblib")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--pos_test", required=True); ap.add_argument("--neg_test", required=True)
    ap.add_argument("--n_repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    clf = joblib.load(a.rf_uncalibrated)
    cols = json.load(open(a.metadata))["winner_meta"]["columns"]
    Xte, yte = load_tsv(a.pos_test, a.neg_test, cols)

    imp = getattr(clf, "feature_importances_", None)
    pte = clf.predict_proba(Xte.to_numpy())[:, 1]
    print(f"[test] AUROC {roc_auc_score(yte, pte):.4f}  (n={len(yte):,}, pos={int(yte.sum()):,})\n")

    perm = permutation_importance(clf, Xte.to_numpy(), yte, scoring="roc_auc",
                                  n_repeats=a.n_repeats, random_state=a.seed, n_jobs=-1)

    rows = []
    for i, f in enumerate(cols):
        rho = spearmanr(Xte[f].to_numpy(), yte).correlation
        pr = PRIOR.get(f, 0)
        sign_ok = "n/a" if pr == 0 else ("ok" if np.sign(rho) == pr else "INVERTED")
        rows.append({"feature": f,
                     "impurity": float(imp[i]) if imp is not None else np.nan,
                     "perm_auroc_drop": float(perm.importances_mean[i]),
                     "perm_sd": float(perm.importances_std[i]),
                     "spearman": float(rho), "prior": pr, "direction": sign_ok})
    R = pd.DataFrame(rows).sort_values("perm_auroc_drop", ascending=False)

    print(f"{'feature':<22}{'impurity':>10}{'perm_dAUROC':>13}{'spearman':>10}{'dir':>10}")
    for _, r in R.iterrows():
        print(f"{r.feature:<22}{r.impurity:>10.3f}{r.perm_auroc_drop:>13.4f}"
              f"{r.spearman:>+10.3f}{r.direction:>10}")
    inv = R[R.direction == "INVERTED"]
    if len(inv):
        print(f"\n[flag] {len(inv)} feature(s) point against prior: {list(inv.feature)}")
        print("       check these before any mechanism claim.")
    R.to_csv("te_rf_importance.tsv", sep="\t", index=False)
    print("\n[out] te_rf_importance.tsv")

if __name__ == "__main__":
    main()