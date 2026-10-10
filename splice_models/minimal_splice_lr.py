# -*- coding: utf-8 -*-

# (v3) fit very small grouped-cv logistic regression models for splice environment using

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def get_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_splits: int,
    seed: int,
    cv_mode: str,
):
    if cv_mode == "groupkfold":
        if groups is None:
            raise ValueError("groupkfold requires groups")
        return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    elif cv_mode == "stratkfold":
        return list(StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        ).split(X, y))
    else:
        raise ValueError(f"Unknown cv_mode: {cv_mode}")


def make_gene_index(groups: np.ndarray):
    uniq = pd.unique(groups)
    return uniq, {g: np.flatnonzero(groups == g) for g in uniq}


def gene_block_bootstrap_metric(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    uniq, g2idx = make_gene_index(groups)
    rng = np.random.default_rng(seed)

    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([g2idx[g] for g in samp])
        yb = y[idx]
        pb = pred[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(metric_fn(yb, pb))
    return np.asarray(vals, dtype=float)


def mean_ci(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.mean(x)),
        float(np.quantile(x, alpha / 2)),
        float(np.quantile(x, 1 - alpha / 2)),
    )


def make_model(C: float, max_iter: int, class_weight_balanced: bool, seed: int):
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="l2",
            C=float(C),
            solver="lbfgs",
            max_iter=int(max_iter),
            class_weight=("balanced" if class_weight_balanced else None),
            n_jobs=1,
            random_state=int(seed),
        )),
    ])


def cv_oof_and_fold_coefs(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    feat_names: List[str],
    model,
    splits,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    oof = np.full(len(y), np.nan, dtype=float)
    rows = []

    for fold, (tr, te) in enumerate(splits, start=1):
        m = clone(model)
        clf = m.named_steps["clf"]
        if hasattr(clf, "random_state"):
            clf.random_state = seed + fold
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

        beta = m.named_steps["clf"].coef_.ravel().astype(float)
        for f, b in zip(feat_names, beta):
            rows.append({"fold": fold, "feature": f, "coef": float(b)})

    return oof, pd.DataFrame(rows)


def first_existing(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find a usable column for {label}. Tried: {candidates}")


def mean_from_existing(df: pd.DataFrame, candidates: List[str], label: str) -> pd.Series:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise SystemExit(f"Could not find usable columns for {label}. Tried: {candidates}")
    return df[cols].mean(axis=1)


def max_from_existing(df: pd.DataFrame, candidates: List[str], label: str) -> pd.Series:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise SystemExit(f"Could not find usable columns for {label}. Tried: {candidates}")
    return df[cols].max(axis=1)


def safe_col(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def center_within_gene(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby("gene", sort=False)
    for c in cols:
        if c in out.columns:
            out[f"{c}__wg"] = out[c] - grp[c].transform("mean")
    return out


def _ncrypt_bin(n: int) -> str:
    if n <= 2:
        return str(n)
    elif n <= 4:
        return "3-4"
    elif n <= 9:
        return "5-9"
    else:
        return "10+"


def add_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    need_u1_mean = [f"bin{i}_u1_mean" for i in range(1, 9)]
    missing_u1_mean = [c for c in need_u1_mean if c not in out.columns]
    if missing_u1_mean:
        raise SystemExit(f"Missing required U1 mean columns: {missing_u1_mean}")

    out["upstream_mean_u1"]   = out[[f"bin{i}_u1_mean" for i in range(1, 7)]].mean(axis=1)
    out["downstream_mean_u1"] = out[[f"bin{i}_u1_mean" for i in range(7, 9)]].mean(axis=1)
    out["pas_prox_u1"]        = out["bin6_u1_mean"]

    out["upstream_max_u1"]   = max_from_existing(out, [f"bin{i}_u1_max" for i in range(1, 7)], "upstream_max_u1")
    out["downstream_max_u1"] = max_from_existing(out, [f"bin{i}_u1_max" for i in range(7, 9)], "downstream_max_u1")
    out["pas_prox_u1_max"]   = max_from_existing(out, ["bin6_u1_max", "bin5_u1_max"], "pas_prox_u1_max")
    out["all_max_u1"]        = max_from_existing(out, ["all_u1_max"] + [f"bin{i}_u1_max" for i in range(1, 9)], "all_max_u1")

    term_3ss_source = first_existing(out, [
        "term_me3_top3mean_500_25", "term_me3_mean_500_25",
        "term_me3_top3mean_250_25", "term_me3_mean_250_25",
        "all_me3_decay_up_tau400",  "bin5_me3_max", "bin6_me3_max",
    ], "term_3ss")
    out["term_3ss"] = out[term_3ss_source]

    term_3ss_max_source = first_existing(out, [
        "term_me3_max_500_25", "term_me3_best_500_25",
        "term_me3_max_250_25", "term_me3_best_250_25",
        "bin5_me3_max", "bin6_me3_max",
    ], "term_3ss_max")
    out["term_3ss_max"] = out[term_3ss_max_source]

    out["upstream_mean_me3"] = mean_from_existing(out, [f"bin{i}_me3_max" for i in range(1, 7)], "upstream_mean_me3")
    out["upstream_max_me3"]  = max_from_existing(out,  [f"bin{i}_me3_max" for i in range(1, 7)], "upstream_max_me3")
    out["local_me3"]         = mean_from_existing(out, ["bin5_me3_max", "bin6_me3_max"], "local_me3")

    upstream_decay_u1_source = first_existing(out, [
        "all_u1_decay_up_tau400", "all_u1_decay_up_tau150",
        "all_u1_decay_up_tau50",  "upstream_mean_u1",
    ], "upstream_decay_u1")
    out["upstream_decay_u1"] = out[upstream_decay_u1_source]

    downstream_decay_u1_source = first_existing(out, [
        "all_u1_decay_dn_tau400", "all_u1_decay_dn_tau150",
        "all_u1_decay_dn_tau50",  "downstream_mean_u1",
    ], "downstream_decay_u1")
    out["downstream_decay_u1"] = out[downstream_decay_u1_source]

    upstream_decay_3ss_source = first_existing(out, [
        "all_me3_decay_up_tau400", "all_me3_decay_up_tau150",
        "all_me3_decay_up_tau50",  "upstream_max_me3", "term_3ss",
    ], "upstream_decay_3ss")
    out["upstream_decay_3ss"] = out[upstream_decay_3ss_source]

    downstream_decay_3ss_source = first_existing(out, [
        "all_me3_decay_dn_tau400", "all_me3_decay_dn_tau150",
        "all_me3_decay_dn_tau50",  "local_me3",
    ], "downstream_decay_3ss")
    out["downstream_decay_3ss"] = out[downstream_decay_3ss_source]

    me5_local_source = first_existing(out, [
        "bin6_me5_mean", "bin6_me5_max", "bin5_me5_mean",
    ], "local_me5")
    out["local_me5"] = out[me5_local_source]

    me5_local_max_source = first_existing(out, [
        "bin6_me5_max", "bin5_me5_max", "bin6_me5_mean",
    ], "local_me5_max")
    out["local_me5_max"] = out[me5_local_max_source]

    out["upstream_mean_me5"] = mean_from_existing(out, [f"bin{i}_me5_mean" for i in range(1, 7)], "upstream_mean_me5")
    out["upstream_max_me5"]  = max_from_existing(out,  [f"bin{i}_me5_max"  for i in range(1, 7)], "upstream_max_me5")

    out["onegood_u1"]             = out["upstream_max_u1"]
    out["onegood_3ss"]            = out["term_3ss_max"]
    out["onegood_local_suppress"] = max_from_existing(
        out, ["local_me5_max", "pas_prox_u1_max"], "onegood_local_suppress"
    )

    # v2: gradient / asymmetry features
    out["u1_gradient"] = out["upstream_mean_u1"] - out["downstream_mean_u1"]

    if "all_u1_decay_up_tau400" in out.columns and "all_u1_decay_dn_tau400" in out.columns:
        out["u1_gradient_decay"] = out["all_u1_decay_up_tau400"] - out["all_u1_decay_dn_tau400"]
    else:
        out["u1_gradient_decay"] = out["u1_gradient"]

    # v2: 3'ss prominence / contrast features
    out["me3_prominence"] = out["term_3ss"] - out["upstream_mean_me3"]

    eps = 1e-6
    denom = out["upstream_mean_me3"].clip(lower=eps)
    out["me3_upstream_ratio"] = out["term_3ss"] / denom

    out["me3_max_prominence"] = out["term_3ss"] - out["upstream_max_me3"]

    # v2: combined local suppression score
    out["local_suppression"] = out["pas_prox_u1"] + out["local_me5"]

    # v2: terminal identity score
    out["terminal_identity"] = out["upstream_mean_u1"] + out["me3_prominence"]

    out["dn5ss_best"]    = safe_col(out, "dn5ss_best_score")
    out["dn5ss_nearest"] = safe_col(out, "dn5ss_nearest_strong_dist")
    out["dn5ss_count"]   = safe_col(out, "dn5ss_n_strong", default=0.0)
    out["dn5ss_top3"]    = safe_col(out, "dn5ss_top3mean")
    out["dn5ss_dn250"]   = safe_col(out, "dn5ss_mean_dn250")

    out["dn5ss_has_strong"] = np.where(
        np.isfinite(out["dn5ss_nearest"].values), 1.0, 0.0
    )

    out["interv_me5_max"]      = safe_col(out, "interv_me5_max_500")
    out["interv_me5_mean"]     = safe_col(out, "interv_me5_mean_500")
    out["interv_me5_n_strong"] = safe_col(out, "interv_me5_n_strong_500", default=0.0)
    out["interv_geometry"]     = safe_col(out, "interv_exon_geometry_500", default=0.0)
    out["interv_len"]          = safe_col(out, "interv_len_500")

    if "interv_me5_max_1000" in out.columns:
        m = out["interv_me5_max"].isna()
        out.loc[m, "interv_me5_max"] = out.loc[m, "interv_me5_max_1000"]
    if "interv_me5_n_strong_1000" in out.columns:
        m = out["interv_me5_n_strong"].isna()
        out.loc[m, "interv_me5_n_strong"] = out.loc[m, "interv_me5_n_strong_1000"]
    if "interv_exon_geometry_1000" in out.columns:
        m = out["interv_geometry"].isna()
        out.loc[m, "interv_geometry"] = out.loc[m, "interv_exon_geometry_1000"]

    out["exon_clean"] = np.where(
        (out["interv_me5_n_strong"].fillna(0) == 0) & (out["interv_geometry"].fillna(0) == 1),
        1.0, 0.0
    )

    center_cols = [
        "upstream_mean_u1", "downstream_mean_u1", "pas_prox_u1",
        "upstream_max_u1",  "downstream_max_u1",  "pas_prox_u1_max",
        "all_max_u1",
        "term_3ss", "term_3ss_max",
        "upstream_mean_me3", "upstream_max_me3", "local_me3",
        "upstream_decay_u1", "downstream_decay_u1",
        "upstream_decay_3ss", "downstream_decay_3ss",
        "local_me5", "local_me5_max",
        "upstream_mean_me5", "upstream_max_me5",
        "onegood_u1", "onegood_3ss", "onegood_local_suppress",
        "u1_gradient", "u1_gradient_decay",
        "me3_prominence", "me3_upstream_ratio", "me3_max_prominence",
        "local_suppression", "terminal_identity",
        "dn5ss_best", "dn5ss_nearest", "dn5ss_count",
        "dn5ss_top3", "dn5ss_dn250", "dn5ss_has_strong",
        "interv_me5_max", "interv_me5_mean", "interv_me5_n_strong",
        "interv_geometry", "interv_len", "exon_clean",
    ]
    out = center_within_gene(out, center_cols)

    return out


def per_gene_prediction_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    dat = pd.DataFrame({"y": y, "pred": pred, "gene": groups})

    rows = []
    for gene, gdf in dat.groupby("gene", sort=False):
        if gdf["y"].nunique() < 2:
            continue

        yg = gdf["y"].values.astype(int)
        pg = gdf["pred"].values.astype(float)
        n_real    = int(np.sum(yg == 1))
        n_cryptic = int(np.sum(yg == 0))

        auc_g = float(roc_auc_score(yg, pg))

        real_mean    = float(np.mean(pg[yg == 1]))
        cryptic_mean = float(np.mean(pg[yg == 0]))
        delta = real_mean - cryptic_mean

        if delta > 0:
            acc = 1.0
        elif delta < 0:
            acc = 0.0
        else:
            acc = 0.5

        if len(yg) >= 4:
            rho, _ = spearmanr(pg, yg)
            spearman_rho = float(rho) if np.isfinite(rho) else np.nan
        else:
            spearman_rho = np.nan

        top1_is_real = int(yg[np.argmax(pg)] == 1)

        real_scores = pg[yg == 1]
        best_real_score = float(np.max(real_scores))
        cryptic_scores = pg[yg == 0]
        n_above = int(np.sum(cryptic_scores > best_real_score))
        rank_of_real = 1 + n_above
        reciprocal_rank = 1.0 / float(rank_of_real)

        n_cryptic_bin = _ncrypt_bin(n_cryptic)

        chance_rr = 1.0 / float(n_cryptic + n_real)

        rows.append({
            "gene":                   str(gene),
            "within_gene_auc":        auc_g,
            "real_mean_pred":         real_mean,
            "cryptic_mean_pred":      cryptic_mean,
            "real_minus_cryptic_pred": delta,
            "within_gene_acc":        acc,
            "within_gene_spearman":   spearman_rho,
            "top1_is_real":           top1_is_real,
            "rank_of_real":           rank_of_real,
            "reciprocal_rank":        reciprocal_rank,
            "chance_rr":              chance_rr,
            "n_real":                 n_real,
            "n_cryptic":              n_cryptic,
            "n_cryptic_bin":          n_cryptic_bin,
            "n_sites":               int(len(yg)),
        })

    return pd.DataFrame(rows)


def bootstrap_from_gene_metric_table(
    gene_df: pd.DataFrame,
    metric_col: str,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    if gene_df.empty:
        return np.asarray([], dtype=float)
    vals = gene_df[metric_col].dropna().values.astype(float)
    n = len(vals)
    if n == 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(vals[idx])))
    return np.asarray(boots, dtype=float)


def compute_stratified_metrics(gene_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if gene_df.empty:
        return pd.DataFrame()

    rows = []
    for bin_label, bdf in gene_df.groupby("n_cryptic_bin", sort=False):
        n_genes = len(bdf)
        p_at_1 = float(bdf["top1_is_real"].mean())
        mrr = float(bdf["reciprocal_rank"].mean())
        chance = float(bdf["chance_rr"].mean())
        mean_rank = float(bdf["rank_of_real"].mean())
        rows.append({
            "model":        model_name,
            "n_cryptic_bin": bin_label,
            "n_genes":      n_genes,
            "p_at_1":       p_at_1,
            "mrr":          mrr,
            "chance_mrr":   chance,
            "mean_rank":    mean_rank,
            "lift_over_chance": mrr / chance if chance > 0 else np.nan,
        })

    n_genes = len(gene_df)
    rows.append({
        "model":        model_name,
        "n_cryptic_bin": "all",
        "n_genes":      n_genes,
        "p_at_1":       float(gene_df["top1_is_real"].mean()),
        "mrr":          float(gene_df["reciprocal_rank"].mean()),
        "chance_mrr":   float(gene_df["chance_rr"].mean()),
        "mean_rank":    float(gene_df["rank_of_real"].mean()),
        "lift_over_chance": (
            float(gene_df["reciprocal_rank"].mean()) / float(gene_df["chance_rr"].mean())
            if float(gene_df["chance_rr"].mean()) > 0 else np.nan
        ),
    })

    return pd.DataFrame(rows)


def build_model_library() -> Tuple[Dict[str, List[str]], Dict[str, str]]:

    model_defs: Dict[str, List[str]] = {
        "model4": [
            "upstream_mean_u1", "downstream_mean_u1", "term_3ss", "pas_prox_u1"
        ],
        "hybrid4b": [
            "upstream_mean_u1", "local_me5", "term_3ss", "pas_prox_u1"
        ],
        "hybrid_broad_peak4": [
            "upstream_mean_u1", "upstream_max_me3", "local_me5", "pas_prox_u1"
        ],
        "model4_wg": [
            "upstream_mean_u1__wg", "downstream_mean_u1__wg",
            "term_3ss__wg", "pas_prox_u1__wg"
        ],
        "hybrid4b_wg": [
            "upstream_mean_u1__wg", "local_me5__wg",
            "term_3ss__wg", "pas_prox_u1__wg"
        ],


        "mech_axes6": [
            "upstream_mean_u1",
            "term_3ss",
            "local_me5",
            "pas_prox_u1",
            "dn5ss_has_strong",
            "interv_me5_max",
        ],
        "mech_axes7": [
            "upstream_mean_u1",
            "upstream_max_me3",
            "term_3ss",
            "local_me5",
            "pas_prox_u1",
            "dn5ss_has_strong",
            "interv_me5_max",
        ],
        "mech_axes8": [
            "upstream_mean_u1",
            "u1_gradient",
            "upstream_max_me3",
            "term_3ss",
            "local_me5",
            "pas_prox_u1",
            "dn5ss_has_strong",
            "interv_me5_max",
        ],

        "tele_integrity6": [
            "upstream_mean_u1",
            "pas_prox_u1",
            "dn5ss_has_strong",
            "dn5ss_best",
            "interv_me5_max",
            "term_3ss",
        ],
        "tele_integrity7": [
            "upstream_mean_u1",
            "downstream_mean_u1",
            "pas_prox_u1",
            "dn5ss_has_strong",
            "dn5ss_best",
            "interv_me5_max",
            "term_3ss",
        ],

        "terminal_exon6": [
            "term_3ss",
            "upstream_max_me3",
            "local_me5",
            "interv_me5_max",
            "interv_me5_n_strong",
            "interv_geometry",
        ],
        "terminal_exon7": [
            "upstream_mean_u1",
            "term_3ss",
            "upstream_max_me3",
            "local_me5",
            "interv_me5_max",
            "interv_me5_n_strong",
            "interv_geometry",
        ],

        "mech_axes6_wg": [
            "upstream_mean_u1__wg",
            "term_3ss__wg",
            "local_me5__wg",
            "pas_prox_u1__wg",
            "dn5ss_has_strong__wg",
            "interv_me5_max__wg",
        ],
        "mech_axes7_wg": [
            "upstream_mean_u1__wg",
            "upstream_max_me3__wg",
            "term_3ss__wg",
            "local_me5__wg",
            "pas_prox_u1__wg",
            "dn5ss_has_strong__wg",
            "interv_me5_max__wg",
        ],
        "tele_integrity6_wg": [
            "upstream_mean_u1__wg",
            "pas_prox_u1__wg",
            "dn5ss_has_strong__wg",
            "dn5ss_best__wg",
            "interv_me5_max__wg",
            "term_3ss__wg",
        ],
        "terminal_exon6_wg": [
            "term_3ss__wg",
            "upstream_max_me3__wg",
            "local_me5__wg",
            "interv_me5_max__wg",
            "interv_me5_n_strong__wg",
            "interv_geometry__wg",
        ],

        "model3":               ["upstream_mean_u1", "downstream_mean_u1", "term_3ss"],
        "mech4":                ["upstream_decay_u1", "pas_prox_u1", "term_3ss", "upstream_decay_3ss"],
        "hybrid4a":             ["upstream_decay_u1", "downstream_mean_u1", "term_3ss", "pas_prox_u1"],

        "u1_mean2":             ["upstream_mean_u1", "downstream_mean_u1"],
        "u1_max2":              ["upstream_max_u1", "downstream_max_u1"],
        "u1_meanmax4":          ["upstream_mean_u1", "upstream_max_u1", "downstream_mean_u1", "downstream_max_u1"],
        "u1_local3":            ["upstream_mean_u1", "pas_prox_u1", "pas_prox_u1_max"],
        "u1_burden3":           ["upstream_decay_u1", "downstream_decay_u1", "pas_prox_u1"],
        "u1_peak4":             ["upstream_max_u1", "downstream_max_u1", "pas_prox_u1", "pas_prox_u1_max"],

        "me3_term2":            ["term_3ss", "term_3ss_max"],
        "me3_broad2":           ["upstream_mean_me3", "upstream_max_me3"],
        "me3_local2":           ["local_me3", "term_3ss"],
        "me3_burden3":          ["term_3ss", "upstream_decay_3ss", "downstream_decay_3ss"],
        "me3_peak3":            ["term_3ss_max", "upstream_max_me3", "local_me3"],

        "local_competition3":   ["local_me5", "local_me5_max", "pas_prox_u1"],
        "local_competition4":   ["local_me5", "local_me5_max", "pas_prox_u1", "pas_prox_u1_max"],
        "local_acceptor_donor4":["local_me5", "local_me5_max", "local_me3", "term_3ss"],

        "hybrid_meanmax4":      ["upstream_mean_u1", "upstream_max_u1", "term_3ss", "local_me5"],
        "hybrid_peak4":         ["upstream_max_u1", "local_me5", "term_3ss", "pas_prox_u1"],
        "hybrid_peakmax4":      ["upstream_max_u1", "local_me5_max", "term_3ss_max", "pas_prox_u1_max"],
        "hybrid_broad_acceptor4":["upstream_mean_u1", "term_3ss", "upstream_mean_me3", "local_me5"],
        "hybrid_decay_peak4":   ["upstream_decay_u1", "term_3ss_max", "local_me5", "pas_prox_u1"],
        "hybrid_decay_broad4":  ["upstream_decay_u1", "term_3ss", "local_me5", "downstream_mean_u1"],

        "onegood3":             ["onegood_u1", "onegood_3ss", "onegood_local_suppress"],
        "onegood4":             ["onegood_u1", "onegood_3ss", "onegood_local_suppress", "pas_prox_u1_max"],
        "winner_take_all4":     ["upstream_max_u1", "term_3ss_max", "local_me5_max", "pas_prox_u1_max"],
        "winner_plus_context4": ["upstream_max_u1", "term_3ss_max", "local_me5_max", "downstream_mean_u1"],

        "u1_mean2_wg":          ["upstream_mean_u1__wg", "downstream_mean_u1__wg"],
        "u1_max2_wg":           ["upstream_max_u1__wg", "downstream_max_u1__wg"],
        "u1_local3_wg":         ["upstream_mean_u1__wg", "pas_prox_u1__wg", "pas_prox_u1_max__wg"],
        "me3_term2_wg":         ["term_3ss__wg", "term_3ss_max__wg"],
        "me3_broad2_wg":        ["upstream_mean_me3__wg", "upstream_max_me3__wg"],
        "local_competition3_wg":["local_me5__wg", "local_me5_max__wg", "pas_prox_u1__wg"],
        "local_acceptor_donor4_wg":["local_me5__wg", "local_me5_max__wg", "local_me3__wg", "term_3ss__wg"],
        "hybrid_peak4_wg":      ["upstream_max_u1__wg", "local_me5__wg", "term_3ss__wg", "pas_prox_u1__wg"],
        "hybrid_peakmax4_wg":   ["upstream_max_u1__wg", "local_me5_max__wg", "term_3ss_max__wg", "pas_prox_u1_max__wg"],
        "hybrid_meanmax4_wg":   ["upstream_mean_u1__wg", "upstream_max_u1__wg", "term_3ss__wg", "local_me5__wg"],
        "onegood3_wg":          ["onegood_u1__wg", "onegood_3ss__wg", "onegood_local_suppress__wg"],
        "winner_take_all4_wg":  ["upstream_max_u1__wg", "term_3ss_max__wg", "local_me5_max__wg", "pas_prox_u1_max__wg"],

        "gradient_2feat":       ["u1_gradient", "term_3ss"],
        "gradient_3feat":       ["u1_gradient", "term_3ss", "pas_prox_u1"],
        "gradient_4feat":       ["u1_gradient", "term_3ss", "local_me5", "pas_prox_u1"],
        "gradient_decay_3feat": ["u1_gradient_decay", "term_3ss", "pas_prox_u1"],
        "gradient_decay_4feat": ["u1_gradient_decay", "term_3ss", "local_me5", "pas_prox_u1"],

        "prominence_2feat":     ["me3_prominence", "upstream_mean_u1"],
        "prominence_3feat":     ["me3_prominence", "upstream_mean_u1", "local_me5"],
        "prominence_4feat":     ["me3_prominence", "upstream_mean_u1", "local_me5", "pas_prox_u1"],
        "prominence_ratio4":    ["me3_upstream_ratio", "upstream_mean_u1", "local_me5", "pas_prox_u1"],
        "prominence_max4":      ["me3_max_prominence", "upstream_mean_u1", "local_me5", "pas_prox_u1"],
        "prominence_gradient4": ["me3_prominence", "u1_gradient", "local_me5", "pas_prox_u1"],

        "suppress_combine3":    ["local_suppression", "upstream_mean_u1", "term_3ss"],
        "suppress_combine4":    ["local_suppression", "upstream_mean_u1", "term_3ss", "u1_gradient"],
        "suppress_prominence4": ["local_suppression", "me3_prominence", "upstream_mean_u1", "u1_gradient"],

        "terminal_id_2feat":    ["terminal_identity", "local_suppression"],
        "terminal_id_3feat":    ["terminal_identity", "local_me5", "pas_prox_u1"],
        "terminal_id_4feat":    ["terminal_identity", "local_me5", "pas_prox_u1", "u1_gradient"],

        "best_plus_prominence5":["upstream_mean_u1", "upstream_max_me3", "me3_prominence", "local_me5", "pas_prox_u1"],
        "best_plus_gradient5":  ["upstream_mean_u1", "upstream_max_me3", "local_me5", "pas_prox_u1", "u1_gradient"],

        "gradient_4feat_wg":    ["u1_gradient__wg", "term_3ss__wg", "local_me5__wg", "pas_prox_u1__wg"],
        "prominence_4feat_wg":  ["me3_prominence__wg", "upstream_mean_u1__wg", "local_me5__wg", "pas_prox_u1__wg"],
        "prominence_gradient4_wg": ["me3_prominence__wg", "u1_gradient__wg", "local_me5__wg", "pas_prox_u1__wg"],
        "suppress_combine3_wg": ["local_suppression__wg", "upstream_mean_u1__wg", "term_3ss__wg"],
        "suppress_combine4_wg": ["local_suppression__wg", "upstream_mean_u1__wg", "term_3ss__wg", "u1_gradient__wg"],
        "terminal_id_3feat_wg": ["terminal_identity__wg", "local_me5__wg", "pas_prox_u1__wg"],
        "terminal_id_4feat_wg": ["terminal_identity__wg", "local_me5__wg", "pas_prox_u1__wg", "u1_gradient__wg"],
        "suppress_prominence4_wg": ["local_suppression__wg", "me3_prominence__wg", "upstream_mean_u1__wg", "u1_gradient__wg"],

        "dn5ss_only2":          ["dn5ss_has_strong", "dn5ss_best"],
        "dn5ss_3feat":          ["dn5ss_has_strong", "dn5ss_best", "dn5ss_count"],
        "dn5ss_full4":          ["dn5ss_has_strong", "dn5ss_best", "dn5ss_count", "dn5ss_dn250"],

        "tele_3feat":           ["dn5ss_has_strong", "upstream_mean_u1", "pas_prox_u1"],
        "tele_4feat":           ["dn5ss_has_strong", "dn5ss_best", "upstream_mean_u1", "pas_prox_u1"],
        "tele_gradient4":       ["dn5ss_has_strong", "dn5ss_best", "u1_gradient", "pas_prox_u1"],

        "integrity_2feat":      ["interv_me5_max", "interv_geometry"],
        "integrity_3feat":      ["interv_me5_max", "interv_geometry", "term_3ss"],
        "integrity_4feat":      ["interv_me5_max", "interv_me5_n_strong", "interv_geometry", "term_3ss"],
        "exon_clean_3feat":     ["exon_clean", "term_3ss", "upstream_mean_u1"],
        "exon_clean_4feat":     ["exon_clean", "term_3ss", "upstream_mean_u1", "pas_prox_u1"],

        "mech_full4":           ["dn5ss_has_strong", "interv_me5_max", "term_3ss", "upstream_mean_u1"],
        "mech_full5":           ["dn5ss_has_strong", "interv_me5_max", "term_3ss", "upstream_mean_u1", "pas_prox_u1"],
        "mech_full5b":          ["dn5ss_has_strong", "interv_me5_max", "term_3ss", "upstream_mean_u1", "local_me5"],
        "mech_geometry5":       ["dn5ss_has_strong", "exon_clean", "term_3ss", "upstream_mean_u1", "pas_prox_u1"],
        "mech_gradient5":       ["dn5ss_has_strong", "interv_me5_max", "term_3ss", "u1_gradient", "pas_prox_u1"],
        "mech_prominence5":     ["dn5ss_has_strong", "interv_me5_max", "me3_prominence", "upstream_mean_u1", "pas_prox_u1"],

        "best_v1_plus_dn5ss":   ["upstream_mean_u1", "upstream_max_me3", "local_me5", "pas_prox_u1", "dn5ss_has_strong"],
        "best_v1_plus_integ":   ["upstream_mean_u1", "upstream_max_me3", "local_me5", "pas_prox_u1", "interv_me5_max"],
        "best_v1_plus_both":    ["upstream_mean_u1", "upstream_max_me3", "local_me5", "dn5ss_has_strong", "interv_me5_max"],

        "dn5ss_3feat_wg":       ["dn5ss_has_strong__wg", "dn5ss_best__wg", "dn5ss_count__wg"],
        "tele_4feat_wg":        ["dn5ss_has_strong__wg", "dn5ss_best__wg", "upstream_mean_u1__wg", "pas_prox_u1__wg"],
        "integrity_3feat_wg":   ["interv_me5_max__wg", "interv_geometry__wg", "term_3ss__wg"],
        "exon_clean_4feat_wg":  ["exon_clean__wg", "term_3ss__wg", "upstream_mean_u1__wg", "pas_prox_u1__wg"],
        "mech_full4_wg":        ["dn5ss_has_strong__wg", "interv_me5_max__wg", "term_3ss__wg", "upstream_mean_u1__wg"],
        "mech_full5_wg":        ["dn5ss_has_strong__wg", "interv_me5_max__wg", "term_3ss__wg", "upstream_mean_u1__wg", "pas_prox_u1__wg"],
    }

    model_tiers: Dict[str, str] = {}

    for name in [
        "model4", "hybrid4b", "hybrid_broad_peak4", "model4_wg", "hybrid4b_wg"
    ]:
        model_tiers[name] = "core"

    for name in [
        "mech_axes6", "mech_axes7", "mech_axes8",
        "tele_integrity6", "tele_integrity7",
        "terminal_exon6", "terminal_exon7",
        "mech_axes6_wg", "mech_axes7_wg",
        "tele_integrity6_wg", "terminal_exon6_wg",
    ]:
        model_tiers[name] = "exploratory"

    for name in model_defs:
        model_tiers.setdefault(name, "legacy")

    return model_defs, model_tiers


def filter_model_library(
    model_defs: Dict[str, List[str]],
    model_tiers: Dict[str, str],
    which: str,
    max_features: int,
) -> Dict[str, List[str]]:
    keep: Dict[str, List[str]] = {}

    for model_name, feat_names in model_defs.items():
        if len(feat_names) > int(max_features):
            continue

        tier = model_tiers.get(model_name, "legacy")

        if which == "core":
            if tier != "core":
                continue
        elif which == "exploratory":
            if tier not in {"core", "exploratory"}:
                continue
        elif which == "all":
            pass
        else:
            raise ValueError(f"Unknown model tier: {which}")

        keep[model_name] = feat_names

    return keep
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def _model_color(name: str) -> str:
    is_v3 = any(tag in name for tag in [
        "dn5ss", "tele_", "integrity", "exon_clean", "mech_full",
        "mech_geometry", "mech_gradient", "mech_prominence",
        "best_v1_plus",
    ])
    is_v2 = any(tag in name for tag in [
        "gradient", "prominence", "suppress", "terminal_id", "best_plus",
    ])

    if name.endswith("_wg"):
        if is_v3:
            return "#2CA02C"
        elif is_v2:
            return "#E07B54"
        else:
            return "#888888"
    else:
        if is_v3:
            return "#D62728"
        elif is_v2:
            return "#3B6FB6"
        else:
            return "#111111"


def plot_performance(summary_df: pd.DataFrame, out_png: str):
    tmp = summary_df.copy()
    fig, axes = plt.subplots(1, 4, figsize=(30, max(8.5, 0.30 * len(tmp))))

    metric_pairs = [
        ("auc",              "auc_ci_lo",              "auc_ci_hi",
         "Grouped OOF AUC", "Site-level held-out-gene AUC"),
        ("within_gene_auc",  "within_gene_auc_ci_lo",  "within_gene_auc_ci_hi",
         "Mean within-gene AUC", "Within-gene site-ranking AUC"),
        ("mrr",              "mrr_ci_lo",              "mrr_ci_hi",
         "Mean reciprocal rank", "MRR (1.0 = real always ranked #1)"),
        ("top1_acc",         "top1_acc_ci_lo",         "top1_acc_ci_hi",
         "Top-1 accuracy (P@1)", "Fraction of genes: real site ranked #1"),
    ]

    for ax, (col, lo_col, hi_col, xlabel, title) in zip(axes, metric_pairs):
        if col not in tmp.columns:
            ax.set_visible(False)
            continue
        sub = tmp.dropna(subset=[col]).sort_values(col, ascending=True).reset_index(drop=True)
        ypos = np.arange(len(sub))
        x  = sub[col].values
        lo = sub[lo_col].values if lo_col in sub.columns else x
        hi = sub[hi_col].values if hi_col in sub.columns else x

        colors = [_model_color(m) for m in sub["model"].values]
        ax.errorbar(
            x, ypos,
            xerr=np.vstack([x - lo, hi - x]),
            fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5, zorder=2,
        )
        for yi, (xi, ci) in enumerate(zip(x, colors)):
            ax.scatter(xi, yi, s=28, color=ci, edgecolors="black",
                       linewidths=0.6, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["model"].values, fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="x", color="#D9DDE3", linewidth=0.7, alpha=0.7)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D62728",
               markeredgecolor="black", markersize=7, label="v3 (raw)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2CA02C",
               markeredgecolor="black", markersize=7, label="v3 (wg-centered)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3B6FB6",
               markeredgecolor="black", markersize=7, label="v2 (raw)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E07B54",
               markeredgecolor="black", markersize=7, label="v2 (wg-centered)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#111111",
               markeredgecolor="black", markersize=7, label="v1 (raw)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor="black", markersize=7, label="v1 (wg-centered)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_top1_acc(summary_df: pd.DataFrame, out_png: str, chance_p1: float = None):
    if "top1_acc" not in summary_df.columns:
        return
    tmp = summary_df.dropna(subset=["top1_acc"]).sort_values("top1_acc", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, max(6.5, 0.28 * len(tmp))))
    ypos = np.arange(len(tmp))
    x  = tmp["top1_acc"].values
    lo = tmp["top1_acc_ci_lo"].values if "top1_acc_ci_lo" in tmp.columns else x
    hi = tmp["top1_acc_ci_hi"].values if "top1_acc_ci_hi" in tmp.columns else x
    colors = [_model_color(m) for m in tmp["model"].values]

    ax.errorbar(x, ypos,
                xerr=np.vstack([x - lo, hi - x]),
                fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5, zorder=2)
    for yi, (xi, ci) in enumerate(zip(x, colors)):
        ax.scatter(xi, yi, s=28, color=ci, edgecolors="black", linewidths=0.6, zorder=3)

    if chance_p1 is not None:
        ax.axvline(chance_p1, color="0.6", linewidth=1.0, linestyle="--", label=f"Chance ({chance_p1:.2f})")
    ax.set_yticks(ypos)
    ax.set_yticklabels(tmp["model"].values, fontsize=7)
    ax.set_xlabel("Top-1 accuracy  (fraction of genes where top-ranked site is real)")
    ax.set_title("Does the model identify the primary PAS in each gene?")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="x", color="#D9DDE3", linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_mrr(summary_df: pd.DataFrame, out_png: str, chance_mrr: float = None):
    if "mrr" not in summary_df.columns:
        return
    tmp = summary_df.dropna(subset=["mrr"]).sort_values("mrr", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, max(6.5, 0.28 * len(tmp))))
    ypos = np.arange(len(tmp))
    x  = tmp["mrr"].values
    lo = tmp["mrr_ci_lo"].values if "mrr_ci_lo" in tmp.columns else x
    hi = tmp["mrr_ci_hi"].values if "mrr_ci_hi" in tmp.columns else x
    colors = [_model_color(m) for m in tmp["model"].values]

    ax.errorbar(x, ypos,
                xerr=np.vstack([x - lo, hi - x]),
                fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5, zorder=2)
    for yi, (xi, ci) in enumerate(zip(x, colors)):
        ax.scatter(xi, yi, s=28, color=ci, edgecolors="black", linewidths=0.6, zorder=3)

    if chance_mrr is not None:
        ax.axvline(chance_mrr, color="0.6", linewidth=1.0, linestyle="--", label=f"Chance ({chance_mrr:.2f})")
        ax.legend(loc="lower right", frameon=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels(tmp["model"].values, fontsize=7)
    ax.set_xlabel("Mean reciprocal rank  (1.0 = real always ranked #1)")
    ax.set_title("MRR: how well does the model rank the real PAS?")
    ax.grid(True, axis="x", color="#D9DDE3", linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(coef_df: pd.DataFrame, out_png: str):
    tmp = coef_df.copy()
    models = list(tmp["model"].drop_duplicates())
    ncol = 3
    nrow = math.ceil(len(models) / ncol)

    fig, axes = plt.subplots(nrow, ncol, figsize=(5.7 * ncol, 4.0 * nrow))
    axes = np.array(axes).reshape(-1)

    for ax, model_name in zip(axes, models):
        sub = tmp[tmp["model"] == model_name].copy().sort_values("coef_mean")
        ypos = np.arange(len(sub))
        x  = sub["coef_mean"].values
        lo = sub["coef_ci_lo"].values
        hi = sub["coef_ci_hi"].values

        colors = ["#3B6FB6" if v > 0 else "#C65F5F" for v in x]
        ax.barh(ypos, x, color=colors, alpha=0.6, height=0.6, zorder=1)
        ax.errorbar(x, ypos,
                    xerr=np.vstack([x - lo, hi - x]),
                    fmt="none", ecolor="black", elinewidth=1.2, capsize=3, zorder=2)
        ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["feature"].values, fontsize=8)
        ax.set_xlabel("Coefficient (standardised)")
        ax.set_title(model_name)

    for ax in axes[len(models):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--features_tsv", required=True)
    ap.add_argument("--out_prefix",   required=True)

    ap.add_argument("--cv_mode", choices=["groupkfold", "stratkfold"], default="groupkfold")
    ap.add_argument("--n_splits",  type=int,   default=5)
    ap.add_argument("--seed",      type=int,   default=1)
    ap.add_argument("--n_boot",    type=int,   default=2000)

    ap.add_argument("--C",         type=float, default=1.0)
    ap.add_argument("--max_iter",  type=int,   default=2000)
    ap.add_argument("--class_weight_balanced", action="store_true")

    ap.add_argument("--set_col",      default="set")
    ap.add_argument("--analyze_set",  default=None)

    ap.add_argument(
        "--model_tier",
        choices=["core", "exploratory", "all"],
        default="core",
        help=(
            "core = small publication-facing models only; "
            "exploratory = core + capped 6-8 feature mechanistic models; "
            "all = everything in the full zoo"
        ),
    )
    ap.add_argument(
        "--max_features",
        type=int,
        default=8,
        help="Hard cap on number of features per model.",
    )

    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_tsv, sep="\t")
    need = {"y", "gene"}
    if not need.issubset(df.columns):
        raise SystemExit(f"features_tsv must contain columns {sorted(need)}")

    set_col = str(args.set_col)
    if set_col not in df.columns:
        df[set_col] = "NA"

    if args.analyze_set:
        want = {x.strip() for x in args.analyze_set.split(",") if x.strip()}
        df = df[df[set_col].astype(str).isin(want)].copy()

    df = add_minimal_features(df)

    y      = df["y"].astype(int).values
    groups = df["gene"].astype(str).values
    use_groups = groups if args.cv_mode == "groupkfold" else None

    model = make_model(
        C=args.C,
        max_iter=args.max_iter,
        class_weight_balanced=args.class_weight_balanced,
        seed=args.seed,
    )

    model_defs, model_tiers = build_model_library()
    model_defs = filter_model_library(
        model_defs=model_defs,
        model_tiers=model_tiers,
        which=args.model_tier,
        max_features=args.max_features,
    )

    valid_models = {}
    for model_name, feat_names in model_defs.items():
        missing = [f for f in feat_names if f not in df.columns]
        if missing:
            print(f"[SKIP] model {model_name!r}: missing columns {missing}")
        else:
            valid_models[model_name] = feat_names
    model_defs = valid_models

    summary_rows     = []
    coef_rows        = []
    oof_rows         = []
    gene_metric_rows = []
    stratified_rows  = []

    for i, (model_name, feat_names) in enumerate(model_defs.items()):
        X = df[feat_names].values
        splits = get_splits(X, y, use_groups, args.n_splits, args.seed, args.cv_mode)

        oof, fold_coef_df = cv_oof_and_fold_coefs(
            X=X, y=y, groups=use_groups, feat_names=feat_names,
            model=model, splits=splits,
            seed=args.seed + 100 * i,
        )

        auc = float(roc_auc_score(y, oof))
        apv = float(average_precision_score(y, oof))

        auc_boot = gene_block_bootstrap_metric(
            y=y, pred=oof, groups=groups, metric_fn=roc_auc_score,
            n_boot=args.n_boot, seed=args.seed + 1000 + i,
        )
        ap_boot = gene_block_bootstrap_metric(
            y=y, pred=oof, groups=groups, metric_fn=average_precision_score,
            n_boot=args.n_boot, seed=args.seed + 2000 + i,
        )
        _, auc_lo, auc_hi = mean_ci(auc_boot)
        _, ap_lo,  ap_hi  = mean_ci(ap_boot)

        gene_pred_df = per_gene_prediction_metrics(y=y, pred=oof, groups=groups)
        gene_pred_df["model"] = model_name
        gene_metric_rows.append(gene_pred_df)

        wg_auc    = float(gene_pred_df["within_gene_auc"].mean())      if not gene_pred_df.empty else np.nan
        wg_acc    = float(gene_pred_df["within_gene_acc"].mean())      if not gene_pred_df.empty else np.nan
        wg_margin = float(gene_pred_df["real_minus_cryptic_pred"].mean()) if not gene_pred_df.empty else np.nan
        n_paired  = int(len(gene_pred_df))

        spearman_vals = gene_pred_df["within_gene_spearman"].dropna()
        wg_spearman   = float(spearman_vals.mean()) if len(spearman_vals) > 0 else np.nan
        n_spearman    = int(len(spearman_vals))

        top1_acc      = float(gene_pred_df["top1_is_real"].mean()) if not gene_pred_df.empty else np.nan

        mrr           = float(gene_pred_df["reciprocal_rank"].mean()) if not gene_pred_df.empty else np.nan

        wg_auc_boot  = bootstrap_from_gene_metric_table(gene_pred_df, "within_gene_auc",    args.n_boot, args.seed + 3000 + i)
        wg_acc_boot  = bootstrap_from_gene_metric_table(gene_pred_df, "within_gene_acc",    args.n_boot, args.seed + 4000 + i)
        sp_boot      = bootstrap_from_gene_metric_table(gene_pred_df, "within_gene_spearman", args.n_boot, args.seed + 5000 + i)
        top1_boot    = bootstrap_from_gene_metric_table(gene_pred_df, "top1_is_real",       args.n_boot, args.seed + 6000 + i)
        mrr_boot     = bootstrap_from_gene_metric_table(gene_pred_df, "reciprocal_rank",    args.n_boot, args.seed + 7000 + i)

        _, wg_auc_lo,  wg_auc_hi  = mean_ci(wg_auc_boot)
        _, wg_acc_lo,  wg_acc_hi  = mean_ci(wg_acc_boot)
        _, sp_lo,      sp_hi      = mean_ci(sp_boot)
        _, top1_lo,    top1_hi    = mean_ci(top1_boot)
        _, mrr_lo,     mrr_hi     = mean_ci(mrr_boot)

        strat_df = compute_stratified_metrics(gene_pred_df, model_name)
        stratified_rows.append(strat_df)

        full_fit = clone(model)
        full_fit.fit(X, y)
        beta_full = full_fit.named_steps["clf"].coef_.ravel().astype(float)

        summary_rows.append({
            "model":                     model_name,
            "n_features":                len(feat_names),
            "auc":                       auc,
            "auc_ci_lo":                 auc_lo,
            "auc_ci_hi":                 auc_hi,
            "ap":                        apv,
            "ap_ci_lo":                  ap_lo,
            "ap_ci_hi":                  ap_hi,
            "within_gene_auc":           wg_auc,
            "within_gene_auc_ci_lo":     wg_auc_lo,
            "within_gene_auc_ci_hi":     wg_auc_hi,
            "within_gene_acc":           wg_acc,
            "within_gene_acc_ci_lo":     wg_acc_lo,
            "within_gene_acc_ci_hi":     wg_acc_hi,
            "within_gene_spearman":      wg_spearman,
            "within_gene_spearman_ci_lo": sp_lo,
            "within_gene_spearman_ci_hi": sp_hi,
            "n_genes_spearman":          n_spearman,
            "top1_acc":                  top1_acc,
            "top1_acc_ci_lo":            top1_lo,
            "top1_acc_ci_hi":            top1_hi,
            "mrr":                       mrr,
            "mrr_ci_lo":                 mrr_lo,
            "mrr_ci_hi":                 mrr_hi,
            "within_gene_margin":        wg_margin,
            "n_paired_genes":            n_paired,
        })

        for f, bf in zip(feat_names, beta_full):
            sub = fold_coef_df[fold_coef_df["feature"] == f]["coef"].values.astype(float)
            coef_rows.append({
                "model":         model_name,
                "feature":       f,
                "coef_full_fit": float(bf),
                "coef_mean":     float(np.mean(sub)),
                "coef_std":      float(np.std(sub, ddof=0)),
                "coef_ci_lo":    float(np.quantile(sub, 0.025)),
                "coef_ci_hi":    float(np.quantile(sub, 0.975)),
            })

        tmp_oof = df[["y", "gene"]].copy()
        if "id" in df.columns:
            tmp_oof["id"] = df["id"].astype(str)
        else:
            tmp_oof["id"] = np.arange(len(df)).astype(str)
        tmp_oof["model"] = model_name
        tmp_oof["p"]     = oof
        for f in feat_names:
            tmp_oof[f] = df[f].values
        oof_rows.append(tmp_oof)

    summary_df      = pd.DataFrame(summary_rows)
    coef_df         = pd.DataFrame(coef_rows)
    oof_df          = pd.concat(oof_rows, axis=0, ignore_index=True)
    gene_metrics_df = pd.concat(gene_metric_rows, axis=0, ignore_index=True)
    stratified_df   = pd.concat(stratified_rows, axis=0, ignore_index=True)

    summary_df.to_csv(str(out_prefix) + ".summary.tsv",        sep="\t", index=False)
    coef_df.to_csv(   str(out_prefix) + ".coefs.tsv",          sep="\t", index=False)
    oof_df.to_csv(    str(out_prefix) + ".oof.tsv",            sep="\t", index=False)
    gene_metrics_df.to_csv(str(out_prefix) + ".gene_metrics.tsv", sep="\t", index=False)
    stratified_df.to_csv(  str(out_prefix) + ".stratified.tsv",   sep="\t", index=False)

    plot_performance(summary_df, str(out_prefix) + ".performance.png")
    _chance_row = stratified_df.loc[stratified_df["n_cryptic_bin"] == "all"]
    _chance_p1  = float(_chance_row["chance_mrr"].iloc[0]) if len(_chance_row) else None
    _chance_mrr = _chance_p1
    plot_top1_acc(   summary_df, str(out_prefix) + ".top1_acc.png", chance_p1=_chance_p1)
    plot_mrr(        summary_df, str(out_prefix) + ".mrr.png",      chance_mrr=_chance_mrr)
    plot_coefficients(coef_df,   str(out_prefix) + ".coefficients.png")

    print_cols = [
        "model", "n_features",
        "auc", "mrr", "top1_acc", "within_gene_auc",
        "within_gene_acc", "n_paired_genes",
    ]
    print_cols = [c for c in print_cols if c in summary_df.columns]
    print(
        summary_df
        .sort_values("mrr", ascending=False, na_position="last")
        [print_cols]
        .to_string(index=False)
    )

    if not stratified_df.empty and not summary_df.empty:
        best_model = summary_df.sort_values("mrr", ascending=False).iloc[0]["model"]
        print(f"\n── Stratified metrics for best MRR model: {best_model} ──")
        strat_best = stratified_df[stratified_df["model"] == best_model]
        print(strat_best.to_string(index=False))

    print(f"\n[OK] wrote {out_prefix}.summary.tsv")
    print(f"[OK] wrote {out_prefix}.coefs.tsv")
    print(f"[OK] wrote {out_prefix}.oof.tsv")
    print(f"[OK] wrote {out_prefix}.gene_metrics.tsv")
    print(f"[OK] wrote {out_prefix}.stratified.tsv")
    print(f"[OK] wrote {out_prefix}.performance.png")
    print(f"[OK] wrote {out_prefix}.top1_acc.png")
    print(f"[OK] wrote {out_prefix}.mrr.png")
    print(f"[OK] wrote {out_prefix}.coefficients.png")


if __name__ == "__main__":
    main()


