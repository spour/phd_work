#!/usr/bin/env python3
# the point of this figure: internal (competitor) peaks are sequence-indistinguishable
import argparse, os, sys, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

SCORES = {
    "APARENT2": "aparent2_prob",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--old_scan_dir", required=True,
                    help="dir of OLD per-gene parquets carrying aparent2_prob etc.")
    ap.add_argument("--contested_only", action="store_true", default=True)
    ap.add_argument("--vicinity", type=int, default=250,
                    help="window (nt) for --vicinity_nonrf max")
    ap.add_argument("--vicinity_nonrf", action="store_true", default=False,
                    help="give non-RF scores their max within +/-vicinity nt (RF always exact; default: exact for all)")
    ap.add_argument("--n_boot", type=int, default=2000, help="bootstrap for AUROC/AUPRC CIs")
    ap.add_argument("--out", default="cpa_discrim")
    a = ap.parse_args()

    W = pd.read_parquet(a.shortlist)
    if a.contested_only:
        W = W[W.n_internal > 0].copy()
    genes = W.gene.unique()
    print(f"[data] {len(W):,} shortlist peaks over {len(genes):,} contested genes", file=sys.stderr)

    need_cols = ["window_start_genomic"] + list(SCORES.values())
    tol = a.vicinity
    merged = []
    miss_gene = 0; miss_row = 0
    for g, gg in W.groupby("gene", sort=False):
        p = os.path.join(a.old_scan_dir, f"{g}.parquet")
        if not os.path.exists(p):
            miss_gene += 1; continue
        d = pd.read_parquet(p, columns=[c for c in need_cols if c])
        dws = d["window_start_genomic"].to_numpy()
        arrs = {name: d[col].to_numpy() for name, col in SCORES.items()}
        gg = gg.copy()
        for name, col in SCORES.items():
            use_vic = (name == "APARENT2")
            vals = np.full(len(gg), np.nan)
            for i, w0 in enumerate(gg["window_start_genomic"].to_numpy()):
                if use_vic:
                    near = np.abs(dws - w0) <= tol
                    if near.any(): vals[i] = np.nanmax(arrs[name][near])
                else:
                    hit = np.flatnonzero(dws == w0)
                    if len(hit): vals[i] = arrs[name][hit[0]]
            gg[col] = vals
        miss_row += int(gg[SCORES["APARENT2"]].isna().sum())
        merged.append(gg)
    M = pd.concat(merged, ignore_index=True)
    print(f"[merge] {miss_gene} genes had no old parquet; "
          f"{miss_row:,} peaks missing APARENT2 after merge", file=sys.stderr)
    for c in SCORES.values():
        M[c] = pd.to_numeric(M[c], errors="coerce")
    M = M.dropna(subset=list(SCORES.values())).copy()
    y = M["is_canonical"].astype(int).to_numpy()
    base = y.mean()
    print(f"[label] {len(M):,} peaks; canonical prevalence {base:.4f} "
          f"(AUPRC baseline)", file=sys.stderr)

    from sklearn.metrics import roc_curve, precision_recall_curve
    rng = np.random.default_rng(0)
    rows = []; dist_rows = []; curve_rows = []
    for name, col in SCORES.items():
        s = M[col].to_numpy()
        auroc = roc_auc_score(y, s); auprc = average_precision_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        def thin(x, y2, k=300):
            if len(x) <= k: return x, y2
            idx = np.linspace(0, len(x)-1, k).astype(int); return x[idx], y2[idx]
        fx, tx = thin(fpr, tpr); rx, px = thin(rec, prec)
        for a_, b_ in zip(fx, tx): curve_rows.append(dict(score=name, curve="ROC", x=float(a_), y=float(b_)))
        for a_, b_ in zip(rx, px): curve_rows.append(dict(score=name, curve="PR",  x=float(a_), y=float(b_)))
        bo_roc = []; bo_prc = []
        n = len(y)
        for _ in range(a.n_boot):
            ix = rng.integers(0, n, n)
            if y[ix].sum() == 0 or y[ix].sum() == len(ix): continue
            bo_roc.append(roc_auc_score(y[ix], s[ix]))
            bo_prc.append(average_precision_score(y[ix], s[ix]))
        rows.append(dict(score=name, method=("exact(selection-baseline)" if name=="RF" else ("vicinity-max" if False else "exact")), auroc=round(auroc,4),
                         auroc_lo=round(np.percentile(bo_roc,2.5),4),
                         auroc_hi=round(np.percentile(bo_roc,97.5),4),
                         auprc=round(auprc,4),
                         auprc_lo=round(np.percentile(bo_prc,2.5),4),
                         auprc_hi=round(np.percentile(bo_prc,97.5),4),
                         auprc_baseline=round(base,4), n=n))
        for lab, mask in [("canonical", y==1), ("competitor", y==0)]:
            v = s[mask]
            if len(v) > 8000: v = rng.choice(v, 8000, replace=False)
            for x in v: dist_rows.append(dict(score=name, cls=lab, value=float(x)))

    R = pd.DataFrame(rows)
    R.to_csv(f"{a.out}_auroc_auprc.tsv", sep="\t", index=False)
    pd.DataFrame(dist_rows).to_csv(f"{a.out}_score_dist.tsv", sep="\t", index=False)
    pd.DataFrame(curve_rows).to_csv(f"{a.out}_curves.tsv", sep="\t", index=False)
    print(f"\n=== canonical-vs-competitor discrimination (contested peaks) ===", file=sys.stderr)
    print(R[["score","auroc","auroc_lo","auroc_hi","auprc","auprc_baseline"]].to_string(index=False), file=sys.stderr)
    print(f"\n[interpret] AUROC ~0.5 and AUPRC ~baseline ({base:.3f}) => the score cannot "
          f"tell canonical from competitor. Compare to Fig-1 CPA-vs-background AUROC ~0.99.", file=sys.stderr)
    print(f"[out] {a.out}_auroc_auprc.tsv, {a.out}_score_dist.tsv, {a.out}_curves.tsv", file=sys.stderr)

if __name__ == "__main__":
    main()