#!/usr/bin/env python3
# are our contested genes enriched for pcpa in oh 2017

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcpa", required=True)
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--q", type=float, default=0.01,
                    help="q-value cutoff for both of Oh's tests")
    ap.add_argument("--gene-col", default=None,
                    help="column in our peak table holding the gene symbol")
    ap.add_argument("--out", default="pcpa_gene_test")
    args = ap.parse_args()

    oh = pd.read_csv(args.pcpa, sep="\t", low_memory=False)
    need = ["gene_name", "mRNA_slope_qvalue", "terminalExon_decrease_qvalue"]
    miss = [c for c in need if c not in oh.columns]
    if miss:
        sys.exit(f"missing {miss} in {args.pcpa}. Has: {list(oh.columns)}")
    for c in need[1:]:
        oh[c] = pd.to_numeric(oh[c], errors="coerce")
    oh["pcpaed"] = (oh["mRNA_slope_qvalue"] <= args.q) & \
                   (oh["terminalExon_decrease_qvalue"] <= args.q)
    oh["gene_name"] = oh["gene_name"].astype(str).str.upper()
    oh = oh.drop_duplicates("gene_name")
    print(f"Oh 2017: {len(oh):,} genes tested, {int(oh['pcpaed'].sum()):,} PCPAed "
          f"at q<={args.q} ({100*oh['pcpaed'].mean():.1f}%)")
    print("  Oh reports 3,590 PCPAed of 9,744 expressed at 8 h. If the numbers")
    print("  above are far off, the q cutoff or the file is not the 8 h one.")

    peaks = (pd.read_parquet(args.peaks) if args.peaks.endswith(".parquet")
             else pd.read_csv(args.peaks, sep="\t", low_memory=False))
    gcol = args.gene_col or ("gene_name" if "gene_name" in peaks.columns else "gene")
    if gcol not in peaks.columns:
        cols = list(peaks.columns)[:20]
        sys.exit(f"no gene column. Tried '{gcol}'. Has: {cols}")
    peaks[gcol] = peaks[gcol].astype(str).str.upper().str.replace(r"\..*$", "",
                                                                  regex=True)

    per_gene = (peaks.groupby(gcol)
                .agg(n_peaks=("is_canonical", "size"),
                     n_competitors=("is_canonical", lambda x: int((~x.astype(bool)).sum())))
                .reset_index().rename(columns={gcol: "gene_name"}))
    print(f"ours: {len(per_gene):,} contested genes, median "
          f"{per_gene['n_competitors'].median():.0f} competitors per gene")

    m = oh.merge(per_gene, on="gene_name", how="left")
    m["contested"] = m["n_peaks"].notna()
    n_match = int(m["contested"].sum())
    print(f"\noverlap with Oh's universe: {n_match:,} of {len(per_gene):,} "
          f"contested genes ({100*n_match/max(len(per_gene),1):.1f}%)")
    if n_match < 500:
        print("  Low. Our gene IDs are probably not symbols. Pass --gene-col,")
        print("  or map Ensembl IDs to symbols first.")

    tab = pd.crosstab(m["contested"], m["pcpaed"])
    print("\n[1] contested vs PCPAed, within Oh's tested genes")
    print(tab.to_string())
    if tab.shape == (2, 2):
        odds, p = fisher_exact(tab.to_numpy())
        r_con = m.loc[m["contested"], "pcpaed"].mean()
        r_non = m.loc[~m["contested"], "pcpaed"].mean()
        print(f"  PCPA rate: contested {r_con:.3f}  not contested {r_non:.3f}")
        print(f"  odds ratio {odds:.2f}   Fisher p {p:.3g}")
    else:
        print("  not a 2x2 table; check the merge")

    print("\n[2] PCPA rate by competitor count (dose response)")
    c = m[m["contested"]].copy()
    if len(c) > 100:
        try:
            c["bin"] = pd.qcut(c["n_competitors"], 5, duplicates="drop")
        except ValueError:
            c["bin"] = pd.cut(c["n_competitors"], 5)
        g = c.groupby("bin", observed=True).agg(
            n=("pcpaed", "size"), pcpa_rate=("pcpaed", "mean"),
            median_competitors=("n_competitors", "median"))
        print(g.round(3).to_string())
        rho, p = spearmanr(c["n_competitors"], c["pcpaed"].astype(int))
        print(f"  Spearman n_competitors vs PCPA: rho {rho:+.3f}  p {p:.3g}")

    print("\n[3] adjusted for expression")
    expr = next((x for x in ("CtrlAMO_RPKM", "cAMO_RPKM") if x in m.columns), None)
    if expr is None:
        print("  no control RPKM column found; skipped")
    else:
        m[expr] = pd.to_numeric(m[expr], errors="coerce")
        e = m.dropna(subset=[expr])
        u = mannwhitneyu(e.loc[e["contested"], expr],
                         e.loc[~e["contested"], expr], alternative="two-sided")
        print(f"  expression differs between groups: "
              f"contested median {e.loc[e['contested'], expr].median():.2f}  "
              f"other {e.loc[~e['contested'], expr].median():.2f}  p {u.pvalue:.3g}")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            X0 = np.log10(np.clip(e[expr].to_numpy(float), 1e-3, None)).reshape(-1, 1)
            X1 = np.column_stack([X0, e["contested"].astype(float)])
            y = e["pcpaed"].astype(int).to_numpy()
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            mdl.fit(X1, y)
            coef = mdl[-1].coef_[0]
            sd = X1[:, 1].std()
            print(f"  logistic: log10 expression coef {coef[0]:+.3f}, "
                  f"contested coef {coef[1]:+.3f}")
            print(f"  odds ratio for contested, expression held fixed: "
                  f"{np.exp(coef[1] / max(sd, 1e-9)):.2f}")
            print("  If this collapses toward 1, the raw enrichment was expression.")
        except ImportError:
            print("  sklearn missing; skipped the adjusted model")

    m.to_csv(f"{args.out}_per_gene.tsv", sep="\t", index=False)
    print(f"\nwrote {args.out}_per_gene.tsv")


if __name__ == "__main__":
    main()