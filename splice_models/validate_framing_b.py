#!/usr/bin/env python3
# is the framing-b result real?
import argparse, sys
import numpy as np, pandas as pd

SCORERS = [("cpa_only", "score_cpa_only"),
           ("splicing_only", "score_splicing_only"),
           ("learned_combiner", "score_learned")]


def per_gene_table(df):
    rows = []
    for g, sub in df.groupby("gene", sort=False):
        ncanon = int((sub["is_canonical"] == 1).sum())
        if ncanon == 0 or len(sub) < 2:
            continue
        rec = {"gene": g, "n_comp": len(sub) - 1, "chance": ncanon / len(sub)}
        for label, col in SCORERS:
            top = sub[col].idxmax()
            rec["hit_" + label] = int(sub.loc[top, "is_canonical"] == 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def boot_mean(v, B=4000, seed=0):
    v = np.asarray(v, float); n = len(v); rng = np.random.default_rng(seed)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, n, (B, n))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_delta(a, b, B=4000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a); rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def restrict_rf(df, T):
    if T <= 0:
        return df
    keep = []
    for g, sub in df.groupby("gene", sort=False):
        canon = sub[(sub["is_canonical"] == 1) & (sub["rf_prob"] >= T)]
        cryp = sub[(sub["is_canonical"] == 0) & (sub["rf_prob"] >= T)]
        if len(canon) and len(cryp):
            keep.append(pd.concat([canon, cryp]))
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored_tsv", required=True)
    ap.add_argument("--rf_threshold", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=4000)
    args = ap.parse_args()

    df = pd.read_csv(args.scored_tsv, sep="\t")
    df = restrict_rf(df, args.rf_threshold)
    pg = per_gene_table(df)
    if pg.empty:
        sys.exit("no evaluable genes after rf restriction")
    N = len(pg)
    print(f"[val] {N} test genes (rf_threshold={args.rf_threshold}), "
          f"median competitors={int(pg['n_comp'].median())}", file=sys.stderr)

    print(f"\n[val] OVERALL P@1 (95% CI over {args.bootstrap} gene bootstraps):", file=sys.stderr)
    print(f"  {'scorer':<18s}{'P@1':>8s}{'95% CI':>18s}", file=sys.stderr)
    ch_m, ch_lo, ch_hi = boot_mean(pg["chance"], args.bootstrap)
    print(f"  {'chance':<18s}{ch_m:>8.3f}   [{ch_lo:.3f}, {ch_hi:.3f}]", file=sys.stderr)
    for label, _ in SCORERS:
        m, lo, hi = boot_mean(pg["hit_" + label], args.bootstrap)
        print(f"  {label:<18s}{m:>8.3f}   [{lo:.3f}, {hi:.3f}]", file=sys.stderr)

    print(f"\n[val] PAIRED DELTAS (95% CI; CI excluding 0 = significant):", file=sys.stderr)
    for name, a, b in [("splicing - chance", pg["hit_splicing_only"], pg["chance"]),
                       ("splicing - cpa_only", pg["hit_splicing_only"], pg["hit_cpa_only"]),
                       ("combiner - splicing", pg["hit_learned_combiner"], pg["hit_splicing_only"])]:
        d, lo, hi = boot_delta(a, b, args.bootstrap)
        sig = "" if (lo <= 0 <= hi) else "  *"
        print(f"  {name:<22s}{d:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{sig}", file=sys.stderr)

    # 2: competitor-count stratification
    def stratum(c):
        return "1" if c == 1 else ("2-3" if c <= 3 else "4+")
    pg["stratum"] = pg["n_comp"].apply(stratum)
    print(f"\n[val] BY COMPETITOR COUNT (splicing must survive harder competitions):", file=sys.stderr)
    print(f"  {'competitors':<12s}{'n_genes':>8s}{'chance':>8s}{'cpa':>8s}"
          f"{'splicing':>10s}{'splice-chance':>15s}", file=sys.stderr)
    for s in ["1", "2-3", "4+"]:
        sub = pg[pg["stratum"] == s]
        if sub.empty:
            continue
        ch = sub["chance"].mean()
        cpa = sub["hit_cpa_only"].mean()
        sp, sp_lo, sp_hi = boot_mean(sub["hit_splicing_only"], args.bootstrap)
        d, dlo, dhi = boot_delta(sub["hit_splicing_only"], sub["chance"], args.bootstrap)
        sig = "" if (dlo <= 0 <= dhi) else " *"
        print(f"  {s:<12s}{len(sub):>8d}{ch:>8.3f}{cpa:>8.3f}{sp:>10.3f}"
              f"   {d:+.3f}[{dlo:+.3f},{dhi:+.3f}]{sig}", file=sys.stderr)

    print(f"\n[val] headline = splicing_only vs chance. if the delta CI excludes 0 across strata, "
          f"splicing genuinely selects the canonical among sequence-matched competitors. "
          f"note: scored at curated label centers, so this is discrimination among known "
          f"candidates, not genome-wide gene-end finding.", file=sys.stderr)


if __name__ == "__main__":
    main()
