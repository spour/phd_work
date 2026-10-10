#!/usr/bin/env python3
# panel c (residual-miss) reports 4,709 overridden genes
import argparse, sys
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scored",required=True)
    a=ap.parse_args()
    df=pd.read_parquet(a.scored)
    print(f"[data] {len(df):,} peaks, {df.gene.nunique():,} genes",file=sys.stderr)

    geo_over=set(); geo_skip_nocanon=set()
    res_over=set()
    for g,gg in df.groupby("gene",sort=False):
        rf_c =int(gg["center"].to_numpy()[int(np.argmax(gg["rf_prob"].to_numpy()))])
        spl_c=int(gg["center"].to_numpy()[int(np.argmax(gg["splice_score"].to_numpy()))])
        overrode = rf_c!=spl_c
        if overrode: res_over.add(g)
        if not gg["is_canonical"].any():
            geo_skip_nocanon.add(g)
            continue
        if overrode: geo_over.add(g)

    print(f"\n[geometry def]  overrode & has-canonical : {len(geo_over):,}",file=sys.stderr)
    print(f"[residual def]  overrode (any gene)       : {len(res_over):,}",file=sys.stderr)
    print(f"[diff] genes overridden but NO canonical in shortlist: "
          f"{len(res_over & geo_skip_nocanon):,}",file=sys.stderr)
    only_res = res_over - geo_over
    print(f"[diff] in residual set but not geometry set: {len(only_res):,}",file=sys.stderr)
    exp = only_res & geo_skip_nocanon
    print(f"       of which explained by missing-canonical: {len(exp):,}",file=sys.stderr)
    print(f"       unexplained (investigate): {len(only_res - exp):,}",file=sys.stderr)
    print(f"\n[conclusion] the geometry panel drops genes whose shortlist has no "
          f"canonical peak (can't compute distance-to-canonical); residual-miss keeps "
          f"them. The difference should equal the missing-canonical count.",file=sys.stderr)

if __name__=="__main__": main()