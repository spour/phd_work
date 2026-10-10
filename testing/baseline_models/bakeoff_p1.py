#!/usr/bin/env python3
# within-gene p@1 bake-off across feature sets, using the frozen
import argparse, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conditional_logit_gate import (logit, prep, fit_clogit, evaluate,
                                     contested_at, RF_TRAIN, RF_VAL, RF_TEST)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--raw", action="store_true", default=True,
                    help="use raw splice features (default); centering is irrelevant here")
    a = ap.parse_args()
    df = pd.read_parquet(a.peak_table); df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr"+df["chrom"])
    df["rf_logit"] = logit(df["rf_prob"])

    sp = "" if a.raw else "__wg"
    SPL = [f+sp for f in ["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt",
                          "local_me5","pas_prox_u1","dn5ss_has_strong","interv_me5_max"]]
    SPL = [c for c in SPL if c in df.columns]
    allcols = ["rf_logit"] + SPL
    for c in allcols: df[c] = df[c].fillna(0.0)

    me3 = "upstream_max_me3"+sp; t3 = "term_3ss"+sp
    SETS = {
        "cpa_only            ": ["rf_logit"],
        "splice_only(me3)    ": [me3],
        "splice_only(all8)   ": SPL,
        "cpa+me3             ": ["rf_logit", me3],
        "cpa+me3+term3ss     ": ["rf_logit", me3, t3],
        "cpa+all8splice      ": ["rf_logit"] + SPL,
    }

    train_chroms = RF_TRAIN | RF_VAL
    tr = df[df.chrom.isin(train_chroms)].copy(); te = df[df.chrom.isin(RF_TEST)].copy()
    trp,s_tr,c_tr,tp_tr,pig_tr = prep(tr, allcols)
    tep,s_te,c_te,tp_te,pig_te = prep(te, allcols)
    mean = trp[allcols].to_numpy().mean(0); sd = trp[allcols].to_numpy().std(0); sd[sd==0]=1
    Xtr = (trp[allcols].to_numpy()-mean)/sd; ytr = np.zeros(len(trp)); ytr[tp_tr]=1.0
    print(f"[data] train choice-genes {len(pig_tr):,}  test choice-genes {len(pig_te):,}\n")

    base_ll = None
    print(f"{'feature_set':<22}{'contested_P@1':>14}{'cont_random':>12}{'all_P@1':>9}{'cont_bits_vs_cpa':>18}")
    for name, feats in SETS.items():
        cols = [allcols.index(f) for f in feats]
        b = fit_clogit(Xtr[:,cols], s_tr, c_tr, tp_tr, ytr)
        bfull = np.zeros(len(allcols))
        for bi,ci in zip(b,cols): bfull[ci]=bi
        ll,p1,rnd,con = evaluate(tep, allcols, mean, sd, bfull, pig_te, contested_at)
        con = con.astype(bool)
        cll = ll[con].mean()
        if "cpa_only" in name: base_ll = cll
        bits = (cll - base_ll)/np.log(2) if base_ll is not None else 0.0
        print(f"{name:<22}{p1[con].mean():>14.3f}{rnd[con].mean():>12.3f}"
              f"{p1.mean():>9.3f}{bits:>+18.4f}")
    print("\nread: contested_P@1 = how often the model nails the true end among CPA-competent")
    print("      competitors. cpa_only is ~0 there by construction. cont_bits_vs_cpa = info")
    print("      each set adds over CPA-only on contested genes (held out).")

if __name__ == "__main__":
    main()