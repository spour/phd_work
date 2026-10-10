#!/usr/bin/env python3
# fill nans in the splice columns of the augmented feature tsvs so the randomforest
import argparse, os, sys, json, shutil
import numpy as np, pandas as pd

MIN_FEATS={"spl_interv_me5_max","spl_local_me5","spl_term_3ss","spl_upstream_max_me3",
           "spl_comp_up_3ss","spl_comp_dn_3ss","spl_term_3ss_abs"}
DIST_FEATS={"spl_dist_nearest_5ss","spl_dist_up_5ss","spl_dist_dn_5ss"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feat_dir",required=True)
    ap.add_argument("--out_dir",default=None,help="default: in place (with .bak)")
    ap.add_argument("--dist_sentinel",type=float,default=2000.0)
    ap.add_argument("--splits",default="pos_train,neg_train,pos_val,neg_val,pos_test,neg_test")
    a=ap.parse_args()
    splits=a.splits.split(",")
    out_dir=a.out_dir or a.feat_dir
    os.makedirs(out_dir,exist_ok=True)

    # 1) derive fill values from train splits only
    train_files=[s for s in splits if "train" in s]
    cols=None; train_df=[]
    for s in train_files:
        df=pd.read_csv(f"{a.feat_dir}/{s}.tsv",sep="\t",index_col=0)
        train_df.append(df)
    T=pd.concat(train_df,axis=0)
    spl=[c for c in T.columns if c.startswith("spl_")]
    fill={}
    for c in spl:
        if c in MIN_FEATS:
            v=np.nanmin(T[c].to_numpy()) if np.isfinite(T[c]).any() else 0.0
            fill[c]=float(v)
        elif c in DIST_FEATS:
            fill[c]=float(a.dist_sentinel)
        else:
            fill[c]=0.0
    json.dump(fill,open(f"{out_dir}/splice_impute_values.json","w"),indent=2)
    print(f"[impute] derived {len(fill)} fill values from train; "
          f"MIN-imputed={sorted(MIN_FEATS & set(spl))}",file=sys.stderr)

    # 2) apply to every split, in train-derived values
    for s in splits:
        p=f"{a.feat_dir}/{s}.tsv"
        df=pd.read_csv(p,sep="\t",index_col=0)
        before=df[spl].isna().sum().sum()
        for c in spl:
            if c in df.columns: df[c]=df[c].fillna(fill[c])
        after=df[spl].isna().sum().sum()
        resid=df.isna().sum().sum()
        outp=f"{out_dir}/{s}.tsv"
        if out_dir==a.feat_dir and not os.path.exists(p+".bak"):
            shutil.copy(p,p+".bak")
        df.to_csv(outp,sep="\t")
        print(f"  {s}: splice NaN {before} -> {after}; total residual NaN={resid}",file=sys.stderr)
        if resid>0:
            print(f"   [WARN] {s} still has {resid} non-splice NaN -- check PWM/kmer cols!",file=sys.stderr)
    print(f"[out] imputed TSVs in {out_dir} (originals -> *.bak); values in splice_impute_values.json",file=sys.stderr)

if __name__=="__main__": main()