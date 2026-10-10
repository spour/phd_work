#!/usr/bin/env python3
# train and evaluate distance-blind choice models for 3' end selection
import argparse, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

def read_chroms(p): return set(l.strip() for l in open(p) if l.strip())

# conditional logit (gene-stratified softmax)
def seg_softmax_ll(eta, starts, counts, true_pos):
    gmax=np.maximum.reduceat(eta,starts); gmax_row=np.repeat(gmax,counts)
    ex=np.exp(eta-gmax_row); gsum=np.add.reduceat(ex,starts)
    p=ex/np.repeat(gsum,counts); ll=eta[true_pos]-(np.log(gsum)+gmax)
    return p,ll
def fit_clogit(X,starts,counts,true_pos):
    y=np.zeros(len(X)); y[true_pos]=1.0; k=X.shape[1]
    def ng(b):
        p,ll=seg_softmax_ll(X@b,starts,counts,true_pos)
        return -ll.sum(), X.T@(p-y)
    return minimize(ng,np.zeros(k),jac=True,method="L-BFGS-B",options=dict(maxiter=500)).x

def prep_groups(df):
    df=df.sort_values("gene",kind="stable").reset_index(drop=True)
    codes=pd.factorize(df["gene"].to_numpy())[0]
    starts=np.r_[0,np.flatnonzero(np.diff(codes))+1]; counts=np.diff(np.r_[starts,len(df)])
    keep_r=[]; ks=[]; kc=[]; ktp=[]; cur=0; pig=[]
    it=df["is_true"].to_numpy()
    for s,c in zip(starts,counts):
        w=np.where(it[s:s+c]==1)[0]
        if len(w)>=1 and c>=2:
            tp=s+int(w[0])
            keep_r.extend(range(s,s+c)); ks.append(cur); kc.append(c); ktp.append(cur+int(w[0])); pig.append((cur,c)); cur+=c
    dfk=df.iloc[keep_r].reset_index(drop=True)
    return dfk,np.array(ks),np.array(kc),np.array(ktp),pig

def argmax_acc(df,score,pig):
    s=df[score].to_numpy(); it=df["is_true"].to_numpy(); hit=rnd=0
    for (st,c) in pig:
        sl=slice(st,st+c)
        hit+=int(it[sl][int(np.argmax(s[sl]))]==1); rnd+=it[sl].sum()/c
    return hit/len(pig), rnd/len(pig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--train_chroms", required=True); ap.add_argument("--test_chroms", required=True)
    ap.add_argument("--out_json", default=None)
    a=ap.parse_args()
    df=pd.read_parquet(a.features); df["chrom"]=df["chrom"].astype(str)
    trC=read_chroms(a.train_chroms); teC=read_chroms(a.test_chroms)

    BANNED={"gene","chrom","pos","strand","is_true"}
    feats=[c for c in df.columns if c not in BANNED]
    assert not any(("dist" in f or "length" in f or "pos" in f or "zone" in f) for f in feats), \
        "a distance/position feature leaked into the whitelist"
    for f in feats: df[f]=df[f].fillna(df[f].median())
    print(f"[features used] {feats}\n")

    tr=df[df.chrom.isin(trC)].copy(); te=df[df.chrom.isin(teC)].copy()
    trp,s_tr,c_tr,tp_tr,pig_tr=prep_groups(tr); tep,s_te,c_te,tp_te,pig_te=prep_groups(te)
    print(f"[data] train genes {len(pig_tr):,} | test genes {len(pig_te):,}")

    res={}
    if "rf_prob" in te.columns and te["rf_prob"].notna().any():
        acc,rnd=argmax_acc(tep,"rf_prob",pig_te)
        res["RF_baseline"]={"argmax_acc":acc,"random":rnd}
        print(f"[RF_baseline]  argmax acc = {acc:.3f}  (random {rnd:.3f})  <- the bar")

    mean=trp[feats].to_numpy().mean(0); sd=trp[feats].to_numpy().std(0); sd[sd==0]=1
    Xtr=(trp[feats].to_numpy()-mean)/sd; Xte=(tep[feats].to_numpy()-mean)/sd
    b=fit_clogit(Xtr,s_tr,c_tr,tp_tr)
    tep=tep.copy(); tep["clogit_score"]=Xte@b
    acc,rnd=argmax_acc(tep,"clogit_score",pig_te)
    res["clogit"]={"argmax_acc":acc,"random":rnd,"coef":dict(zip(feats,b.tolist()))}
    print(f"[clogit]       argmax acc = {acc:.3f}  (random {rnd:.3f})")
    print("   coefficients (standardized, sign = direction toward TRUE end):")
    for f,bi in sorted(zip(feats,b),key=lambda x:-abs(x[1])):
        print(f"     {f:<24}{bi:+.3f}")

    clf=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,
        max_leaf_nodes=63,l2_regularization=1.0,random_state=0)
    clf.fit(trp[feats].to_numpy(), trp["is_true"].to_numpy())
    tep["gbt_score"]=clf.predict_proba(tep[feats].to_numpy())[:,1]
    acc,rnd=argmax_acc(tep,"gbt_score",pig_te)
    res["gbt_choice"]={"argmax_acc":acc,"random":rnd}
    print(f"[gbt_choice]   argmax acc = {acc:.3f}  (random {rnd:.3f})")

    if "rf_prob" in df.columns and df["rf_prob"].notna().any():
        rf_idx=feats.index("rf_prob")
        b0=fit_clogit(Xtr[:,[rf_idx]],s_tr,c_tr,tp_tr)
        _,ll0=seg_softmax_ll(Xte[:,[rf_idx]]@b0,s_te,c_te,tp_te)
        _,ll1=seg_softmax_ll(Xte@b,s_te,c_te,tp_te)
        bits=(ll1.mean()-ll0.mean())/np.log(2)
        res["clogit_bits_vs_rf"]=bits
        print(f"[info] clogit adds {bits:+.4f} bits/gene over rf_prob-alone (held out)")

    rng=np.random.default_rng(0); tp_sh=[]
    for (st,c) in pig_tr: tp_sh.append(st+int(rng.integers(c)))
    bs=fit_clogit(Xtr,s_tr,c_tr,np.array(tp_sh))
    tep["sh"]=Xte@bs; acc_sh,rnd_sh=argmax_acc(tep,"sh",pig_te)
    res["shuffle_null_acc"]=acc_sh
    flag="OK ~random" if abs(acc_sh-rnd_sh)<0.02 else "LEAK SUSPECT"
    print(f"[shuffle null] clogit argmax acc = {acc_sh:.3f} (random {rnd_sh:.3f})  {flag}")

    if a.out_json:
        json.dump(res,open(a.out_json,"w"),indent=2); print(f"[done] {a.out_json}")

if __name__=="__main__":
    main()