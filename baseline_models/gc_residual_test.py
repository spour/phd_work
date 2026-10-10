#!/usr/bin/env python3
# is the dominant-vs-cryptic signal "just gc"?
import argparse, sys
import numpy as np, pandas as pd
A,C,G,Tn=0,1,2,3

def auc_mw(s,y):
    s=np.asarray(s,float); y=np.asarray(y,int); ok=np.isfinite(s); s=s[ok]; y=y[ok]
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    r=pd.Series(np.concatenate([pos,neg])).rank(method="average").to_numpy()
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def frac(X, chans, lo, hi):
    sub=X[:,lo:hi,:]; sel=sub[:,:,chans].sum(2); has=sub.sum(2)>0
    num=(sel*has).sum(1); den=has.sum(1)
    return np.where(den>0,num/np.maximum(den,1),np.nan)

def region_gc(X, cl):
    L=X.shape[1]
    regions={"gc_all":(0,L),"gc_up":(max(0,cl-400),cl-50),"gc_core":(cl-50,cl+50),
             "gc_dn":(cl+50,min(L,cl+400)),"gc_far_up":(0,max(1,cl-400))}
    cols={}
    for nm,(lo,hi) in regions.items():
        if hi-lo>=5: cols[nm]=frac(X,[G,C],lo,hi)
    return pd.DataFrame(cols)

def builtin_features(X, cl):
    L=X.shape[1]
    f={}
    f["A_up_40"]=frac(X,[A],cl-40,cl-5)
    f["U_dn_13_30"]=frac(X,[Tn],cl+13,cl+30)
    f["AT_up_200"]=frac(X,[A,Tn],max(0,cl-200),cl-50)
    f["pur_up_300"]=frac(X,[A,G],max(0,cl-300),cl-50)
    return pd.DataFrame(f)

def matched_subset(y, gc, n_per=None, tol=0.01, rng=None):
    rng=rng or np.random.default_rng(0)
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    rng.shuffle(pos)
    neg_gc=gc[neg]; used=np.zeros(len(neg),bool)
    order=np.argsort(neg_gc)
    neg_sorted=neg[order]; neg_gc_sorted=neg_gc[order]; used_sorted=np.zeros(len(neg),bool)
    keep_pos=[]; keep_neg=[]
    for p in pos:
        g=gc[p]
        if not np.isfinite(g): continue
        lo=np.searchsorted(neg_gc_sorted,g-tol); hi=np.searchsorted(neg_gc_sorted,g+tol)
        cand=[i for i in range(lo,hi) if not used_sorted[i]]
        if not cand: continue
        pick=cand[len(cand)//2]
        used_sorted[pick]=True
        keep_pos.append(p); keep_neg.append(neg_sorted[pick])
    idx=np.array(keep_pos+keep_neg)
    return idx

def nested_auc_increment(base_df, add, y, n_boot=200, rng=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    rng=rng or np.random.default_rng(0)
    def cv_p(Xdf):
        X=Xdf.to_numpy(float)
        med=np.nanmedian(X,axis=0); X=np.where(np.isnan(X),med,X)
        return cross_val_predict(LogisticRegression(max_iter=1000),X,y,cv=5,method="predict_proba")[:,1]
    p_base=cv_p(base_df)
    full=base_df.copy()
    if isinstance(add,pd.DataFrame):
        for c in add.columns: full[c]=add[c].to_numpy()
    else:
        full["feat"]=np.asarray(add,float)
    p_full=cv_p(full)
    a_base=auc_mw(p_base,y); a_full=auc_mw(p_full,y)
    incs=[]
    n=len(y); idx=np.arange(n)
    for _ in range(n_boot):
        b=rng.choice(idx,n,replace=True)
        incs.append(auc_mw(p_full[b],y[b])-auc_mw(p_base[b],y[b]))
    incs=np.array(incs)
    return a_base,a_full,a_full-a_base,np.percentile(incs,2.5),np.percentile(incs,97.5)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",required=True); ap.add_argument("--meta",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--cnn_scores",default=None,help="tsv with a 'score' column aligned to meta row order (or site_id+score)")
    ap.add_argument("--subset",type=int,default=6000)
    ap.add_argument("--gc_tol",type=float,default=0.01)
    ap.add_argument("--n_boot",type=int,default=200)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",default="gc_residual")
    a=ap.parse_args()
    rng=np.random.default_rng(a.seed)
    X=np.load(a.npz)["X"].astype(np.float32); m=pd.read_parquet(a.meta).reset_index(drop=True)
    y=m["label"].to_numpy().astype(int); L=X.shape[1]; cl=a.upstream_nt
    if a.subset>0 and a.subset<len(X):
        pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
        npos=min(len(pos),a.subset//2); nneg=min(len(neg),a.subset-npos)
        sel=np.concatenate([rng.choice(pos,npos,replace=False),rng.choice(neg,nneg,replace=False)])
        rng.shuffle(sel); X=X[sel]; m=m.iloc[sel].reset_index(drop=True); y=y[sel]
    print(f"[data] {len(X):,} sites ({int((y==1).sum())} dominant, {int((y==0).sum())} cryptic)",file=sys.stderr)

    gc_df=region_gc(X,cl)
    gc_all=gc_df["gc_all"].to_numpy()

    print(f"\n=== (1) GC-ONLY FLOOR ===",file=sys.stderr)
    print(f"  global GC AUC          = {auc_mw(gc_all,y):.3f}",file=sys.stderr)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    Xg=gc_df.to_numpy(float); med=np.nanmedian(Xg,0); Xg=np.where(np.isnan(Xg),med,Xg)
    p_gc=cross_val_predict(LogisticRegression(max_iter=1000),Xg,y,cv=5,method="predict_proba")[:,1]
    print(f"  multi-region GC AUC    = {auc_mw(p_gc,y):.3f}   (regions: {list(gc_df.columns)})",file=sys.stderr)

    if a.cnn_scores:
        sc=pd.read_csv(a.cnn_scores,sep="\t")
        if "score" in sc.columns and len(sc)==len(m):
            feat=pd.DataFrame({"cnn_score":sc["score"].to_numpy()})
        elif {"site_id","score"}.issubset(sc.columns) and "name" in m.columns:
            mp=dict(zip(sc["site_id"],sc["score"]))
            feat=pd.DataFrame({"cnn_score":m["name"].map(mp).to_numpy()})
        else:
            sys.exit("[fatal] --cnn_scores needs a 'score' col aligned to meta rows, or site_id+score with meta 'name'")
        feat_name="cnn_score"
    else:
        feat=builtin_features(X,cl); feat_name="builtin("+",".join(feat.columns)+")"
        print(f"\n[note] no --cnn_scores; testing built-in interpretable features",file=sys.stderr)

    print(f"\n=== (2) GC-MATCHED SUBSET ===",file=sys.stderr)
    idx=matched_subset(y,gc_all,tol=a.gc_tol,rng=rng)
    if len(idx)<40:
        print(f"  too few matched pairs (tol={a.gc_tol}); loosen --gc_tol",file=sys.stderr)
    else:
        ym=y[idx]; gcm=gc_all[idx]
        print(f"  matched n={len(idx)} ({int((ym==1).sum())}/{int((ym==0).sum())}), "
              f"GC AUC on matched = {auc_mw(gcm,ym):.3f}  (should be ~0.5 if matching worked)",file=sys.stderr)
        for c in feat.columns:
            print(f"  {c:<16} AUC on GC-matched = {auc_mw(feat[c].to_numpy()[idx],ym):.3f}",file=sys.stderr)

    print(f"\n=== (3) SIGNAL BEYOND GC (nested: GC vs GC+feature) ===",file=sys.stderr)
    a_base,a_full,inc,lo,hi=nested_auc_increment(gc_df,feat,y,n_boot=a.n_boot,rng=rng)
    print(f"  GC-only AUC        = {a_base:.3f}",file=sys.stderr)
    print(f"  GC + {feat_name:<10} AUC = {a_full:.3f}",file=sys.stderr)
    if lo>0:        verdict="REAL signal beyond GC (CI excludes 0)"
    elif hi<0:      verdict="feature is WORSE than GC alone (CI below 0; likely noise)"
    else:           verdict="NOT distinguishable from GC (CI includes 0)"
    print(f"  increment          = {inc:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}",file=sys.stderr)

    pd.DataFrame([dict(gc_only_auc=a_base,gc_plus_feat_auc=a_full,increment=inc,ci_lo=lo,ci_hi=hi,
                       global_gc_auc=auc_mw(gc_all,y),feature=feat_name)]
                 ).to_csv(f"{a.out}_summary.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_summary.tsv",file=sys.stderr)

if __name__=="__main__": main()