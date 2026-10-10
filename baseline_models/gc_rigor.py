#!/usr/bin/env python3
# rigorously test whether upstream gc content differentiates canonical from competitor peaks
import argparse, sys, math
import numpy as np, pandas as pd

def auc_mannwhitney(scores, labels):
    pos=scores[labels==1]; neg=scores[labels==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    alls=np.concatenate([pos,neg]); order=alls.argsort()
    ranks=np.empty(len(alls)); ranks[order]=np.arange(1,len(alls)+1)
    _,inv,cnt=np.unique(alls,return_inverse=True,return_counts=True)
    s=pd.Series(alls); ranks=s.rank(method="average").to_numpy()
    Rpos=ranks[:len(pos)].sum()
    U=Rpos-len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))

def auprc(scores, labels):
    order=scores.argsort()[::-1]; y=labels[order]
    tp=np.cumsum(y); fp=np.cumsum(1-y)
    prec=tp/(tp+fp); rec=tp/max(y.sum(),1)
    return float(np.trapezoid(prec, rec)) if hasattr(np,"trapezoid") else float(np.trapz(prec,rec))

def gc_at(X, lo, hi, cleave):
    a0=max(0,cleave-lo); a1=max(0,cleave-hi)
    sub=X[:,a0:a1,:]
    gc=sub[:,:,1]+sub[:,:,2]
    has=sub.sum(2)>0
    num=(gc*has).sum(1); den=has.sum(1)
    return np.where(den>0, num/np.maximum(den,1), np.nan)

def urich_at(X, lo, hi, cleave):
    a0=max(0,cleave-lo); a1=max(0,cleave-hi)
    sub=X[:,a0:a1,:]; u=sub[:,:,3]; has=sub.sum(2)>0
    num=(u*has).sum(1); den=has.sum(1)
    return np.where(den>0, num/np.maximum(den,1), np.nan)

def within_gene_p1(meta, score):
    m=meta.copy(); m["s"]=score; hit=[]; floor=[]
    for g,gg in m.groupby("gene",sort=False):
        s=gg["s"].to_numpy(); c=gg["is_canonical"].to_numpy().astype(bool)
        if np.isnan(s).all(): continue
        top=int(np.nanargmax(s)); hit.append(int(c[top])); floor.append(1.0/len(gg))
    return np.array(hit), np.array(floor)

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",required=True);  ap.add_argument("--test_meta",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--out",default="gc_rigor")
    a=ap.parse_args()

    X=np.load(a.test_npz)["X"].astype(np.float32); m=pd.read_parquet(a.test_meta).reset_index(drop=True)
    cleave=a.upstream_nt
    y=m["is_canonical"].to_numpy().astype(int)
    rng=np.random.default_rng(0)
    print(f"[data] {len(X):,} test peaks, canonical rate {y.mean():.3f}",file=sys.stderr)

    gc=gc_at(X,400,250,cleave); ok=~np.isnan(gc)
    g=gc[ok]; yy=y[ok]
    auc=auc_mannwhitney(g,yy); aup=auprc(g,yy)
    nulls=[auc_mannwhitney(g,rng.permutation(yy)) for _ in range(200)]
    p_perm=(np.sum(np.array(nulls)>=auc)+1)/(len(nulls)+1)
    bs=[]
    for _ in range(300):
        idx=rng.integers(0,len(g),len(g)); bs.append(auc_mannwhitney(g[idx],yy[idx]))
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f"\n=== GC (-400/-250) as canon-vs-comp classifier (TEST) ===",file=sys.stderr)
    print(f"  AUC = {auc:.3f}  [{lo:.3f},{hi:.3f}]   (0.5=useless, 1.0=perfect)",file=sys.stderr)
    print(f"  AUPRC = {aup:.3f}   (baseline = canonical rate = {yy.mean():.3f})",file=sys.stderr)
    print(f"  permutation p (AUC vs shuffled labels) = {p_perm:.3f}",file=sys.stderr)
    print(f"  null AUC mean = {np.mean(nulls):.3f}",file=sys.stderr)

    gcfull=gc_at(X,400,250,cleave)
    hit,floor=within_gene_p1(m.assign(_g=gcfull).rename(columns={}), gcfull)
    p,plo,phi=wilson(int(hit.sum()),len(hit))
    print(f"\n=== within-gene ranking by GC (the reranking-relevant test) ===",file=sys.stderr)
    print(f"  P@1 (argmax GC picks canonical) = {p:.3f} [{plo:.3f},{phi:.3f}]  (n={len(hit)} genes)",file=sys.stderr)
    print(f"  random floor (mean 1/n)         = {floor.mean():.3f}",file=sys.stderr)
    print(f"  GC P@1 - floor = {p-floor.mean():+.3f}   {'(GC ranks above chance)' if p>floor.mean() else '(GC no better than chance)'}",file=sys.stderr)

    print(f"\n=== AUC of GC by upstream window position (scan) ===",file=sys.stderr)
    scan=[]
    for center_up in range(50, 1000, 50):
        lo_,hi_=center_up+50, center_up-50
        gci=gc_at(X,lo_,max(hi_,1),cleave); o=~np.isnan(gci)
        au=auc_mannwhitney(gci[o],y[o]); scan.append((-center_up,au))
    sc=pd.DataFrame(scan,columns=["rel_center","auc"])
    sc.to_csv(f"{a.out}_position_auc.tsv",sep="\t",index=False)
    best=sc.iloc[(sc.auc-0.5).abs().idxmax()]
    for _,r in sc.iterrows():
        bar="#"*int(abs(r.auc-0.5)*200)
        print(f"  rel {int(r.rel_center):>5}: AUC {r.auc:.3f} {bar}",file=sys.stderr)
    print(f"  best-separating window: rel {int(best.rel_center)} (AUC {best.auc:.3f})",file=sys.stderr)

    ur=urich_at(X,200,100,cleave); o=~np.isnan(ur)
    auc_u=auc_mannwhitney(ur[o],y[o])
    hit_u,_=within_gene_p1(m, -ur)
    pu,_,_=wilson(int(hit_u.sum()),len(hit_u))
    print(f"\n=== competitor U-richness (-200/-100) for comparison ===",file=sys.stderr)
    print(f"  AUC (U marks canonical) = {auc_u:.3f}  (expect <0.5; U marks competitor)",file=sys.stderr)
    print(f"  |AUC-0.5| for U-rich = {abs(auc_u-0.5):.3f}  vs  GC = {abs(auc-0.5):.3f}",file=sys.stderr)
    print(f"  within-gene P@1 ranking by LOW U-richness = {pu:.3f}",file=sys.stderr)

    print(f"\n  VERDICT: AUC near 0.5 and P@1 near floor => GC is a WEAK differentiator.",file=sys.stderr)
    print(f"  AUC clearly >0.5 (say >0.6) and P@1 clearly above floor => GC is meaningful.",file=sys.stderr)
    print(f"[out] {a.out}_position_auc.tsv",file=sys.stderr)

if __name__=="__main__": main()