#!/usr/bin/env python3
# head-to-head with the true end collapsed to one candidate
import argparse, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
TEST={"chr18","chr19","chr20","chr21","chr22","chrX"}; EXCL=TEST|{"chr3","chr13"}
WG=["upstream_mean_u1__wg","pas_prox_u1__wg","upstream_max_me3__wg","term_3ss__wg"]
def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def collapse(g):
    iz=g.in_zone.to_numpy(); rf=g.rf_prob.to_numpy(); sp=g.sp.to_numpy()
    izi=np.where(iz==1)[0]
    keep_canon=izi[int(np.argmax(rf[izi]))]
    drop=set(izi.tolist())-{keep_canon}
    keepmask=np.array([i not in drop for i in range(len(g))])
    rf2=rf[keepmask]; sp2=sp[keepmask]
    is_can=np.zeros(len(rf2),bool)
    new_idx=np.where(np.where(keepmask)[0]==keep_canon)[0][0]
    is_can[new_idx]=True
    return rf2,sp2,is_can

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--peak_table",required=True)
    ap.add_argument("--n_random",type=int,default=300); a=ap.parse_args()
    df=pd.read_parquet(a.peak_table)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"].astype(str))
    feats=[c for c in WG if c in df.columns]; df[feats]=df[feats].fillna(0.0)
    tr=df[~df.chrom.isin(EXCL)]
    clf=LogisticRegression(max_iter=1000,class_weight="balanced").fit(
        tr[feats].to_numpy(),tr["in_zone"].astype(int).to_numpy())
    te=df[df.chrom.isin(TEST)].copy(); te["sp"]=clf.predict_proba(te[feats].to_numpy())[:,1]
    rows=[]
    for gene,g in te.groupby("gene",sort=False):
        if g.in_zone.sum()==0: continue
        rf,sp,isc=collapse(g)
        rf_hit=int(isc[int(np.argmax(rf))])
        crf=rf[isc].max(); cand=np.where(isc|(rf>=crf))[0]; m=len(cand)
        ncan=int(isc[cand].sum())
        sp_hit=int(isc[cand[int(np.argmax(sp[cand]))]])
        rows.append((gene,rf_hit,sp_hit,m,ncan/m))
    R=pd.DataFrame(rows,columns=["gene","rf","splice","m","randp"])
    def block(sub,label):
        n=len(sub)
        prf,lrf,hrf=wilson(int(sub.rf.sum()),n); psp,lsp,hsp=wilson(int(sub.splice.sum()),n)
        randp=sub.randp.mean()
        sims=[ (np.random.default_rng(s).random(n)<sub.randp.to_numpy()).mean() for s in range(a.n_random)]
        rlo,rhi=np.percentile(sims,[2.5,97.5])
        print(f"  [{label}]  n={n}   mean set size m={sub.m.mean():.2f}")
        print(f"    RF-alone        {prf:.3f} [{lrf:.3f},{hrf:.3f}]")
        print(f"    random tiebreak {randp:.3f} [{rlo:.3f},{rhi:.3f}]")
        print(f"    splice tiebreak {psp:.3f} [{lsp:.3f},{hsp:.3f}]")
        v=("splice BEATS random (CIs disjoint)" if lsp>rhi else
           "splice WORSE than random (CIs disjoint)" if hsp<rlo else "splice ~ random (overlap)")
        print(f"    --> {v}")
    print("\n========== HEAD TO HEAD v2 (true end collapsed to ONE candidate) ==========")
    block(R,"ALL scorable test genes")
    print(); block(R[R.m>1],"CONTESTED genes (m>1)")
    easy=(R.m==1).sum(); print(f"\n  {easy}/{len(R)} genes have m=1 (RF already right; all rules score 1.0)")
if __name__=="__main__": main()