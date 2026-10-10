#!/usr/bin/env python3
# does the reranking signal exist in raw site values
import argparse, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

TEST={"chr18","chr19","chr20","chr21","chr22","chrX"}; EXCL=TEST|{"chr3","chr13"}
WG=["upstream_mean_u1__wg","pas_prox_u1__wg","upstream_max_me3__wg","term_3ss__wg"]
AX={ "upstream_mean_u1__wg":"upstream_mean_u1","pas_prox_u1__wg":"pas_prox_u1",
     "upstream_max_me3__wg":"upstream_max_me3","term_3ss__wg":"term_3ss"}

def wilson(k,n,z=1.96):
    if n==0: return (0,0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--peak_table",required=True); ap.add_argument("--gene_baselines",required=True)
    a=ap.parse_args()
    df=pd.read_parquet(a.peak_table)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"].astype(str))
    base=pd.read_csv(a.gene_baselines,sep="\t").drop_duplicates("gene",keep="first").set_index("gene")

    raw_cols=[]
    for f in WG:
        ax=AX[f]; bcol=f"baseline_{ax}"; rc=f"site_{ax}"
        df[rc]=df[f].fillna(0.0)+df.gene.map(base[bcol]) if bcol in base.columns else df[f]
        raw_cols.append(rc)
    for f in WG: df[f]=df[f].fillna(0.0)
    df[raw_cols]=df[raw_cols].fillna(0.0)

    def cmarg(g):
        c=g.loc[g.in_zone==1,"rf_prob"]; i=g.loc[g.in_zone==0,"rf_prob"]
        return float(c.max()-i.max()) if len(c) and len(i) else np.nan
    te=df[df.chrom.isin(TEST)].copy()
    cm=te.groupby("gene").apply(cmarg,include_groups=False)
    contested=set(cm[cm<=0].index)
    tec=te[te.gene.isin(contested)]

    def within_gene_winrate(cols):
        out={}
        for f in cols:
            wins=0; n=0
            for gene,g in tec.groupby("gene",sort=False):
                if g.in_zone.sum()==0: continue
                v=g[f].to_numpy(); iz=g.in_zone.to_numpy()
                if not np.isfinite(v).any(): continue
                wins+=int(iz[int(np.argmax(v))]==1); n+=1
            out[f]=(wins,n)
        return out

    def model_recovery(cols):
        tr=df[~df.chrom.isin(EXCL)]
        clf=LogisticRegression(max_iter=1000,class_weight="balanced").fit(
            tr[cols].to_numpy(), tr["in_zone"].astype(int).to_numpy())
        t=te.copy(); t["sp"]=clf.predict_proba(t[cols].to_numpy())[:,1]
        hit=0; n=0
        for gene,g in t[t.gene.isin(contested)].groupby("gene",sort=False):
            if g.in_zone.sum()==0: continue
            rf=g.rf_prob.to_numpy(); sp=g.sp.to_numpy(); iz=g.in_zone.to_numpy()
            crf=float(rf[iz==1].max()); cand=np.where((iz==1)|(rf>=crf))[0]
            hit+=int(iz[cand[int(np.argmax(sp[cand]))]]==1); n+=1
        return hit,n,clf.coef_[0]

    def null_recovery():
        rng=np.random.default_rng(0); hit=0; n=0
        for gene,g in tec.groupby("gene",sort=False):
            if g.in_zone.sum()==0: continue
            rf=g.rf_prob.to_numpy(); iz=g.in_zone.to_numpy()
            crf=float(rf[iz==1].max()); cand=np.where((iz==1)|(rf>=crf))[0]
            hit+=int(iz[cand[rng.integers(len(cand))]]==1); n+=1
        return hit,n

    print("\n=== WITHIN-GENE: does the canonical have the TOP value among its gene's candidates? ===")
    print("  (chance ~ 1/median_contest_size, not 0.5; contested genes only)")
    print(f"  {'feature':<26}{'centered(wg)':>14}{'raw(site)':>12}")
    wg_w=within_gene_winrate(WG); rw_w=within_gene_winrate(raw_cols)
    for f,rc in zip(WG,raw_cols):
        kw,nw=wg_w[f]; kr,nr=rw_w[rc]
        print(f"  {AX[f]:<26}{kw}/{nw} ({kw/nw:.3f}){'':>3}{kr}/{nr} ({kr/nr:.3f})")

    print("\n=== TRAINED MODEL: contested recovery at cap=all (splice-among) ===")
    hk,nk,cw=model_recovery(WG); pr,lo,hi=wilson(hk,nk)
    hr,nr,cr=model_recovery(raw_cols); pr2,lo2,hi2=wilson(hr,nr)
    hn,nn=null_recovery(); pn,lon,hin=wilson(hn,nn)
    print(f"  centered __wg : {pr:.3f} [{lo:.3f},{hi:.3f}]   coef={np.round(cw,3)}")
    print(f"  raw site      : {pr2:.3f} [{lo2:.3f},{hi2:.3f}]   coef={np.round(cr,3)}")
    print(f"  NULL (random pick among competitors): {pn:.3f} [{lon:.3f},{hin:.3f}]")
    print("\n  READ: the NULL is the real baseline on contested genes (pick at random among")
    print("  the at-or-above set). If centered/raw recovery is NOT clearly above the NULL,")
    print("  there is no genuine reranking signal. If raw >> null but centered ~ null, the")
    print("  centering destroyed a real local signal. If both ~ null, splicing does not")
    print("  rerank the true end here.")

if __name__=="__main__": main()