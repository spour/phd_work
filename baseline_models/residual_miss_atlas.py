#!/usr/bin/env python3
# for contested test genes, characterize the residual errors of the splice reranker vs
import argparse, sys, math
import numpy as np, pandas as pd

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def load_atlas_bed(bed):
    df=pd.read_csv(bed,sep="\t",header=None,
                   names=["chrom","start","end","cid","tpm","strand","supp",
                          "nprot","tpm2","ann","pas"],usecols=range(11))
    df["chrom"]=df["chrom"].astype(str)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"])
    pos1=df["cid"].astype(str).str.split(":").str[1]
    df["pos"]=pd.to_numeric(pos1,errors="coerce")-1
    df=df.dropna(subset=["pos"])
    by={}
    for c,gg in df.groupby("chrom"):
        order=np.argsort(gg["pos"].to_numpy())
        by[c]=(gg["pos"].to_numpy()[order], gg["ann"].to_numpy()[order])
    print(f"[atlas] {len(df):,} sites over {len(by)} chroms",file=sys.stderr)
    return by

def nearest(chrom,pos,by,tol):
    if chrom not in by: return (None,np.inf)
    P,A=by[chrom]; i=np.searchsorted(P,pos)
    best=(None,np.inf)
    for j in (i-1,i):
        if 0<=j<len(P):
            d=abs(P[j]-pos)
            if d<best[1]: best=(A[j],d)
    return best if best[1]<=tol else (None,np.inf)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scored",required=True,help="rerank_test_scored.parquet")
    ap.add_argument("--atlas_bed",required=True)
    ap.add_argument("--tol",type=int,default=100)
    ap.add_argument("--out",default="residual_miss")
    a=ap.parse_args()

    df=pd.read_parquet(a.scored)
    need={"gene","chrom","center","rf_prob","is_canonical","splice_score"}
    if not need.issubset(df.columns):
        sys.exit(f"[err] scored file missing {need-set(df.columns)}; has {list(df.columns)}")
    sz=df.groupby("gene").size(); contested=set(sz[sz>1].index)
    df=df[df.gene.isin(contested)].reset_index(drop=True)
    atlas=load_atlas_bed(a.atlas_bed)

    rows=[]
    for g,gg in df.groupby("gene",sort=False):
        gg=gg.reset_index(drop=True)
        i_rr=int(np.argmax(gg["splice_score"].to_numpy()))
        i_rf=int(np.argmax(gg["rf_prob"].to_numpy()))
        for tag,i in [("reranker",i_rr),("cpa",i_rf)]:
            correct=bool(gg["is_canonical"].iloc[i])
            ann,dist=nearest(gg["chrom"].iloc[i],int(gg["center"].iloc[i]),atlas,a.tol)
            rows.append(dict(gene=g,method=tag,correct=correct,
                             on_atlas=(ann is not None),atlas_ann=ann if ann else "none",
                             dist=int(dist) if np.isfinite(dist) else -1))
    R=pd.DataFrame(rows); R.to_csv(f"{a.out}.tsv",sep="\t",index=False)
    n_genes=R.gene.nunique()
    print(f"\n=== contested test genes: {n_genes} ===",file=sys.stderr)
    for method in ("cpa","reranker"):
        m=R[R.method==method]; wrong=m[~m.correct]
        nw=len(wrong); on=int(wrong.on_atlas.sum())
        p,lo,hi=wilson(on,nw)
        corr=int(m.correct.sum())
        print(f"\n[{method}] correct {corr}/{len(m)} = {corr/len(m):.3f}; "
              f"WRONG picks: {nw}",file=sys.stderr)
        print(f"  of wrong picks, on a real atlas site (<= {a.tol}nt): {on}/{nw} = "
              f"{p:.3f} [{lo:.3f},{hi:.3f}]",file=sys.stderr)
        br=wrong[wrong.on_atlas].atlas_ann.value_counts()
        if len(br):
            print(f"  annotation of wrong-but-real picks: "
                  + ", ".join(f"{k}={v}({v/on:.2f})" for k,v in br.items()),file=sys.stderr)
    cw=R[(R.method=="cpa")&(~R.correct)]; rw=R[(R.method=="reranker")&(~R.correct)]
    pc,_,_=wilson(int(cw.on_atlas.sum()),len(cw)); pr,_,_=wilson(int(rw.on_atlas.sum()),len(rw))
    print(f"\n[head-to-head] wrong picks on a real atlas APA site: "
          f"cpa {pc:.3f}  vs  reranker {pr:.3f}  (delta {pr-pc:+.3f})",file=sys.stderr)
    print(f"  -> if reranker > cpa, the reranker's residual errors are real alternative "
          f"APA ends (candidate cell-type-specific / next-best), not noise.",file=sys.stderr)
    # paired: restrict to genes the reranker got wrong
    # picks on those same genes (apples-to-apples)
    rr_wrong_genes=set(R[(R.method=="reranker")&(~R.correct)].gene)
    sub=R[R.gene.isin(rr_wrong_genes)]
    cw=sub[sub.method=="cpa"]; rw=sub[sub.method=="reranker"]
    pc,lc,hc=wilson(int(cw.on_atlas.sum()),len(cw))
    pr,lr,hr=wilson(int(rw.on_atlas.sum()),len(rw))
    print(f"\n[PAIRED on {len(rr_wrong_genes)} genes the reranker missed] "
          f"fraction of picks on a real atlas APA site:",file=sys.stderr)
    print(f"  cpa pick      {pc:.3f} [{lc:.3f},{hc:.3f}]",file=sys.stderr)
    print(f"  reranker pick {pr:.3f} [{lr:.3f},{hr:.3f}]   delta {pr-pc:+.3f}",file=sys.stderr)
    cwi=cw.set_index("gene")["on_atlas"]; rwi=rw.set_index("gene")["on_atlas"]
    common=cwi.index.intersection(rwi.index)
    both=int((cwi[common]&rwi[common]).sum()); neither=int((~cwi[common]&~rwi[common]).sum())
    only_c=int((cwi[common]&~rwi[common]).sum()); only_r=int((~cwi[common]&rwi[common]).sum())
    print(f"  per-gene: both-on-atlas={both}  cpa-only={only_c}  reranker-only={only_r}  "
          f"neither={neither}",file=sys.stderr)
    print(f"  (reranker-only > cpa-only would mean the reranker recovers real sites cpa "
          f"misses; the reverse means reranker drifts to off-atlas/suppressed sites)",file=sys.stderr)
    drift=int((cwi[common]&~rwi[common]).sum())
    print(f"  reranker DRIFT to off-atlas (cpa hit a real site, reranker did not): "
          f"{drift}/{len(common)} = {drift/len(common):.3f}",file=sys.stderr)
    # fair comparison: only genes where the reranker moved the pick off cpa's
    sc=pd.read_parquet(a.scored)
    sz2=sc.groupby("gene").size(); sc=sc[sc.gene.isin(set(sz2[sz2>1].index))]
    changed=[]; same=0
    rr_atlas_changed=[]; cpa_atlas_changed=[]
    for g,gg in sc.groupby("gene",sort=False):
        gg=gg.reset_index(drop=True)
        i_rr=int(np.argmax(gg["splice_score"].to_numpy()))
        i_rf=int(np.argmax(gg["rf_prob"].to_numpy()))
        if i_rr==i_rf:
            same+=1; continue
        changed.append(g)
        a_rr=nearest(gg["chrom"].iloc[i_rr],int(gg["center"].iloc[i_rr]),atlas,a.tol)[0] is not None
        a_rf=nearest(gg["chrom"].iloc[i_rf],int(gg["center"].iloc[i_rf]),atlas,a.tol)[0] is not None
        rr_atlas_changed.append(a_rr); cpa_atlas_changed.append(a_rf)
    nch=len(changed)
    rr=np.array(rr_atlas_changed); cp=np.array(cpa_atlas_changed)
    prr,lrr,hrr=wilson(int(rr.sum()),nch); pcp,lcp,hcp=wilson(int(cp.sum()),nch)
    print(f"\n[FAIR: genes where reranker MOVED the pick off cpa, n={nch}; "
          f"{same} genes unchanged and excluded]",file=sys.stderr)
    print(f"  where the reranker MOVED TO is on atlas: {prr:.3f} [{lrr:.3f},{hrr:.3f}]",file=sys.stderr)
    print(f"  where cpa WOULD have gone is on atlas:   {pcp:.3f} [{lcp:.3f},{hcp:.3f}]   "
          f"delta {prr-pcp:+.3f}",file=sys.stderr)
    print(f"  CONFOUND: cpa picks the max-RF window, which correlates with atlas membership",file=sys.stderr)
    print(f"  (strong RF == strong cleavage grammar == more likely observed/cleaved). So cpa is",file=sys.stderr)
    print(f"  atlas-biased BY CONSTRUCTION; a lower reranker atlas rate may reflect that bias,",file=sys.stderr)
    print(f"  OR the splice signal genuinely favoring complete-but-silent (off-atlas) sites.",file=sys.stderr)
    print(f"  If reranker-moved-to atlas rate >= cpa here, the prior holds; if <, the reranker",file=sys.stderr)
    print(f"  drifts toward suppressed sites (consistent with the donor-strength suppression result).",file=sys.stderr)
    pd.DataFrame([
        dict(which="RF would pick", on_atlas_rate=pcp, n_changed=nch),
        dict(which="reranker moves to", on_atlas_rate=prr, n_changed=nch),
    ]).to_csv(f"{a.out}_fair.tsv",sep="\t",index=False)
    print(f"[out] {a.out}.tsv")

if __name__=="__main__": main()