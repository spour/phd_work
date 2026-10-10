#!/usr/bin/env python3
# how much does splice reranking help as a function of (1) gene length
import argparse, sys, math, re
import numpy as np, pandas as pd

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def load_gene_lengths(gtf):
    lens={}
    pat=re.compile(r'gene_name "([^"]+)"')
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"): continue
            f=line.split("\t")
            if len(f)<9 or f[2]!="gene": continue
            m=pat.search(f[8])
            if not m: continue
            sym=m.group(1); L=int(f[4])-int(f[3])+1
            if sym not in lens or L>lens[sym]: lens[sym]=L
    print(f"[gtf] {len(lens):,} gene lengths",file=sys.stderr)
    return lens

def per_gene(df):
    rows=[]
    for g,gg in df.groupby("gene",sort=False):
        s=gg["splice_score"].to_numpy()
        top=int(np.argmax(s))
        rows.append((g, int(bool(gg["is_canonical"].to_numpy()[top])), len(gg)))
    return pd.DataFrame(rows,columns=["gene","hit","n_peaks"])

def summarize(sub):
    n=len(sub); 
    if n==0: return None
    p1=sub.hit.mean(); _,lo,hi=wilson(int(sub.hit.sum()),n)
    floor=float((1.0/sub.n_peaks).mean())
    return dict(n=n,p1=p1,p1_lo=lo,p1_hi=hi,floor=floor,boost=p1-floor)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scored",required=True)
    ap.add_argument("--gtf",required=True)
    ap.add_argument("--summary",required=True,help="shortlist_summary.tsv (all genes incl uncontested)")
    ap.add_argument("--out",default="rerank_strat")
    a=ap.parse_args()

    df=pd.read_parquet(a.scored)
    sz=df.groupby("gene").size(); contested_set=set(sz[sz>1].index)
    dfc=df[df.gene.isin(contested_set)]
    pgc=per_gene(dfc)

    S=pd.read_csv(a.summary,sep="\t")
    rr_hit=dict(zip(pgc.gene,pgc.hit))
    recs=[]
    for _,r in S.iterrows():
        g=r["gene"]; contested = (r.get("n_internal",0)>0)
        if contested:
            rf_ok=0
            spl_ok=int(rr_hit.get(g,0))
        else:
            rf_ok=1; spl_ok=1
        recs.append(dict(gene=g,contested=int(contested),rf_correct=rf_ok,splice_correct=spl_ok))
    A=pd.DataFrame(recs)

    lens=load_gene_lengths(a.gtf)
    A["length"]=A["gene"].map(lens)
    n_nolen=A["length"].isna().sum()
    print(f"[match] {len(A)-n_nolen}/{len(A)} genes matched a GENCODE length "
          f"({n_nolen} unmatched)",file=sys.stderr)
    A=A.dropna(subset=["length"]).copy()

    # stratify by gene length (quintiles), rates per method
    A["len_bin"]=pd.qcut(A["length"],5,labels=False,duplicates="drop")
    rowsL=[]
    for b,gg in A.groupby("len_bin"):
        n=len(gg)
        rf=gg.rf_correct.mean(); sp=gg.splice_correct.mean()
        _,rlo,rhi=wilson(int(gg.rf_correct.sum()),n)
        _,slo,shi=wilson(int(gg.splice_correct.sum()),n)
        rowsL.append(dict(bin=int(b),n=n,n_contested=int(gg.contested.sum()),
            len_lo=int(gg.length.min()),len_hi=int(gg.length.max()),len_med=int(gg.length.median()),
            rf_rate=rf,rf_lo=rlo,rf_hi=rhi,splice_rate=sp,splice_lo=slo,splice_hi=shi,
            gain=sp-rf))
    L=pd.DataFrame(rowsL)
    L.to_csv(f"{a.out}_bylength.tsv",sep="\t",index=False)

    pg=pgc

    # stratify by contest size (n_peaks bins)
    bins=[(2,2),(3,4),(5,8),(9,16),(17,10**9)]; labels=["2","3-4","5-8","9-16","17+"]
    rowsC=[]
    for (lo,hi),lab in zip(bins,labels):
        gg=pg[(pg.n_peaks>=lo)&(pg.n_peaks<=hi)]
        s=summarize(gg)
        if s: s["contest"]=lab; s["c_lo"]=lo; s["c_hi"]=hi; rowsC.append(s)
    C=pd.DataFrame(rowsC)
    C.to_csv(f"{a.out}_bycontest.tsv",sep="\t",index=False)

    print(f"\n=== correct-rate by GENE LENGTH (all genes; RF argmax vs RF+splice) ===",file=sys.stderr)
    for _,r in L.iterrows():
        print(f"  Q{int(r.bin)+1} len[{r.len_lo:,}-{r.len_hi:,}] n={int(r.n):<5} "
              f"(contested {int(r.n_contested)})  RF={r.rf_rate:.3f}  RF+splice={r.splice_rate:.3f}  "
              f"gain={r.gain:+.3f}",file=sys.stderr)
    print(f"\n=== boost over floor by CONTEST SIZE ===",file=sys.stderr)
    for _,r in C.iterrows():
        print(f"  {r.contest:<5} n={int(r.n):<5} P@1={r.p1:.3f} floor={r.floor:.3f} "
              f"boost={r.boost:+.3f}",file=sys.stderr)
    print(f"\n[out] {a.out}_bylength.tsv, {a.out}_bycontest.tsv",file=sys.stderr)

if __name__=="__main__": main()