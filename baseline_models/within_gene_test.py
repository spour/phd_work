#!/usr/bin/env python3
# the only framing the selection bias cannot fake
import argparse, sys, time
from collections import defaultdict
import numpy as np, pandas as pd

def parse_gencode(gtf):
    from collections import defaultdict
    gene_span={}; tx_exons=defaultdict(list)
    with open(gtf) as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            f=ln.split("\t")
            if len(f)<9 or f[2] not in ("gene","exon"): continue
            name=None; tid=None
            for kv in f[8].split(";"):
                kv=kv.strip()
                if kv.startswith("gene_name"): name=kv.split('"')[1]
                elif kv.startswith("transcript_id"): tid=kv.split('"')[1]
            if name is None: continue
            if f[2]=="gene": gene_span[name]=[f[0],f[6],int(f[3]),int(f[4])]
            else: tx_exons[(name,tid)].append((int(f[3]),int(f[4])))
    donors=defaultdict(set); introns_longest={}
    by_gene=defaultdict(list)
    for (name,tid),ex in tx_exons.items(): by_gene[name].append(ex)
    for name,txs in by_gene.items():
        if name not in gene_span: continue
        strand=gene_span[name][1]; longest=max(txs,key=lambda e:sum(b-a for a,b in e))
        for ex in txs:
            ex=sorted(ex)
            for k in range(len(ex)-1):
                a=ex[k][1]+1; b=ex[k+1][0]-1
                if b<=a: continue
                donors[name].add(a if strand=="+" else b)
        L=sorted(longest); intr=[]
        for k in range(len(L)-1):
            a=L[k][1]+1; b=L[k+1][0]-1
            if b>a: intr.append((a,b))
        introns_longest[name]=intr
    for k in donors: donors[k]=np.array(sorted(donors[k]))
    return gene_span,donors,introns_longest

def nearest_up(arr,c,strand):
    if arr is None or len(arr)==0: return np.nan
    if strand=="+":
        up=arr[arr<c]; return float(c-up.max()) if up.size else np.nan
    else:
        up=arr[arr>c]; return float(up.min()-c) if up.size else np.nan
def nearest_dn3(intr,c,strand):
    ends=[]
    for (a,b) in intr: ends.append(b if strand=="+" else a)
    ends=np.array(sorted(ends)) if ends else np.array([])
    if ends.size==0: return np.nan
    if strand=="+":
        dn=ends[ends>c]; return float(dn.min()-c) if dn.size else np.nan
    else:
        dn=ends[ends<c]; return float(c-dn.max()) if dn.size else np.nan
def host_intron(intr,c):
    for (a,b) in intr:
        if a<=c<=b: return float(b-a+1), (a,b)
    return np.nan,None

def within_gene_auc(df, feat, higher_is_used=True):
    pairs=0; conc=0; p1_hit=0; p1_n=0
    for g,sub in df.groupby("gene",sort=False):
        u=sub[sub.label==1]; c=sub[sub.label==0]
        if len(u)==0 or len(c)==0: continue
        fu=u[feat].to_numpy(float); fc=c[feat].to_numpy(float)
        fu=fu[np.isfinite(fu)]; fc=fc[np.isfinite(fc)]
        if len(fu)==0 or len(fc)==0: continue
        uval=np.nanmean(fu)
        for cv in fc:
            pairs+=1
            if uval>cv: conc+=1
            elif uval==cv: conc+=0.5
        allf=np.concatenate([np.full(len(fu),uval),fc])
        lab=np.concatenate([np.ones(len(fu)),np.zeros(len(fc))])
        order=np.argsort(-allf) if higher_is_used else np.argsort(allf)
        p1_n+=1; p1_hit+=int(lab[order[0]]==1)
    auc=conc/pairs if pairs else np.nan
    p1=p1_hit/p1_n if p1_n else np.nan
    return auc,p1,p1_n,pairs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels",required=True); ap.add_argument("--gtf",required=True)
    ap.add_argument("--n_perm",type=int,default=1000)
    ap.add_argument("--out",default="within_gene")
    a=ap.parse_args()
    df=pd.read_parquet(a.labels)
    if "compartment" in df.columns: df=df[df.compartment=="INTERNAL"].reset_index(drop=True)
    has_both=df.groupby("gene")["label"].agg(lambda s:(s==1).any() and (s==0).any())
    elig=set(has_both[has_both].index)
    df=df[df.gene.isin(elig)].reset_index(drop=True)
    ng=df.gene.nunique()
    print(f"[data] {len(df):,} sites in {ng:,} genes that contain BOTH used and cryptic internal sites",file=sys.stderr)
    print(f"       ({int((df.label==1).sum())} used, {int((df.label==0).sum())} cryptic)",file=sys.stderr)
    if ng<30:
        print(f"[fatal] only {ng} eligible genes; within-gene test underpowered. "
              f"Few genes have both a used-internal and a cryptic-internal site, which is "
              f"itself informative: used and cryptic internal sites rarely co-occur in a gene.",file=sys.stderr)
        df.to_parquet(f"{a.out}.parquet",index=False); sys.exit(0)

    gene_span,donors,introns_longest=parse_gencode(a.gtf)
    rows=[]
    for r in df.itertuples(index=False):
        g=r.gene; c=int(r.center); st=r.strand
        d=dict(dist_up_5ss=np.nan,host_intron_size=np.nan,dist_dn_3ss=np.nan,frac_along_gene=np.nan,
                in_largest_intron=np.nan,frac_along_intron=np.nan)
        if g in gene_span:
            ch,strand,gs,ge=gene_span[g]; gl=ge-gs+1
            d["frac_along_gene"]=((c-gs)/gl) if strand=="+" else ((ge-c)/gl)
            d["dist_up_5ss"]=nearest_up(donors.get(g),c,strand)
            intr=introns_longest.get(g,[])
            hs,which=host_intron(intr,c); d["host_intron_size"]=hs
            d["dist_dn_3ss"]=nearest_dn3(intr,c,strand)
            if intr:
                sizes=[b-a for a,b in intr]; biggest=max(sizes) if sizes else 0
                d["in_largest_intron"]=1.0 if (which and (which[1]-which[0])==biggest) else 0.0
                if which:
                    a0,b0=which
                    pos=(c-a0)/(b0-a0+1) if strand=="+" else (b0-c)/(b0-a0+1)
                    d["frac_along_intron"]=pos
        rows.append(d)
    F=pd.DataFrame(rows); out=pd.concat([df.reset_index(drop=True),F],axis=1)
    out.to_parquet(f"{a.out}.parquet",index=False)

    feats=["dist_up_5ss","host_intron_size","dist_dn_3ss","frac_along_gene","in_largest_intron","frac_along_intron"]
    print(f"\n=== within-gene discrimination (used vs its cryptic gene-mates) ===",file=sys.stderr)
    print(f"  {'feature':<18}{'wg_AUC':>8}{'P@1':>7}{'perm_p':>9}",file=sys.stderr)
    rng=np.random.default_rng(0)
    res=[]
    for f in feats:
        au,p1,p1n,npairs=within_gene_auc(out,f)
        null=[]
        sub=out[["gene","label",f]].copy()
        for _ in range(a.n_perm):
            s=sub.copy()
            s["label"]=s.groupby("gene")["label"].transform(lambda x: rng.permutation(x.to_numpy()))
            na,_,_,_=within_gene_auc(s.rename(columns={f:f}),f)
            null.append(na)
        null=np.array([x for x in null if np.isfinite(x)])
        p=( (np.sum(np.abs(null-0.5)>=abs(au-0.5))+1)/(len(null)+1) ) if len(null) and np.isfinite(au) else np.nan
        res.append(dict(feature=f,wg_auc=au,p_at_1=p1,perm_p=p,n_genes=p1n))
        star=" *" if (np.isfinite(p) and p<0.05) else ""
        print(f"  {f:<18}{au:>8.3f}{p1:>7.3f}{p:>9.3f}{star}",file=sys.stderr)
    pd.DataFrame(res).to_csv(f"{a.out}_summary.tsv",sep="\t",index=False)
    print(f"\n  wg_AUC 0.5 = feature does NOT rank used above cryptic within genes.",file=sys.stderr)
    print(f"  perm_p < 0.05 (*) = within-gene signal survives the gene-structure-safe null.",file=sys.stderr)
    print(f"[out] {a.out}.parquet, {a.out}_summary.tsv",file=sys.stderr)

if __name__=="__main__": main()