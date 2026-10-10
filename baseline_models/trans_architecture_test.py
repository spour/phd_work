#!/usr/bin/env python3
# the trans test, at the right scale
import argparse, sys, time
from collections import defaultdict
import numpy as np, pandas as pd

TEST={"chr18","chr19","chr20","chr21","chr22","chrX"}

def auc_mw(s,y):
    s=np.asarray(s,float); y=np.asarray(y,int); ok=np.isfinite(s); s=s[ok]; y=y[ok]
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    r=pd.Series(np.concatenate([pos,neg])).rank(method="average").to_numpy()
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def cohens_d(x,y):
    x=np.asarray(x,float); y=np.asarray(y,int); ok=np.isfinite(x); x=x[ok]; y=y[ok]
    a=x[y==1]; b=x[y==0]
    if len(a)<2 or len(b)<2: return np.nan
    return (a.mean()-b.mean())/(np.sqrt((a.var()+b.var())/2)+1e-9)

def parse_gencode(gtf):
    gene_span={}
    tx_exons=defaultdict(list)
    tx_gene={}
    t0=time.time(); nlines=0
    with open(gtf) as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            f=ln.split("\t")
            if len(f)<9: continue
            typ=f[2]
            if typ not in ("gene","exon"): continue
            attrs=f[8]; name=None; tid=None
            for kv in attrs.split(";"):
                kv=kv.strip()
                if kv.startswith("gene_name"): name=kv.split('"')[1]
                elif kv.startswith("transcript_id"): tid=kv.split('"')[1]
            if name is None: continue
            if typ=="gene":
                gene_span[name]=[f[0],f[6],int(f[3]),int(f[4])]
            else:
                tx_exons[(name,tid)].append((int(f[3]),int(f[4])))
            nlines+=1
            if nlines%500000==0: print(f"[gtf] {nlines:,} lines  {(time.time()-t0)/60:.1f}m",file=sys.stderr)
    donors=defaultdict(set); acceptors=defaultdict(set); introns_longest={}; n_introns={}
    by_gene=defaultdict(list)
    for (name,tid),ex in tx_exons.items(): by_gene[name].append(ex)
    for name,txs in by_gene.items():
        if name not in gene_span: continue
        strand=gene_span[name][1]
        longest=max(txs,key=lambda e:sum(b-a for a,b in e))
        for ex in txs:
            ex=sorted(ex)
            for k in range(len(ex)-1):
                istart=ex[k][1]+1; iend=ex[k+1][0]-1
                if iend<=istart: continue
                if strand=="+":
                    donors[name].add(istart); acceptors[name].add(iend)
                else:
                    donors[name].add(iend); acceptors[name].add(istart)
        L=sorted(longest); intr=[]
        for k in range(len(L)-1):
            a=L[k][1]+1; b=L[k+1][0]-1
            if b>a: intr.append((a,b))
        introns_longest[name]=intr; n_introns[name]=len(intr)
    for k in donors: donors[k]=np.array(sorted(donors[k]))
    for k in acceptors: acceptors[k]=np.array(sorted(acceptors[k]))
    print(f"[gtf] {len(gene_span):,} genes parsed  {(time.time()-t0)/60:.1f}m",file=sys.stderr)
    return gene_span,donors,acceptors,introns_longest,n_introns

def nearest_dirs(arr, c, strand):
    if arr is None or len(arr)==0: return np.nan,np.nan
    if strand=="+":
        up=arr[arr<c]; dn=arr[arr>c]
        du=(c-up.max()) if up.size else np.nan
        dd=(dn.min()-c) if dn.size else np.nan
    else:
        up=arr[arr>c]; dn=arr[arr<c]
        du=(up.min()-c) if up.size else np.nan
        dd=(c-dn.max()) if dn.size else np.nan
    return float(du),float(dd)

def host_intron(introns, c):
    for (a,b) in introns:
        if a<=c<=b: return float(b-a+1)
    return np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels",required=True); ap.add_argument("--gtf",required=True)
    ap.add_argument("--internal_only",action="store_true",help="restrict to INTERNAL compartment (matched, compartment-controlled)")
    ap.add_argument("--gc_floor",type=float,default=0.629,help="measured GC-only AUC to compare against")
    ap.add_argument("--full_auc",type=float,default=0.726,help="measured full matched AUC for context")
    ap.add_argument("--n_boot",type=int,default=500)
    ap.add_argument("--out",default="trans_arch")
    a=ap.parse_args()

    df=pd.read_parquet(a.labels)
    if a.internal_only and "compartment" in df.columns:
        df=df[df.compartment=="INTERNAL"].reset_index(drop=True)
    print(f"[data] {len(df):,} sites ({int((df.label==1).sum())} used, {int((df.label==0).sum())} cryptic)",file=sys.stderr)
    gene_span,donors,acceptors,introns_longest,n_introns=parse_gencode(a.gtf)

    feats=[]
    miss=0
    for r in df.itertuples(index=False):
        g=getattr(r,"gene",None); c=int(r.center); st=r.strand
        row=dict(gene_length=np.nan,n_introns=np.nan,host_intron_size=np.nan,frac_along_gene=np.nan,
                 dist_up_5ss=np.nan,dist_dn_5ss=np.nan,dist_up_3ss=np.nan,dist_dn_3ss=np.nan)
        if g in gene_span:
            ch,strand,gs,ge=gene_span[g]
            gl=ge-gs+1; row["gene_length"]=gl; row["n_introns"]=n_introns.get(g,np.nan)
            row["frac_along_gene"]=((c-gs)/gl) if strand=="+" else ((ge-c)/gl)
            du5,dd5=nearest_dirs(donors.get(g),c,strand)
            du3,dd3=nearest_dirs(acceptors.get(g),c,strand)
            row["dist_up_5ss"]=du5; row["dist_dn_5ss"]=dd5
            row["dist_up_3ss"]=du3; row["dist_dn_3ss"]=dd3
            row["host_intron_size"]=host_intron(introns_longest.get(g,[]),c)
        else:
            miss+=1
        feats.append(row)
    F=pd.DataFrame(feats)
    F["log_gene_length"]=np.log10(F["gene_length"].clip(lower=1))
    out=pd.concat([df.reset_index(drop=True),F],axis=1)
    out.to_parquet(f"{a.out}.parquet",index=False)
    if miss: print(f"[warn] {miss:,} sites had no GENCODE gene match (gene symbol mismatch)",file=sys.stderr)

    y=out["label"].to_numpy().astype(int)
    exp={"host_intron_size":+1,"dist_up_5ss":+1,"gene_length":+1,"log_gene_length":+1,
         "dist_dn_3ss":+1,"dist_dn_5ss":+1,"frac_along_gene":+1,"n_introns":+1,"dist_up_3ss":-1}
    feat_cols=["host_intron_size","dist_up_5ss","dist_dn_5ss","dist_up_3ss","dist_dn_3ss",
               "gene_length","log_gene_length","n_introns","frac_along_gene"]
    print(f"\n=== per-feature discrimination (used=1 vs cryptic=0) ===",file=sys.stderr)
    print(f"  {'feature':<18}{'AUC':>8}{'cohen_d':>9}   mechanistic prediction",file=sys.stderr)
    note={"host_intron_size":"used in LARGER introns (escape)","dist_up_5ss":"used FARTHER from protective donor (telescripting)",
          "gene_length":"used in LONGER genes (attrition)","dist_dn_3ss":"used FARTHER from removing acceptor",
          "dist_up_3ss":"used CLOSER to licensing acceptor"}
    for f in feat_cols:
        au=auc_mw(out[f],y); d=cohens_d(out[f],y); v=""
        if f in exp and np.isfinite(d):
            ok=(d>0)==(exp[f]>0); v=" match" if (ok and abs(d)>0.05) else (" n.s." if abs(d)<=0.05 else " OPPOSITE")
        ds=max(min(d,99.9),-99.9) if np.isfinite(d) else float("nan")
        print(f"  {f:<18}{au:>8.3f}  {ds:>7.3f}  {note.get(f,'')}{v}",file=sys.stderr)

    # held-out combined trans model vs gc floor
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    chrom=out["chrom"].astype(str)
    tr=~chrom.isin(TEST); te=chrom.isin(TEST)
    Xtr=out.loc[tr,feat_cols].to_numpy(float); Xte=out.loc[te,feat_cols].to_numpy(float)
    ytr=y[tr.values]; yte=y[te.values]
    print(f"\n=== held-out trans model (train non-test chroms, eval test chroms) ===",file=sys.stderr)
    print(f"  test n={te.sum()} ({int(yte.sum())} used / {int((yte==0).sum())} cryptic)",file=sys.stderr)
    clf=HistGradientBoostingClassifier(max_iter=300,max_depth=4,min_samples_leaf=40,
                                       learning_rate=0.05,class_weight="balanced",random_state=0)
    clf.fit(Xtr,ytr)
    p=clf.predict_proba(Xte)[:,1]
    a_trans=auc_mw(p,yte)
    rng=np.random.default_rng(0); n=len(yte); idx=np.arange(n); boots=[]
    for _ in range(a.n_boot):
        b=rng.choice(idx,n,replace=True)
        if len(np.unique(yte[b]))==2: boots.append(auc_mw(p[b],yte[b]))
    lo,hi=np.percentile(boots,[2.5,97.5])
    print(f"  trans-architecture AUC = {a_trans:.3f}  95% CI [{lo:.3f}, {hi:.3f}]",file=sys.stderr)
    print(f"  GC-only floor          = {a.gc_floor:.3f}   (measured)",file=sys.stderr)
    print(f"  full cis (CNN) matched = {a.full_auc:.3f}   (measured, for context)",file=sys.stderr)
    if lo>a.gc_floor:
        verdict=f"TRANS BEATS GC: architecture predicts usage beyond composition (CI lower {lo:.3f} > GC {a.gc_floor:.3f})"
    elif hi<a.gc_floor:
        verdict=f"trans is BELOW the GC floor"
    else:
        verdict=f"trans CI overlaps the GC floor (not clearly beyond composition)"
    print(f"  -> {verdict}",file=sys.stderr)

    pd.DataFrame([dict(trans_auc=a_trans,ci_lo=lo,ci_hi=hi,gc_floor=a.gc_floor,full_auc=a.full_auc)]
                 ).to_csv(f"{a.out}_summary.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}.parquet, {a.out}_summary.tsv",file=sys.stderr)

if __name__=="__main__": main()