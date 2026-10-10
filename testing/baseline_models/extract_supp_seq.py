#!/usr/bin/env python3
# one-hot sequence around used (label 1) and suppressed-cryptic (label 0) sites
import argparse, sys
import numpy as np, pandas as pd
TEST={"chr18","chr19","chr20","chr21","chr22","chrX"}; VAL={"chr3","chr13"}
B={"A":0,"C":1,"G":2,"T":3}
def one_hot(seq,L):
    x=np.zeros((L,4),np.float16)
    if seq is None: return x
    for i,ch in enumerate(seq.upper()[:L]):
        j=B.get(ch,-1)
        if j>=0: x[i,j]=1.0
    return x
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels",required=True); ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--split",default="train",choices=["all","train","val","test"])
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--n_shards",type=int,default=1)
    ap.add_argument("--out",default="suppseq")
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import GenomeFasta
    df=pd.read_parquet(a.labels)
    if a.split=="test": df=df[df.chrom.isin(TEST)]
    elif a.split=="val": df=df[df.chrom.isin(VAL)]
    elif a.split=="train": df=df[~df.chrom.isin(TEST|VAL)]
    df=df.reset_index(drop=True)
    if a.n_shards>1:
        df=df.iloc[a.shard::a.n_shards].reset_index(drop=True)
    L=a.upstream_nt+a.downstream_nt
    print(f"[seq] {len(df):,} sites ({int((df.label==1).sum())} used, "
          f"{int((df.label==0).sum())} cryptic), window {L}",file=sys.stderr)
    fa=GenomeFasta(a.reference_fasta)
    X=np.zeros((len(df),L,4),np.float16)
    for i,r in df.iterrows():
        c=int(r["center"]); st=r["strand"]; ch=r["chrom"]
        if st=="+": w0=max(0,c-a.upstream_nt); w1=c+a.downstream_nt
        else:       w0=max(0,c-a.downstream_nt); w1=c+a.upstream_nt
        try: seq=fa.fetch(ch,w0,w1,strand=st)
        except Exception: seq=None
        X[i]=one_hot(seq,L)
    suf=f".{a.shard:03d}" if a.n_shards>1 else ""
    np.savez_compressed(f"{a.out}{suf}.npz",X=X)
    df.to_parquet(f"{a.out}{suf}.meta.parquet",index=False)
    print(f"[out] {a.out}{suf}.npz {X.shape}, meta",file=sys.stderr)
if __name__=="__main__": main()
