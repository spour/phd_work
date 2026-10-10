#!/usr/bin/env python3
# dump one-hot sequence for each shortlist peak, same window as the splice features
import argparse, sys
import numpy as np, pandas as pd

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS={"chr3","chr13"}
B={"A":0,"C":1,"G":2,"T":3}

def one_hot(seq,L):
    x=np.zeros((L,4),np.float16)
    if seq is None: return x
    s=seq.upper()
    for i,ch in enumerate(s[:L]):
        j=B.get(ch,-1)
        if j>=0: x[i,j]=1.0
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--shortlist",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000)
    ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--split",default="test",choices=["all","test","val","train"])
    ap.add_argument("--contested_only",action="store_true")
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--n_shards",type=int,default=1)
    ap.add_argument("--out",default="peakseq")
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import GenomeFasta

    df=pd.read_parquet(a.shortlist)
    if a.split=="test": df=df[df.chrom.isin(TEST_CHROMS)]
    elif a.split=="val": df=df[df.chrom.isin(VAL_CHROMS)]
    elif a.split=="train": df=df[~df.chrom.isin(TEST_CHROMS|VAL_CHROMS)]
    if a.contested_only: df=df[df.n_internal>0]
    df=df.reset_index(drop=True)
    if a.n_shards>1:
        genes=sorted(df.gene.unique())[a.shard::a.n_shards]
        df=df[df.gene.isin(set(genes))].reset_index(drop=True)
    L=a.upstream_nt+a.downstream_nt
    print(f"[seq] {len(df):,} peaks, {df.gene.nunique():,} genes, window {L}nt",file=sys.stderr)
    if len(df)==0: sys.exit("no peaks")

    fa=GenomeFasta(a.reference_fasta)
    X=np.zeros((len(df),L,4),np.float16)
    for i,r in df.iterrows():
        c=int(r["center"]); strand=r["strand"]; chrom=r["chrom"]
        if strand=="+": w0=max(0,c-a.upstream_nt); w1=c+a.downstream_nt
        else:           w0=max(0,c-a.downstream_nt); w1=c+a.upstream_nt
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: seq=None
        X[i]=one_hot(seq,L)
    suf=f".{a.shard:03d}" if a.n_shards>1 else ""
    np.savez_compressed(f"{a.out}{suf}.npz", X=X)
    df[["gene","chrom","center","strand","rf_prob","is_canonical","n_internal"]].to_parquet(f"{a.out}{suf}.meta.parquet",index=False)
    print(f"[out] {a.out}{suf}.npz ({X.shape}), {a.out}{suf}.meta.parquet",file=sys.stderr)

if __name__=="__main__": main()