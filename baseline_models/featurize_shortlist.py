#!/usr/bin/env python3
# (step 2 of the clean rebuild) featurize the shortlisted peaks for contested test-chrom
import argparse, os, sys
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS={"chr3","chr13"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--shortlist",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",required=True); ap.add_argument("--me5_lookup",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--n_bins",type=int,default=8)
    ap.add_argument("--split",default="test",choices=["all","test","val","train"])
    ap.add_argument("--contested_only",action="store_true")
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--n_shards",type=int,default=1)
    ap.add_argument("--out",default="shortlist_feat")
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan)
    from compute_gene_baselines import derive_axes 
    import splice_extra_features as XF
    from splice_common import compute_splice_row, SPLICE_COLS

    df=pd.read_parquet(a.shortlist)
    if a.split=="test": df=df[df.chrom.isin(TEST_CHROMS)]
    elif a.split=="val": df=df[df.chrom.isin(VAL_CHROMS)]
    elif a.split=="train": df=df[~df.chrom.isin(TEST_CHROMS)]
    if a.contested_only: df=df[df.n_internal>0]
    df=df.reset_index(drop=True)
    if a.n_shards>1:
        genes=sorted(df.gene.unique())[a.shard::a.n_shards]
        df=df[df.gene.isin(set(genes))].reset_index(drop=True)
        print(f"[shard] {a.shard}/{a.n_shards}",file=sys.stderr)
    print(f"[feat] {len(df):,} peaks over {df.gene.nunique():,} contested {a.split} genes",file=sys.stderr)
    if len(df)==0: sys.exit("[err] no peaks after filtering")

    fa=GenomeFasta(a.reference_fasta); u1=load_u1_lookup(a.u1_lookup); me5=load_me5_lookup(a.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    rows=[]
    for _,r in tqdm(df.iterrows(),total=len(df),desc="splice"):
        c=int(r["center"]); strand=r["strand"]; chrom=r["chrom"]
        if strand=="+": w0=max(0,c-a.upstream_nt); w1=c+a.downstream_nt
        else:           w0=max(0,c-a.downstream_nt); w1=c+a.upstream_nt
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: seq=None
        rows.append(compute_splice_row(seq,a.upstream_nt,a.downstream_nt,a.n_bins,me5,score3,m3,u1,derive_axes,XF))
    F=pd.DataFrame(rows,columns=SPLICE_COLS)
    out=pd.concat([df.reset_index(drop=True),F],axis=1)
    for c in SPLICE_COLS: out[c]=pd.to_numeric(out[c],errors="coerce")
    suf=f".{a.shard:03d}" if a.n_shards>1 else ""
    out.to_parquet(f"{a.out}{suf}.parquet",index=False)
    nonnan=out[SPLICE_COLS].notna().mean().mean()
    print(f"[feat] splice non-NaN frac={nonnan:.3f}; canonical peaks={int(out.is_canonical.sum())}, "
          f"competitor peaks={int((~out.is_canonical).sum())}",file=sys.stderr)
    print(f"[out] {a.out}{suf}.parquet",file=sys.stderr)

if __name__=="__main__": main()
