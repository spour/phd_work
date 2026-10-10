#!/usr/bin/env python3
# compute raw splice features (-1000/+500) for every window in a bed, in bed row order
import argparse, sys
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--bed",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",required=True); ap.add_argument("--me5_lookup",required=True)
    ap.add_argument("--upstream_nt",type=int,default=1000)
    ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--n_bins",type=int,default=8)
    ap.add_argument("--id_mode",choices=["coord","name"],default="coord",
                    help="coord = 'chrom:start-end(strand)' (bedtools getfasta -s default); "
                         "name = bed column 4. Must match your .fa headers.")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup,
        init_maxentscan)
    from compute_gene_baselines import derive_axes
    import splice_extra_features as XF
    from splice_common import compute_splice_row, SPLICE_COLS

    fa=GenomeFasta(a.reference_fasta); u1=load_u1_lookup(a.u1_lookup); me5=load_me5_lookup(a.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    bed=pd.read_csv(a.bed,sep="\t",header=None,
                    names=["chrom","start","end","name","score","strand"])
    ids=[]; rows=[]
    for _,r in tqdm(bed.iterrows(),total=len(bed),desc=a.bed.split("/")[-1]):
        chrom=r["chrom"]; start=int(r["start"]); end=int(r["end"]); strand=r["strand"]
        center=(start+end)//2
        if strand=="+": w0=max(0,center-a.upstream_nt); w1=center+a.downstream_nt
        else:           w0=max(0,center-a.downstream_nt); w1=center+a.upstream_nt
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: seq=None
        rows.append(compute_splice_row(seq,a.upstream_nt,a.downstream_nt,a.n_bins,
                                       me5,score3,m3,u1,derive_axes,XF))
        if a.id_mode=="name": ids.append(str(r["name"]))
        else: ids.append(f"{chrom}:{start}-{end}({strand})")
    out=pd.DataFrame(rows,columns=SPLICE_COLS); out.index=ids; out.index.name="id"
    out.to_csv(a.out,sep="\t")
    nf=out.notna().mean().mean()
    print(f"[done] {len(out):,} rows x {out.shape[1]} splice cols -> {a.out} "
          f"(non-NaN frac {nf:.3f})",file=sys.stderr)
if __name__=="__main__": main()