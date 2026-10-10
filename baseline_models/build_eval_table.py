#!/usr/bin/env python3
# per test-chrom gene, the candidate set the augmented model takes argmax over
import argparse, os, sys, tempfile
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

def nms(scores,pos,min_dist,n):
    order=np.argsort(scores)[::-1]; keep=[]; kp=[]
    for i in order:
        if not np.isfinite(scores[i]): continue
        p=pos[i]
        if any(abs(p-q)<=min_dist for q in kp): continue
        keep.append(int(i)); kp.append(int(p))
        if len(keep)>=n: break
    return keep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--per_gene_dir",required=True)
    ap.add_argument("--pc_genes_bed",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--pwms",required=True)
    ap.add_argument("--u1_lookup",required=True); ap.add_argument("--me5_lookup",required=True)
    ap.add_argument("--win500",type=int,default=500)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--n_bins",type=int,default=8); ap.add_argument("--center_offset",type=int,default=249)
    ap.add_argument("--max_peaks",type=int,default=200); ap.add_argument("--min_peak_distance",type=int,default=250)
    ap.add_argument("--tolerance_nt",type=int,default=250)
    ap.add_argument("--test_chroms",default="chr18,chr19,chr20,chr21,chr22,chrX")
    ap.add_argument("--limit_genes",type=int,default=0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan)
    from compute_gene_baselines import derive_axes
    import splice_extra_features as XF
    from splice_common import compute_splice_row, SPLICE_COLS
    from featurization import read_pwms, featurize_fasta_to_df

    test=set(a.test_chroms.split(","))
    fa=GenomeFasta(a.reference_fasta); u1=load_u1_lookup(a.u1_lookup); me5=load_me5_lookup(a.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    pwms=read_pwms(a.pwms,max_len=20,bg=[0.3,0.2,0.2,0.3])
    pc=pd.read_csv(a.pc_genes_bed,sep="\t",header=None,
                   names=["chrom","start","end","name","score","strand"])
    pc["gene"]=pc["name"].astype(str).str.split("_").str[-1]; pc=pc.drop_duplicates("gene").set_index("gene")
    genes=[g[:-8] for g in os.listdir(a.per_gene_dir) if g.endswith(".parquet")]
    genes=[g for g in genes if g in pc.index and pc.loc[g,"chrom"] in test]
    if a.limit_genes>0: genes=sorted(genes)[:a.limit_genes]

    out_rows=[]
    for gene in tqdm(genes,desc="test genes"):
        chrom=pc.loc[gene,"chrom"]; strand=pc.loc[gene,"strand"]
        d=pd.read_parquet(os.path.join(a.per_gene_dir,f"{gene}.parquet"),
                          columns=["window_start_genomic","rf_prob","is_constitutive_cpa"])
        d=d.dropna(subset=["rf_prob"]).reset_index(drop=True)
        if len(d)==0: continue
        pos=d["window_start_genomic"].to_numpy(); rfv=d["rf_prob"].to_numpy()
        canon=d.loc[d["is_constitutive_cpa"]==True,"window_start_genomic"].to_numpy()
        idx=nms(rfv,pos,a.min_peak_distance,a.max_peaks)
        recs=[]; meta=[]
        for j in idx:
            ws=int(pos[j])
            c=ws+a.center_offset if strand=="+" else ws-a.center_offset
            s0=c-a.win500//2; s1=s0+a.win500
            seq500=fa.fetch(chrom,s0,s1,strand=strand)
            if seq500 is None or len(seq500)<a.win500: continue
            isc=int(len(canon)>0 and np.min(np.abs(canon-ws))<=a.tolerance_nt)
            recs.append((f"{gene}|{ws}",seq500)); meta.append((ws,isc,float(rfv[j])))
        if not recs: continue
        with tempfile.NamedTemporaryFile("w",suffix=".fa",delete=False) as tf:
            for rid,s in recs: tf.write(f">{rid}\n{s}\n")
            tmpfa=tf.name
        Xpwm=featurize_fasta_to_df(tmpfa,pwms,bg=[0.3,0.2,0.2,0.3],verbose=False)
        os.unlink(tmpfa)
        Xpwm=Xpwm.sort_index(axis=1)
        for k,(rid,_) in enumerate(recs):
            ws,isc,rf=meta[k]
            c=ws+a.center_offset if strand=="+" else ws-a.center_offset
            if strand=="+": w0=max(0,c-a.upstream_nt); w1=c+a.downstream_nt
            else:           w0=max(0,c-a.downstream_nt); w1=c+a.upstream_nt
            seqL=fa.fetch(chrom,w0,w1,strand=strand)
            spl=compute_splice_row(seqL,a.upstream_nt,a.downstream_nt,a.n_bins,me5,score3,m3,u1,derive_axes,XF)
            rec=dict(gene=gene,chrom=chrom,position=ws,is_constitutive=isc,rf_prob=rf)
            rec.update({col:Xpwm.iloc[k][col] for col in Xpwm.columns})
            rec.update(spl)
            out_rows.append(rec)
    out=pd.DataFrame(out_rows)
    out.to_parquet(a.out,index=False)
    print(f"[done] {len(out):,} candidates, {out.gene.nunique():,} test genes -> {a.out}",file=sys.stderr)
    print(f"[cols] {Xpwm.shape[1]} pwm+kmer + {len(SPLICE_COLS)} splice",file=sys.stderr)
if __name__=="__main__": main()