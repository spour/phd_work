#!/usr/bin/env python3
# (step 1 of the clean splicing-reranker rebuild) per gene, from its rf scan parquet: 1
import argparse, os, sys
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS={"chr3","chr13"}

def load_canonical(bed):
    df=pd.read_csv(bed,sep="\t",header=None,
                   names=["chrom","start","end","symbol","enst","strand"])
    df["chrom"]=df["chrom"].astype(str)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"])
    by={}
    for _,r in df.iterrows():
        by.setdefault(r["symbol"],[]).append((r["chrom"],int(r["start"]),int(r["end"]),r["strand"]))
    print(f"[canon] {len(df):,} canonical windows, {len(by):,} symbols",file=sys.stderr)
    return by

def nms_peaks(centers, scores, merge_dist, reserved_center=None):
    order=np.argsort(scores)[::-1]
    kept=[]; suppressed=np.zeros(len(centers),bool)
    if reserved_center is not None:
        near=np.where(np.abs(centers-reserved_center)<=merge_dist)[0]
        if len(near):
            cpk=near[np.argmax(scores[near])]
            kept.append(cpk)
            suppressed[np.abs(centers-centers[cpk])<=merge_dist]=True
    for i in order:
        if suppressed[i]: continue
        kept.append(i)
        suppressed[np.abs(centers-centers[i])<=merge_dist]=True
    return kept

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--per_gene_dir",required=True)
    ap.add_argument("--canonical_bed",required=True)
    ap.add_argument("--pc_genes_bed",default=None,help="optional, only to restrict/sanity-check genes")
    ap.add_argument("--win",type=int,default=500)
    ap.add_argument("--merge_dist",type=int,default=None,
                    help="NMS collapse distance for neighbouring high windows; default = win")
    ap.add_argument("--split",default="all",choices=["all","test","val","train"],
                    help="restrict to a chromosome split")
    ap.add_argument("--shard",type=int,default=0,help="this shard index (0-based)")
    ap.add_argument("--n_shards",type=int,default=1,help="total shards (SLURM array size)")
    ap.add_argument("--out",default="shortlist")
    a=ap.parse_args()
    off=a.win//2
    merge_dist = a.win if a.merge_dist is None else a.merge_dist

    canon=load_canonical(a.canonical_bed)
    genes=sorted(canon.keys())
    if a.split!="all":
        sel = TEST_CHROMS if a.split=="test" else VAL_CHROMS if a.split=="val" else None
        def keep(g):
            chl={c for c,_,_,_ in canon[g]}
            if a.split=="train": return not (chl&TEST_CHROMS) and not (chl&VAL_CHROMS)
            return bool(chl & sel)
        genes=[g for g in genes if keep(g)]
    if a.n_shards>1:
        genes=genes[a.shard::a.n_shards]
        print(f"[shard] {a.shard}/{a.n_shards}: {len(genes):,} genes",file=sys.stderr)
    print(f"[genes] {len(genes):,} genes in split={a.split}",file=sys.stderr)

    rows=[]; summ=[]; missing=0; no_overlap=0
    for g in tqdm(genes,desc="shortlist"):
        p=os.path.join(a.per_gene_dir,f"{g}.parquet")
        if not os.path.exists(p): missing+=1; continue
        d=pd.read_parquet(p,columns=["window_start_genomic","rf_prob"]).dropna(subset=["rf_prob"])
        if len(d)==0: continue
        ws=d["window_start_genomic"].to_numpy(); rf=d["rf_prob"].to_numpy()
        we=ws+a.win
        is_canon=np.zeros(len(d),bool)
        canon_centers=[]
        for chrom,cs,ce,strand in canon[g]:
            is_canon |= (ws<ce) & (we>cs)
            canon_centers.append((cs+ce)//2)
        if not is_canon.any():
            no_overlap+=1; continue
        rf_canonical=float(rf[is_canon].max())
        short=np.nonzero(rf>=rf_canonical)[0]
        if len(short)==0: continue
        c_all=ws+off
        cc=np.array(canon_centers)
        canon_idx=np.nonzero(is_canon)[0]
        canon_peak_center=int(c_all[canon_idx[np.argmax(rf[canon_idx])]])
        sub_centers=c_all[short]; sub_scores=rf[short]
        keep_local=nms_peaks(sub_centers,sub_scores,merge_dist,
                             reserved_center=canon_peak_center)
        peaks=short[keep_local]
        strand=canon[g][0][3]; chrom=canon[g][0][0]
        peak_rows=[]
        for i in peaks:
            ctr=int(c_all[i])
            d_can=int(np.min(np.abs(cc-ctr)))
            is_c = is_canon[i] or (d_can<=merge_dist)
            peak_rows.append((i,ctr,d_can,is_c))
        n_short=len(peak_rows); n_internal=sum(1 for _,_,_,c in peak_rows if not c)
        for i,ctr,d_can,is_c in peak_rows:
            rows.append(dict(gene=g,chrom=chrom,strand=strand,
                             window_start_genomic=int(ws[i]),center=ctr,
                             rf_prob=float(rf[i]),is_canonical=bool(is_c),
                             dist_to_canon=d_can,rf_canonical=rf_canonical,
                             n_shortlist=n_short,n_internal=n_internal))
        summ.append(dict(gene=g,chrom=chrom,rf_canonical=rf_canonical,
                         n_shortlist=n_short,n_internal=n_internal,contested=int(n_internal>0)))
    R=pd.DataFrame(rows); S=pd.DataFrame(summ)
    suf=f".{a.shard:03d}" if a.n_shards>1 else ""
    R.to_parquet(f"{a.out}_windows{suf}.parquet",index=False)
    S.to_csv(f"{a.out}_summary{suf}.tsv",sep="\t",index=False)

    print(f"\n[built] {len(R):,} shortlisted windows over {len(S):,} genes "
          f"({missing} parquets missing, {no_overlap} genes had no window overlapping canonical)",file=sys.stderr)
    if len(S):
        contested=S[S.contested==1]
        print(f"[contest] {len(contested):,}/{len(S):,} genes contested "
              f"({len(contested)/len(S):.1%}); among contested, mean shortlist "
              f"size={contested.n_shortlist.mean():.2f}, mean competitors={contested.n_internal.mean():.2f}",file=sys.stderr)
        print(f"[contest] shortlist-size distribution (all genes):",file=sys.stderr)
        for lab,lo,hi in [("1",1,1),("2",2,2),("3-4",3,4),("5-8",5,8),("9+",9,10**9)]:
            n=int(((S.n_shortlist>=lo)&(S.n_shortlist<=hi)).sum())
            print(f"    {lab:<5} {n:>6}  {n/len(S):.3f}",file=sys.stderr)
        floor_all=float((1.0/S.n_shortlist).mean())
        floor_con=float((1.0/contested.n_shortlist).mean()) if len(contested) else float("nan")
        print(f"[floor] implied random-tie-break P@1: overall={floor_all:.3f}  "
              f"contested-only={floor_con:.3f}  (this is the ruler, not a method)",file=sys.stderr)
    print(f"[out] {a.out}_windows.parquet, {a.out}_summary.tsv",file=sys.stderr)

if __name__=="__main__": main()