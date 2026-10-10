#!/usr/bin/env python3
# (final analysis) the reranker, when it moves the pick off rf's argmax
import argparse, sys, math
from collections import defaultdict
import numpy as np, pandas as pd

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scored",required=True)
    ap.add_argument("--atlas_tsv",required=True,help="FULL atlas (per-sample cols)")
    ap.add_argument("--tol",type=int,default=100)
    ap.add_argument("--min_samples",type=int,default=3,help="min samples for a cell type")
    ap.add_argument("--out",default="reranker_celltype")
    a=ap.parse_args()

    # load full atlas with per-sample columns
    raw=pd.read_csv(a.atlas_tsv,sep="\t",header=0,dtype=str,low_memory=False)
    rc={c.lower().lstrip("#"):c for c in raw.columns}
    def col(*n):
        for x in n:
            if x in rc: return rc[x]
    c_chrom=col("chrom"); c_rep=col("rep","repsite"); c_ann=col("annotation")
    c_sym=col("gene_name"); c_ens=col("gene_id")
    fixed=set(filter(None,[c_chrom,c_rep,c_ann,c_sym,c_ens,col("chromstart"),
              col("chromend"),col("name"),col("score"),col("strand"),
              col("frac_samples"),col("nr_prots"),col("repsite_signals")]))
    sample_cols=[c for c in raw.columns if "|" in str(c)]
    src_of={c:str(c).split("|")[2] if len(str(c).split("|"))>2 else "" for c in sample_cols}
    src_groups=defaultdict(list)
    for c in sample_cols:
        if src_of[c]: src_groups[src_of[c]].append(c)
    src_groups={s:cs for s,cs in src_groups.items() if len(cs)>=a.min_samples}
    print(f"[atlas] {len(sample_cols)} samples, {len(src_groups)} cell types "
          f">= {a.min_samples} samples",file=sys.stderr)

    atlas=pd.DataFrame({
        "chrom":raw[c_chrom].astype(str),
        "pos":pd.to_numeric(raw[c_rep],errors="coerce")-1,
        "ann":raw[c_ann],
        "gene":raw[c_ens].where(raw[c_ens].notna()&(raw[c_ens]!="NA"),raw[c_sym]),
        "gsym":raw[c_sym],
    })
    atlas["chrom"]=atlas["chrom"].where(atlas["chrom"].str.startswith("chr"),"chr"+atlas["chrom"])
    ctmat={}
    for s,cs in src_groups.items():
        ctmat[s]=raw[cs].apply(pd.to_numeric,errors="coerce").fillna(0.0).sum(axis=1).to_numpy()
    atlas=atlas.dropna(subset=["pos"]).reset_index(drop=True)
    keep_idx=atlas.index if len(atlas)==len(raw) else None
    mask=pd.to_numeric(raw[c_rep],errors="coerce").notna().to_numpy()
    for s in ctmat: ctmat[s]=ctmat[s][mask]
    atlas=atlas.reset_index(drop=True)

    by_chrom={}
    for c,gg in atlas.groupby("chrom"):
        order=np.argsort(gg["pos"].to_numpy())
        by_chrom[c]=(gg["pos"].to_numpy()[order], gg.index.to_numpy()[order])

    def nearest_row(chrom,pos,tol):
        if chrom not in by_chrom: return None
        P,IDX=by_chrom[chrom]; i=np.searchsorted(P,pos); best=None; bd=tol+1
        for j in (i-1,i):
            if 0<=j<len(P):
                d=abs(P[j]-pos)
                if d<bd: bd=d; best=IDX[j]
        return best if best is not None else None

    gene_rows=defaultdict(list)
    for ridx,g in zip(atlas.index, atlas["gene"]):
        if isinstance(g,str) and g!="NA": gene_rows[g].append(ridx)
    total_tpm=np.zeros(len(atlas))
    for s in ctmat: total_tpm+=ctmat[s]

    # reranker picks
    sc=pd.read_parquet(a.scored)
    sz=sc.groupby("gene").size(); sc=sc[sc.gene.isin(set(sz[sz>1].index))]
    n_pick=0; n_on_atlas=0; n_noncanon=0; n_ct_dominant=0
    recs=[]
    for g,gg in sc.groupby("gene",sort=False):
        gg=gg.reset_index(drop=True)
        i_rr=int(np.argmax(gg["splice_score"].to_numpy()))
        chrom=gg["chrom"].iloc[i_rr]; pos=int(gg["center"].iloc[i_rr])
        is_canon=bool(gg["is_canonical"].iloc[i_rr])
        n_pick+=1
        ridx=nearest_row(chrom,pos,a.tol)
        if ridx is None: continue
        n_on_atlas+=1
        gsym=atlas["gsym"].iloc[ridx]; gene_key=atlas["gene"].iloc[ridx]
        rows=gene_rows.get(gene_key,[])
        if len(rows)<2: continue
        cross_dom = rows[int(np.argmax(total_tpm[rows]))]==ridx
        if cross_dom:
            continue
        n_noncanon+=1
        ct_hits=[]
        for s,vec in ctmat.items():
            sub=vec[rows]
            if sub.sum()<=0: continue
            if rows[int(np.argmax(sub))]==ridx: ct_hits.append(s)
        if ct_hits:
            n_ct_dominant+=1
            recs.append(dict(gene=gene_key,gsym=gsym,ann=atlas["ann"].iloc[ridx],
                             chrom=chrom,pos=pos,is_canonical_pred=is_canon,
                             n_celltypes_dominant=len(ct_hits),
                             example_celltypes=";".join(ct_hits[:5])))
    R=pd.DataFrame(recs); R.to_csv(f"{a.out}.tsv",sep="\t",index=False)

    print(f"\n=== reranker pick recovery of cell-type-specific ends ===",file=sys.stderr)
    print(f"  contested genes scored: {n_pick}",file=sys.stderr)
    print(f"  reranker pick on a real atlas cluster (<= {a.tol}nt): {n_on_atlas}",file=sys.stderr)
    print(f"  ... and NOT the cross-sample dominant cluster: {n_noncanon}",file=sys.stderr)
    p,lo,hi=wilson(n_ct_dominant,n_noncanon) if n_noncanon else (0,0,0)
    print(f"  ... of those, DOMINANT in >=1 specific cell type: {n_ct_dominant}/{n_noncanon} "
          f"= {p:.3f} [{lo:.3f},{hi:.3f}]",file=sys.stderr)
    print(f"\n  => when the reranker picks a real-but-non-canonical site, it is the "
          f"cell-type-dominant end in {p:.0%} of cases.",file=sys.stderr)
    if len(R):
        print(f"  annotation of recovered cell-type-specific picks:",file=sys.stderr)
        for k,v in R.ann.value_counts().items():
            print(f"    {k:<5} {v:>5}  {v/len(R):.2f}",file=sys.stderr)
    print(f"[out] {a.out}.tsv",file=sys.stderr)

if __name__=="__main__": main()