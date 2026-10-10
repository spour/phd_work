#!/usr/bin/env python3
# leakage control for the splicing-rerank result
import argparse, os, sys, math
import numpy as np, pandas as pd

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}
TRAIN_EXCL=TEST_CHROMS|{"chr3","chr13"}

def wilson(k,n,z=1.96):
    if n==0: return (0.0,0.0,0.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--peak_table",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",required=True); ap.add_argument("--me5_lookup",required=True)
    ap.add_argument("--gene_baselines",required=True)
    ap.add_argument("--pc_genes_bed",required=True)
    ap.add_argument("--upstream_nt",type=int,default=3000); ap.add_argument("--downstream_nt",type=int,default=1000)
    ap.add_argument("--n_bins",type=int,default=8); ap.add_argument("--center_offset",type=int,default=249)
    ap.add_argument("--features",nargs="+",
                    default=["upstream_mean_u1__wg","pas_prox_u1__wg","upstream_max_me3__wg","term_3ss__wg"])
    ap.add_argument("--jitters",nargs="+",type=int,default=[0,250,500,1000,2000])
    ap.add_argument("--out",default="jitter_control.tsv")
    a=ap.parse_args()

    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import (GenomeFasta,load_u1_lookup,load_me5_lookup,
        init_maxentscan,extract_one_site)
    from compute_gene_baselines import derive_axes
    from sklearn.linear_model import LogisticRegression

    df=pd.read_parquet(a.peak_table)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"].astype(str))
    df[a.features]=df[a.features].fillna(0.0)
    tr=df[~df.chrom.isin(TRAIN_EXCL)]
    clf=LogisticRegression(max_iter=1000,class_weight="balanced").fit(
        tr[a.features].to_numpy(), tr["in_zone"].astype(int).to_numpy())

    te=df[df.chrom.isin(TEST_CHROMS)].copy()
    te["splice_prob"]=clf.predict_proba(te[a.features].to_numpy())[:,1]
    def cmarg(g):
        c=g.loc[g.in_zone==1,"rf_prob"]; i=g.loc[g.in_zone==0,"rf_prob"]
        if len(c)==0 or len(i)==0: return np.nan
        return float(c.max()-i.max())
    cm=te.groupby("gene").apply(cmarg,include_groups=False)
    contested=set(cm[cm<=0].index)

    fa=GenomeFasta(a.reference_fasta); u1=load_u1_lookup(a.u1_lookup); me5=load_me5_lookup(a.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    base=pd.read_csv(a.gene_baselines,sep="\t").drop_duplicates("gene",keep="first").set_index("gene")
    pc=pd.read_csv(a.pc_genes_bed,sep="\t",header=None,
                   names=["chrom","start","end","name","score","strand"])
    pc["gene"]=pc["name"].astype(str).str.split("_").str[-1]; pc=pc.drop_duplicates("gene").set_index("gene")
    AX=["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt","local_me5",
        "pas_prox_u1","dn5ss_has_strong","interv_me5_max"]

    def splice_at(gene,chrom,strand,canon_pos,jit):
        p = canon_pos + (jit if strand=="+" else -jit)
        center = p + a.center_offset if strand=="+" else p - a.center_offset
        if strand=="+": w0=max(0,center-a.upstream_nt); w1=center+a.downstream_nt
        else:           w0=max(0,center-a.downstream_nt); w1=center+a.upstream_nt
        seq=fa.fetch(chrom,w0,w1,strand=strand)
        if seq is None or len(seq)<(a.upstream_nt+a.downstream_nt-100): return None
        feat=extract_one_site(seq,a.upstream_nt,a.downstream_nt,a.n_bins,me5,score3,m3,u1)
        ax=derive_axes(feat,seq,a.upstream_nt); row={}
        for k in AX:
            v=ax.get(k,np.nan); bcol=f"baseline_{k}"
            bv=base.at[gene,bcol] if (gene in base.index and bcol in base.columns) else np.nan
            if np.isfinite(v) and pd.notna(bv):
                row[f"{k}__wg"]=float(v)-float(bv)
            else:
                row[f"{k}__wg"]=0.0
        X=np.array([[row.get(f,0.0) for f in a.features]],dtype=float)
        X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
        return float(clf.predict_proba(X)[:,1][0])

    rows=[]
    for jit in a.jitters:
        hit=0; n=0
        for gene,g in te[te.gene.isin(contested)].groupby("gene",sort=False):
            if g.in_zone.sum()==0: continue
            g=g.reset_index(drop=True)
            ci=int(g.index[g.in_zone==1][0]); chrom=g.chrom.iloc[0]
            strand=pc.loc[gene,"strand"] if gene in pc.index else "+"
            canon_pos=int(g.position.iloc[ci])
            sp=g["splice_prob"].to_numpy().copy()
            if jit!=0:
                v=splice_at(gene,chrom,strand,canon_pos,jit)
                if v is None: continue
                sp[ci]=v
            crf=float(g.loc[ci,"rf_prob"]); cand=np.where((g.in_zone==1)|(g.rf_prob>=crf))[0]
            win=cand[int(np.argmax(sp[cand]))]
            hit+=int(g.loc[win,"in_zone"]==1); n+=1
        p,lo,hi=wilson(hit,n)
        rows.append({"jitter_nt":jit,"contested_n":n,"recovered":hit,"recovery":p,"ci_lo":lo,"ci_hi":hi})
        print(f"  jitter {jit:>5} nt: recovery {p:.3f} [{lo:.3f},{hi:.3f}]  (n={n})",file=sys.stderr)
    out=pd.DataFrame(rows); out.to_csv(a.out,sep="\t",index=False)
    r0=out[out.jitter_nt==0].recovery.iloc[0]; rmax=out.recovery.iloc[-1]
    print(f"\n[interpretation] recovery at 0nt={r0:.3f}, at {a.jitters[-1]}nt={rmax:.3f}.",file=sys.stderr)
    print(f"  {'DEGRADES with distance -> signal is positional/real' if rmax<r0-0.03 else 'FLAT -> not positional; investigate leakage'}",file=sys.stderr)
    print(f"[out] {a.out}",file=sys.stderr)

if __name__=="__main__": main()