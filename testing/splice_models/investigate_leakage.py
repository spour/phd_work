#!/usr/bin/env python3
# diagnose the flat jitter control
import argparse, os, sys, math
import numpy as np, pandas as pd

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}
TRAIN_EXCL=TEST_CHROMS|{"chr3","chr13"}
FEATS_DEFAULT=["upstream_mean_u1__wg","pas_prox_u1__wg","upstream_max_me3__wg","term_3ss__wg"]
AX_OF={ "upstream_mean_u1__wg":"upstream_mean_u1","pas_prox_u1__wg":"pas_prox_u1",
        "upstream_max_me3__wg":"upstream_max_me3","term_3ss__wg":"term_3ss"}
ALL_AX=["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt","local_me5",
        "pas_prox_u1","dn5ss_has_strong","interv_me5_max"]

def aucroc(pos,neg):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    pos=pos[np.isfinite(pos)]; neg=neg[np.isfinite(neg)]
    if len(pos)==0 or len(neg)==0: return float("nan")
    a=np.concatenate([pos,neg]); l=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    o=np.argsort(a); r=np.empty(len(a)); r[o]=np.arange(1,len(a)+1)
    return (r[l==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--peak_table",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",required=True); ap.add_argument("--me5_lookup",required=True)
    ap.add_argument("--gene_baselines",required=True); ap.add_argument("--pc_genes_bed",required=True)
    ap.add_argument("--upstream_nt",type=int,default=3000); ap.add_argument("--downstream_nt",type=int,default=1000)
    ap.add_argument("--n_bins",type=int,default=8); ap.add_argument("--center_offset",type=int,default=249)
    ap.add_argument("--features",nargs="+",default=FEATS_DEFAULT)
    ap.add_argument("--n_genes",type=int,default=40)
    ap.add_argument("--jit_far",type=int,default=2000)
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
    print(f"[model] coef: "+", ".join(f"{f}={w:+.3f}" for f,w in zip(a.features,clf.coef_[0])),file=sys.stderr)

    te=df[df.chrom.isin(TEST_CHROMS)].copy()
    fa=GenomeFasta(a.reference_fasta); u1=load_u1_lookup(a.u1_lookup); me5=load_me5_lookup(a.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    base=pd.read_csv(a.gene_baselines,sep="\t").drop_duplicates("gene",keep="first").set_index("gene")
    pc=pd.read_csv(a.pc_genes_bed,sep="\t",header=None,
                   names=["chrom","start","end","name","score","strand"])
    pc["gene"]=pc["name"].astype(str).str.split("_").str[-1]; pc=pc.drop_duplicates("gene").set_index("gene")

    def raw_axes(gene,chrom,strand,pos,jit):
        p = pos + (jit if strand=="+" else -jit)
        center = p + a.center_offset if strand=="+" else p - a.center_offset
        if strand=="+": w0=max(0,center-a.upstream_nt); w1=center+a.downstream_nt
        else:           w0=max(0,center-a.downstream_nt); w1=center+a.upstream_nt
        seq=fa.fetch(chrom,w0,w1,strand=strand)
        if seq is None or len(seq)<(a.upstream_nt+a.downstream_nt-100): return None
        feat=extract_one_site(seq,a.upstream_nt,a.downstream_nt,a.n_bins,me5,score3,m3,u1)
        return derive_axes(feat,seq,a.upstream_nt)

    def wg_and_prob(gene,raw):
        row={}
        for f in a.features:
            ax=AX_OF[f]; v=raw.get(ax,np.nan); bcol=f"baseline_{ax}"
            bv=base.at[gene,bcol] if (gene in base.index and bcol in base.columns) else np.nan
            row[f]=float(v)-float(bv) if (np.isfinite(v) and pd.notna(bv)) else 0.0
        X=np.nan_to_num(np.array([[row[f] for f in a.features]],float))
        return row, float(clf.predict_proba(X)[:,1][0])

    def cmarg(g):
        c=g.loc[g.in_zone==1,"rf_prob"]; i=g.loc[g.in_zone==0,"rf_prob"]
        if len(c)==0 or len(i)==0: return np.nan
        return float(c.max()-i.max())
    cm=te.groupby("gene").apply(cmarg,include_groups=False)
    contested=[g for g in cm[cm<=0].index if g in pc.index][:a.n_genes]

    # check 1: does jitter move the features?
    print("\n==== CHECK 1: does jittering 0 -> {} nt change canonical features/score? ====".format(a.jit_far))
    print(f"{'gene':<16}{'splice@0':>9}{'splice@far':>11}{'|dprob|':>9}  per-feature |delta wg|")
    dprobs=[]; dfeat_acc=[]
    for gene in contested[:min(15,len(contested))]:
        g=te[te.gene==gene]; gr=g[g.in_zone==1]
        if len(gr)==0: continue
        chrom=g.chrom.iloc[0]; strand=pc.loc[gene,"strand"]; pos=int(gr.position.iloc[0])
        r0=raw_axes(gene,chrom,strand,pos,0); rF=raw_axes(gene,chrom,strand,pos,a.jit_far)
        if r0 is None or rF is None: continue
        w0,p0=wg_and_prob(gene,r0); wF,pF=wg_and_prob(gene,rF)
        dfeat=[abs(w0[f]-wF[f]) for f in a.features]; dfeat_acc.append(dfeat); dprobs.append(abs(p0-pF))
        print(f"{gene:<16}{p0:>9.3f}{pF:>11.3f}{abs(p0-pF):>9.3f}  "+
              " ".join(f"{f.split('__')[0][:10]}:{d:.3f}" for f,d in zip(a.features,dfeat)))
    if dprobs:
        md=float(np.mean(dprobs)); mf=np.mean(dfeat_acc,axis=0)
        print(f"\n  mean |dprob| over genes = {md:.4f}")
        print("  mean |delta wg| per feature = "+", ".join(f"{f.split('__')[0]}:{v:.3f}" for f,v in zip(a.features,mf)))
        print("  VERDICT: "+("jitter MOVES features -> control is valid -> go to CHECK 2"
              if md>0.02 else "jitter BARELY moves features -> control is INERT (window not shifting). FIX THE CONTROL, result may still be fine."))

    # check 2: site term vs baseline term
    print("\n==== CHECK 2: is canonical-vs-internal separation site-level or gene-level? ====")
    print("  AUROC > 0.5 means canonical scores higher on that quantity.")
    print(f"  {'feature':<22}{'wg(model)':>10}{'site':>8}{'baseline':>10}")
    merged=te.copy()
    for f in a.features:
        ax=AX_OF[f]; bcol=f"baseline_{ax}"
        merged[f"__base_{ax}"]=merged.gene.map(base[bcol]) if bcol in base.columns else np.nan
        merged[f"__site_{ax}"]=merged[f]+merged[f"__base_{ax}"]
    canon=merged[merged.in_zone==1]; intern=merged[merged.in_zone==0]
    for f in a.features:
        ax=AX_OF[f]
        a_wg=aucroc(canon[f],intern[f])
        a_site=aucroc(canon[f"__site_{ax}"],intern[f"__site_{ax}"])
        a_base=aucroc(canon[f"__base_{ax}"],intern[f"__base_{ax}"])
        print(f"  {f:<22}{a_wg:>10.3f}{a_site:>8.3f}{a_base:>10.3f}")
    print("\n  READ: if 'baseline' AUROC is far from 0.50, the centered feature inherits")
    print("  gene-level signal (canonical genes differ from internal-only genes in their")
    print("  baselines) -> that is leakage and explains the flat jitter. If 'site' carries")
    print("  the separation and 'baseline' ~0.50, the signal is local and the result holds.")
    print("\n  Within-gene site-level check (the clean test): for each contested gene, is the")
    print("  canonical's SITE value above the max internal SITE value?")
    for f in a.features:
        ax=AX_OF[f]; wins=0; n=0
        for gene,g in merged[merged.gene.isin(contested)].groupby("gene"):
            c=g.loc[g.in_zone==1,f"__site_{ax}"]; i=g.loc[g.in_zone==0,f"__site_{ax}"]
            if len(c)==0 or len(i)==0 or not np.isfinite(c.max()) or not np.isfinite(i.max()): continue
            wins+=int(c.max()>i.max()); n+=1
        print(f"    {f:<22} canonical site beats internal in {wins}/{n} contested genes ({(wins/n if n else 0):.2f})")

if __name__=="__main__": main()