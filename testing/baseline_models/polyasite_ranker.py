#!/usr/bin/env python3
# within-gene dominant-site ranking on the polyasite atlas
import argparse, os, sys, re, math, tempfile
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# constants
TEST_CHROMS = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS  = {"chr3","chr13"}
PAS_TIERS = {"AATAAA":1.0, "ATTAAA":0.8}
PAS_SECONDARY = {"AGTAAA","TATAAA","CATAAA","GATAAA","AATATA","AATACA",
                 "AATAGA","ACTAAA","AAGAAA","AATGAA"}
N_FIXED_COLS = 13

# small stats
def wilson(k, n, z=1.96):
    if n == 0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, max(0,c-h), min(1,c+h)

def boot_diff_ci(hits_a, hits_b, n_boot=2000, seed=0):
    rng=np.random.default_rng(seed); a=np.asarray(hits_a); b=np.asarray(hits_b)
    n=len(a); d=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n); d.append(b[idx].mean()-a[idx].mean())
    return float(b.mean()-a.mean()), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

# atlas loading
def load_atlas(path):
    first=open(path).readline()
    has_header=first.lower().startswith(("chrom","#chrom"))
    sample_meta=[]
    if has_header:
        raw=pd.read_csv(path,sep="\t",header=0,dtype=str,low_memory=False)
        rc={c.lower().lstrip("#"):c for c in raw.columns}
        def col(*names):
            for n in names:
                if n in rc: return rc[n]
            return None
        cc=dict(chrom=col("chrom"),strand=col("strand"),repsite=col("rep","repsite"),
                tpm=col("score","tpm","expression"),ann=col("annotation","ann"),
                sym=col("gene_name","gene_symbol"),ens=col("gene_id"),
                pas=col("repsite_signals","pas"))
        need=[k for k in ("chrom","strand","repsite","tpm") if cc[k] is None]
        if need: sys.exit(f"[err] missing {need}; cols={list(raw.columns)[:14]}")
        df=pd.DataFrame({k:(raw[cc[k]] if cc[k] else np.nan) for k in cc})
        fixed=set(v for v in cc.values() if v)
        sample_cols=[c for c in raw.columns if c not in fixed]
        sample_cols=[c for c in sample_cols if "|" in str(c)]
        smat=raw[sample_cols].apply(pd.to_numeric,errors="coerce").fillna(0.0) if sample_cols else None
        for j,c in enumerate(sample_cols):
            parts=str(c).split("|")
            sample_meta.append(dict(col=c, sid=parts[0] if len(parts)>0 else c,
                                    protocol=parts[1] if len(parts)>1 else "",
                                    source=parts[2] if len(parts)>2 else "",
                                    treatment=parts[4] if len(parts)>4 else ""))
        print(f"[atlas] full TSV; {len(sample_cols)} per-sample columns parsed",file=sys.stderr)
    else:
        raw=pd.read_csv(path,sep="\t",header=None,dtype=str)
        NAMES=["chrom","start","end","cid","tpm","strand","repsite","supp_pct","n_prot",
               "tpm2","ann","sym","ens","pas"]
        ncol=raw.shape[1]
        df=raw.iloc[:,:min(ncol,14)].copy(); df.columns=NAMES[:df.shape[1]]
        for m in NAMES[df.shape[1]:]: df[m]=np.nan
        smat=None
        if ncol<12: print(f"[atlas] WARNING {ncol}-col bed, no gene id; cannot group by gene",file=sys.stderr)
    df["chrom"]=df["chrom"].astype(str)
    df["chrom"]=df["chrom"].where(df["chrom"].str.startswith("chr"),"chr"+df["chrom"])
    df["repsite"]=pd.to_numeric(df["repsite"],errors="coerce")
    df["tpm"]=pd.to_numeric(df["tpm"],errors="coerce")
    df=df.dropna(subset=["repsite","tpm"]).reset_index(drop=True)
    if smat is not None: smat=smat.loc[df.index].reset_index(drop=True)
    return df, sample_meta, smat

def clean_gene(df):
    def _c(s): return s.where(s.notna() & (s.astype(str)!="NA") & (s.astype(str)!="nan"))
    g=_c(df["ens"]).fillna(_c(df["sym"])); df=df.copy(); df["gene"]=g
    n_missing=df["gene"].isna().sum()
    if n_missing: print(f"[atlas] {n_missing:,} sites no gene id (mostly intergenic/antisense); dropped",file=sys.stderr)
    return df[df["gene"].notna()].reset_index(drop=True)

# pas architecture features
def parse_pas(field):
    out=dict(pas_n=0, pas_canon=0, pas_strong=0, pas_tier=0.0, pas_bestdist=np.nan,
             pas_secondary=0)
    if not isinstance(field,str) or field=="NA": return out
    dists=[]; best_tier=0.0
    for tok in field.split(";"):
        m=re.match(r"([ACGT]{6})@(-?\d+)@",tok)
        if not m: continue
        sig=m.group(1); d=abs(int(m.group(2))); out["pas_n"]+=1; dists.append(d)
        if sig in PAS_TIERS:
            out["pas_canon"]=1; best_tier=max(best_tier,PAS_TIERS[sig])
            if sig=="AATAAA": out["pas_strong"]=1
        elif sig in PAS_SECONDARY:
            out["pas_secondary"]=1; best_tier=max(best_tier,0.4)
    out["pas_tier"]=best_tier
    out["pas_bestdist"]=min(dists) if dists else np.nan
    return out
PAS_COLS=["pas_n","pas_canon","pas_strong","pas_secondary","pas_tier","pas_bestdist"]

def label_tpm(df):
    lab=pd.Series(0,index=df.index)
    lab.loc[df.groupby("gene")["tpm"].idxmax()]=1
    return lab

def label_te(df):
    keep=[]; lab=pd.Series(0,index=df.index)
    for g,gg in df.groupby("gene",sort=False):
        te=gg[gg["ann"]=="TE"]
        if len(te)==0 or len(gg)<2: continue
        lab.loc[te["tpm"].idxmax()]=1; keep.append(g)
    return lab, set(keep)

def label_celltype(df, smat, sample_meta, source_substr):
    if smat is None: sys.exit("[err] --target celltype needs the full TSV with per-sample columns")
    cols=[m["col"] for m in sample_meta if source_substr.lower() in m["source"].lower()]
    if not cols: sys.exit(f"[err] no samples match source '{source_substr}'")
    ct=smat[cols].sum(axis=1); df=df.assign(_ct=ct.values)
    lab=pd.Series(0,index=df.index); keep=[]
    for g,gg in df.groupby("gene",sort=False):
        nz=gg[gg["_ct"]>0]
        if len(nz)<2: continue
        lab.loc[nz["_ct"].idxmax()]=1; keep.append(g)
    print(f"[target celltype='{source_substr}'] {len(cols)} sample cols, {len(keep)} genes",file=sys.stderr)
    return lab, set(keep)

def compute_splice(df, args, scripts_dir):
    sys.path.insert(0,scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan)
    from compute_gene_baselines import derive_axes
    import splice_extra_features as XF
    from splice_common import compute_splice_row, SPLICE_COLS
    fa=GenomeFasta(args.reference_fasta); u1=load_u1_lookup(args.u1_lookup); me5=load_me5_lookup(args.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    rows=[]
    for _,r in tqdm(df.iterrows(),total=len(df),desc="splice"):
        c=int(r["repsite"])-1; strand=r["strand"]; chrom=r["chrom"]
        if strand=="+": w0=max(0,c-args.upstream_nt); w1=c+args.downstream_nt
        else:           w0=max(0,c-args.downstream_nt); w1=c+args.upstream_nt
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: seq=None
        rows.append(compute_splice_row(seq,args.upstream_nt,args.downstream_nt,args.n_bins,
                                       me5,score3,m3,u1,derive_axes,XF))
    return pd.DataFrame(rows,columns=SPLICE_COLS,index=df.index), SPLICE_COLS

def compute_cpa(df, args, scripts_dir):
    sys.path.insert(0,scripts_dir)
    from extract_splicing_features import GenomeFasta
    from featurization import read_pwms, featurize_fasta_to_df
    fa=GenomeFasta(args.reference_fasta)
    pwms=read_pwms(args.pwms,max_len=20,bg=[0.3,0.2,0.2,0.3])
    recs=[]; order=[]
    for i,r in df.iterrows():
        c=int(r["repsite"])-1; strand=r["strand"]; chrom=r["chrom"]
        s0=c-args.win500//2; s1=s0+args.win500
        seq=fa.fetch(chrom,s0,s1,strand=strand)
        if seq is None or len(seq)<args.win500: continue
        recs.append((f"row{i}",seq)); order.append(i)
    with tempfile.NamedTemporaryFile("w",suffix=".fa",delete=False) as tf:
        for rid,s in recs: tf.write(f">{rid}\n{s}\n")
        tmp=tf.name
    X=featurize_fasta_to_df(tmp,pwms,bg=[0.3,0.2,0.2,0.3],verbose=False); os.unlink(tmp)
    X=X.sort_index(axis=1); X.index=order
    X=X.reindex(df.index)
    return X, list(X.columns)

def direction_prior(feats):
    P={"spl_u1_suppression_up":-1,"spl_u1_suppression_down":-1,"spl_bp_strength":+1,
       "spl_u2af65_ppt":+1,"spl_u2af35_ag":+1,"spl_term_3ss":+1,"spl_upstream_max_me3":+1,
       "pas_canon":+1,"pas_strong":+1,"pas_n":+1,"pas_tier":+1,"pas_secondary":+1}
    return [int(P.get(f,0)) for f in feats]

def fit_predict(kind, Xtr, ytr, feats, monotone=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    if kind=="logistic":
        sc=StandardScaler().fit(Xtr); m=LogisticRegression(max_iter=2000,class_weight="balanced")
        m.fit(sc.transform(Xtr),ytr)
        if monotone:
            co=m.coef_[0].copy(); mc=np.array(direction_prior(feats),float)
            bad=(mc!=0)&(np.sign(co)!=np.sign(mc)); co[bad]=0.0; m.coef_[0]=co
        return lambda X: m.predict_proba(sc.transform(X))[:,1]
    mc=direction_prior(feats) if monotone else None
    m=HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,
        min_samples_leaf=50,class_weight="balanced",monotonic_cst=mc,random_state=0)
    m.fit(Xtr,ytr); return lambda X: m.predict_proba(X)[:,1]

def p1_mrr_hits(df, scol):
    hits=[]; rr=[]; rnd=[]
    for g,gg in df.groupby("gene",sort=False):
        if gg["is_dom"].sum()==0: continue
        order=np.argsort(gg[scol].to_numpy())[::-1]
        lab=gg["is_dom"].to_numpy()[order]
        r=int(np.argmax(lab==1))+1
        hits.append(int(r==1)); rr.append(1.0/r); rnd.append(1.0/len(gg))
    return np.array(hits), np.array(rr), np.array(rnd)

def eval_block(df, scol, label):
    h,rr,rnd=p1_mrr_hits(df,scol); n=len(h)
    p,lo,hi=wilson(int(h.sum()),n)
    return dict(set=label, n_genes=n, p1=p, ci_lo=lo, ci_hi=hi, mrr=float(rr.mean()),
                random=float(rnd.mean()), hits=h)

def by_contest_size(df, scol):
    df=df.copy(); df["_csz"]=df.groupby("gene")["gene"].transform("size")
    out=[]
    for lo,hi,lab in [(2,2,"2"),(3,4,"3-4"),(5,8,"5-8"),(9,9999,"9+")]:
        sub=df[(df["_csz"]>=lo)&(df["_csz"]<=hi)]
        if sub.gene.nunique()==0: continue
        h,_,rnd=p1_mrr_hits(sub,scol)
        out.append((lab,len(h),h.mean() if len(h) else 0,rnd.mean() if len(rnd) else 0))
    return out

def by_annotation(df, scol):
    dom=df[df.is_dom==1][["gene","ann"]].set_index("gene")["ann"]
    h,_,_=p1_mrr_hits(df,scol)
    rows=[]; genes=[]
    for g,gg in df.groupby("gene",sort=False):
        if gg.is_dom.sum()==0: continue
        order=np.argsort(gg[scol].to_numpy())[::-1]
        rows.append(int(gg.is_dom.to_numpy()[order][0]==1)); genes.append(g)
    hh=pd.DataFrame({"gene":genes,"hit":rows}).set_index("gene")
    hh["ann"]=dom; out=[]
    for ann,gg in hh.groupby("ann"):
        out.append((ann,len(gg),gg.hit.mean()))
    return sorted(out,key=lambda x:-x[1])

def screen(df, feats, n_boot=1000, seed=0):
    rng=np.random.default_rng(seed)
    genes=[]; cand=[]; cpos=[]; msz=[]
    for g,gg in df.groupby("gene",sort=False):
        if gg.is_dom.sum()!=1: continue
        genes.append(g); idx=np.arange(len(gg))
        cand.append(gg); cpos.append(int(np.argmax(gg.is_dom.to_numpy()))); msz.append(len(gg))
    msz=np.array(msz); null=float(np.mean(1.0/msz))
    rows=[]
    for f in feats:
        best=None
        for sign in (+1,-1):
            w=[]
            for gg,cp in zip(cand,cpos):
                v=gg[f].to_numpy()*sign
                if not np.isfinite(v).any(): w.append(0); continue
                w.append(int(np.nanargmax(v)==cp))
            w=np.array(w); diff=w.mean()-null
            d=[]; nullvec=1.0/msz
            for _ in range(n_boot):
                ix=rng.integers(0,len(w),len(w)); d.append(w[ix].mean()-nullvec[ix].mean())
            lo,hi=np.percentile(d,[2.5,97.5])
            if best is None or diff>best[1]: best=(sign,diff,w.mean(),lo,hi)
        sign,diff,wr,lo,hi=best
        verdict="PROMISING" if lo>0 else ("dead" if hi<0.02 else "weak")
        rows.append(dict(feature=f,polarity=sign,win_rate=wr,null=null,
                         win_minus_null=diff,ci_lo=lo,ci_hi=hi,verdict=verdict))
    return pd.DataFrame(rows)

def jitter_control(df, args, scripts_dir, model_predict, feats, jitters=(0,500,2000)):
    sys.path.insert(0,scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan)
    from compute_gene_baselines import derive_axes
    import splice_extra_features as XF
    from splice_common import compute_splice_row, SPLICE_COLS
    fa=GenomeFasta(args.reference_fasta); u1=load_u1_lookup(args.u1_lookup); me5=load_me5_lookup(args.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    dom=df[df.is_dom==1]
    out=[]
    for jit in jitters:
        recov=[]
        for g,gg in df.groupby("gene",sort=False):
            if gg.is_dom.sum()!=1: continue
            r=gg[gg.is_dom==1].iloc[0]; c=int(r["repsite"])-1
            shift=jit if r["strand"]=="+" else -jit; c2=c+shift
            strand=r["strand"]; chrom=r["chrom"]
            if strand=="+": w0=max(0,c2-args.upstream_nt); w1=c2+args.downstream_nt
            else:           w0=max(0,c2-args.downstream_nt); w1=c2+args.upstream_nt
            try: seq=fa.fetch(chrom,w0,w1,strand=strand)
            except Exception: seq=None
            srow=compute_splice_row(seq,args.upstream_nt,args.downstream_nt,args.n_bins,me5,score3,m3,u1,derive_axes,XF)
            gg2=gg.copy()
            for col in SPLICE_COLS:
                if col in gg2.columns:
                    gg2.loc[gg2.is_dom==1,col]=srow.get(col,np.nan)
            X=gg2[feats].fillna(0.0).to_numpy(); s=model_predict(X)
            recov.append(int(np.argmax(s)==int(np.argmax(gg2.is_dom.to_numpy()))))
        recov=np.array(recov); p,lo,hi=wilson(int(recov.sum()),len(recov))
        out.append((jit,len(recov),p,lo,hi)); 
        print(f"  jitter {jit:>5} nt: recovery {p:.3f} [{lo:.3f},{hi:.3f}] (n={len(recov)})",file=sys.stderr)
    return out

def tissue_specificity_stat(df, smat, sample_meta, top_sources=8):
    if smat is None: return []
    from collections import Counter
    srcs=Counter(m["source"] for m in sample_meta if m["source"])
    sources=[s for s,_ in srcs.most_common(top_sources)]
    cross_idx={g:gg.index[int(np.argmax(gg["tpm"].to_numpy()))]
               for g,gg in df.groupby("gene",sort=False)}
    out=[]
    for src in sources:
        cols=[m["col"] for m in sample_meta if m["source"]==src]
        if not cols: continue
        ct=smat[cols].sum(axis=1); dd=df.assign(_ct=ct.values)
        diff=0; n=0
        for g,gg in dd.groupby("gene",sort=False):
            nz=gg[gg["_ct"]>0]
            if len(nz)<2: continue
            ct_dom=nz.index[int(np.argmax(nz["_ct"].to_numpy()))]
            n+=1; diff+=int(ct_dom!=cross_idx[g])
        if n>0: out.append((src,n,diff/n))
    return out

def usage_decisiveness(df):
    fr=[]
    for g,gg in df.groupby("gene",sort=False):
        t=np.sort(gg["tpm"].to_numpy())[::-1]
        if t.sum()<=0: continue
        fr.append(t[0]/t.sum())
    fr=np.array(fr)
    return dict(mean_top_frac=float(fr.mean()), median_top_frac=float(np.median(fr)),
                frac_genes_top_below_0p5=float((fr<0.5).mean()))

def build_feature_sets(have_pas, have_splice, have_cpa):
    fs={}
    if have_pas: fs["pas"]=PAS_COLS
    if have_splice:
        from splice_common import SPLICE_COLS as SC; fs["splice"]=SC
    if have_cpa: fs["cpa"]="__CPA__"
    return fs

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--atlas",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",required=True); ap.add_argument("--me5_lookup",required=True)
    ap.add_argument("--pwms",default=None,help="PWM list; required for --cpa_features")
    ap.add_argument("--cpa_features",action="store_true",help="add RF PWM+kmer features per site")
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--win500",type=int,default=500); ap.add_argument("--n_bins",type=int,default=8)
    ap.add_argument("--targets",default="tpm",help="comma list of tpm,te,celltype")
    ap.add_argument("--cell_type",default="",help="SOURCE substring for --targets celltype (e.g. 'HeLa')")
    ap.add_argument("--min_sites",type=int,default=2); ap.add_argument("--min_tpm",type=float,default=0.0)
    ap.add_argument("--restrict_classes",default="")
    ap.add_argument("--models",default="logistic,gbt")
    ap.add_argument("--monotone",action="store_true")
    ap.add_argument("--jitter",action="store_true",help="run splice jitter control on the winning set")
    ap.add_argument("--limit_genes",type=int,default=0)
    ap.add_argument("--out",default="polyasite")
    a=ap.parse_args()

    df,sample_meta,smat=load_atlas(a.atlas)
    df=clean_gene(df)
    if smat is not None: smat=smat.loc[df.index].reset_index(drop=True); df=df.reset_index(drop=True)
    if a.restrict_classes:
        keep=set(a.restrict_classes.split(",")); m=df["ann"].isin(keep)
        df=df[m].reset_index(drop=True);  smat=smat[m.values].reset_index(drop=True) if smat is not None else None
    if a.min_tpm>0:
        m=df["tpm"]>=a.min_tpm; df=df[m].reset_index(drop=True)
        smat=smat[m.values].reset_index(drop=True) if smat is not None else None
    vc=df.groupby("gene").size(); keep_g=set(vc[vc>=a.min_sites].index)
    m=df["gene"].isin(keep_g); df=df[m].reset_index(drop=True)
    smat=smat[m.values].reset_index(drop=True) if smat is not None else None
    if a.limit_genes>0:
        gene_chrom=df.drop_duplicates("gene").set_index("gene")["chrom"]
        by_chrom={}
        for g,c in gene_chrom.items(): by_chrom.setdefault(c,[]).append(g)
        order=[]; pools=[list(v) for v in by_chrom.values()]
        i=0
        while len(order)<a.limit_genes and any(pools):
            p=pools[i%len(pools)]
            if p: order.append(p.pop(0))
            i+=1
            pools=[p for p in pools if p] or []
            if not pools: break
        keep=set(order); m=df["gene"].isin(keep)
        df=df[m].reset_index(drop=True); smat=smat[m.values].reset_index(drop=True) if smat is not None else None
        n_test=df[df.chrom.isin(TEST_CHROMS)].gene.nunique()
        print(f"[atlas] limited to {len(keep)} genes across chroms ({n_test} on test chroms)",file=sys.stderr)
    print(f"[atlas] {len(df):,} sites, {df.gene.nunique():,} genes (>= {a.min_sites} sites)",file=sys.stderr)
    if len(df)==0: sys.exit("[err] no sites after filtering")
    dec=usage_decisiveness(df)
    print("[decisiveness] dominant site is on avg {:.2f} of gene TPM; {:.2f} of genes "
          "have no majority site (intrinsically near random)".format(
          dec["mean_top_frac"], dec["frac_genes_top_below_0p5"]),file=sys.stderr)
    if smat is not None:
        ts=tissue_specificity_stat(df,smat,sample_meta)
        if ts:
            print("[tissue-specificity] frac of genes whose celltype-dominant site differs "
                  "from the cross-sample dominant (high = a sequence-only ceiling):",file=sys.stderr)
            for s_src,s_n,s_f in ts:
                print("    {:<28} n={:<5} diff={:.2f}".format(s_src[:28],s_n,s_f),file=sys.stderr)
            pd.DataFrame(ts,columns=["source","n_genes","frac_diff_from_crosssample"]).to_csv(
                a.out+"_tissue_specificity.tsv",sep="\t",index=False)

    pas=df["pas"].apply(parse_pas); 
    for c in PAS_COLS: df[c]=[p[c] for p in pas]
    have_pas=True

    Sdf,SPLICE_COLS=compute_splice(df,a,a.scripts_dir); df=pd.concat([df,Sdf],axis=1); have_splice=True

    cpa_cols=[]
    if a.cpa_features:
        if not a.pwms: sys.exit("[err] --cpa_features needs --pwms")
        Cdf,cpa_cols=compute_cpa(df,a,a.scripts_dir); df=pd.concat([df,Cdf],axis=1)
    have_cpa=bool(cpa_cols)
    df.to_parquet(f"{a.out}_sites.parquet",index=False)

    FS={"pas":PAS_COLS, "splice":SPLICE_COLS}
    if have_cpa: FS["cpa"]=cpa_cols
    FS["splice+pas"]=SPLICE_COLS+PAS_COLS
    if have_cpa:
        FS["cpa+splice"]=cpa_cols+SPLICE_COLS
        FS["cpa+pas"]=cpa_cols+PAS_COLS
        FS["cpa+pas+splice"]=cpa_cols+PAS_COLS+SPLICE_COLS
    for k in FS: df[FS[k]]=df[FS[k]].apply(pd.to_numeric,errors="coerce")

    df=df.copy()
    df["_split"]=np.where(df.chrom.isin(TEST_CHROMS),"test",
                  np.where(df.chrom.isin(VAL_CHROMS),"val","train"))
    models=a.models.split(",")
    all_res=[]
    targets=a.targets.split(",")

    for tgt in targets:
        if tgt=="tpm":
            df["is_dom"]=label_tpm(df).values; sub=df
        elif tgt=="te":
            lab,keep=label_te(df); df["is_dom"]=lab.values; sub=df[df.gene.isin(keep)]
        elif tgt=="celltype":
            if not a.cell_type: sys.exit("[err] --targets celltype needs --cell_type")
            lab,keep=label_celltype(df,smat,sample_meta,a.cell_type); df["is_dom"]=lab.values
            sub=df[df.gene.isin(keep)]
        else: sys.exit(f"[err] unknown target {tgt}")
        tr=sub[sub._split=="train"]; te=sub[sub._split=="test"]
        if len(te)==0 or te.is_dom.sum()==0:
            print(f"[target {tgt}] no test data/positives; skipped",file=sys.stderr); continue
        print(f"\n========== TARGET = {tgt}  (test genes={te.gene.nunique()}) ==========",file=sys.stderr)
        print(f"{'model':<10}{'feature_set':<18}{'P@1':>8}{'95% CI':>18}{'MRR':>8}{'random':>9}{'vs_random':>10}",file=sys.stderr)
        per_set_hits={}
        for kind in models:
            for setname,feats in FS.items():
                feats=[f for f in feats if f in tr.columns]
                Xtr=tr[feats].fillna(0.0).to_numpy(); ytr=tr.is_dom.to_numpy()
                if ytr.sum()==0: continue
                pred=fit_predict(kind,Xtr,ytr,feats,monotone=a.monotone)
                te2=te.copy(); te2["s"]=pred(te2[feats].fillna(0.0).to_numpy())
                blk=eval_block(te2,"s",f"{kind}:{setname}")
                per_set_hits[(kind,setname)]=(blk["hits"],te2)
                print(f"{kind:<10}{setname:<18}{blk['p1']:>8.3f}   [{blk['ci_lo']:.3f},{blk['ci_hi']:.3f}]"
                      f"{blk['mrr']:>8.3f}{blk['random']:>9.3f}{blk['p1']-blk['random']:>+10.3f}",file=sys.stderr)
                blk.pop("hits"); blk["target"]=tgt; blk["model"]=kind; all_res.append(blk)
        print("  [does splice ADD over the baseline? paired bootstrap on per-gene P@1]",file=sys.stderr)
        for base,plus in [("pas","splice+pas"),("cpa","cpa+splice"),("cpa+pas","cpa+pas+splice")]:
            for kind in models:
                if (kind,base) in per_set_hits and (kind,plus) in per_set_hits:
                    ha,_=per_set_hits[(kind,base)]; hb,_=per_set_hits[(kind,plus)]
                    d,lo,hi=boot_diff_ci(ha,hb)
                    sig="*" if lo>0 else ("(worse)" if hi<0 else "")
                    print(f"    {kind:<10}{base:>14} -> {plus:<16} dP@1={d:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}",file=sys.stderr)
        best=max((r for r in all_res if r["target"]==tgt), key=lambda r:r["p1"])
        bk,bte=per_set_hits[(best["model"],best["set"].split(":")[1])]
        print(f"  [best={best['set']}] by contest size:",file=sys.stderr)
        for lab,n,p,rnd in by_contest_size(bte,"s"):
            print(f"      sites={lab:<5} n={n:<5} P@1={p:.3f} random={rnd:.3f}",file=sys.stderr)
        print(f"  by dominant-site annotation:",file=sys.stderr)
        for ann,n,p in by_annotation(bte,"s"):
            print(f"      {ann:<4} n={n:<5} P@1={p:.3f}",file=sys.stderr)
        if tgt=="tpm":
            scr=screen(te, [c for c in (SPLICE_COLS+PAS_COLS) if c in te.columns])
            scr.to_csv(f"{a.out}_screen_{tgt}.tsv",sep="\t",index=False)
            print("  [feature screen -> %s_screen_%s.tsv] promising:"%(a.out,tgt),
                  list(scr[scr.verdict=="PROMISING"].feature),file=sys.stderr)
        if a.jitter and "splice" in best["set"]:
            feats=[f for f in FS[best["set"].split(":")[1]] if f in tr.columns]
            pred=fit_predict(best["model"],tr[feats].fillna(0.0).to_numpy(),tr.is_dom.to_numpy(),feats,monotone=a.monotone)
            print("  [jitter control on winning set]",file=sys.stderr)
            jitter_control(te,a,a.scripts_dir,pred,feats)

    pd.DataFrame(all_res).to_csv(f"{a.out}_p1.tsv",sep="\t",index=False)
    print(f"\n[out] {a.out}_sites.parquet, {a.out}_p1.tsv",file=sys.stderr)

if __name__=="__main__": main()