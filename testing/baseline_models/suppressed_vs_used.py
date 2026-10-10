#!/usr/bin/env python3
# the experiment
import argparse, os, sys, math
from collections import defaultdict
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

TEST_CHROMS={"chr18","chr19","chr20","chr21","chr22","chrX"}

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,c-h),min(1,c+h)

def boot_diff(a,b,n_boot=2000,seed=0):
    rng=np.random.default_rng(seed); a=np.asarray(a); b=np.asarray(b)
    d=[]; 
    for _ in range(n_boot):
        d.append(np.mean(rng.choice(b,len(b)))-np.mean(rng.choice(a,len(a))))
    return float(np.mean(b)-np.mean(a)),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))

# gencode introns per gene
def gencode_introns(gtf):
    tx_exons=defaultdict(list); tx_meta={}
    for line in tqdm(open(gtf),desc="gtf"):
        if line.startswith("#"): continue
        f=line.rstrip("\n").split("\t")
        if len(f)<9 or f[2]!="exon": continue
        gn=tx=None
        for kv in f[8].split(";"):
            kv=kv.strip()
            if kv.startswith("gene_name "): gn=kv.split('"')[1]
            elif kv.startswith("transcript_id "): tx=kv.split('"')[1]
        if not gn or not tx: continue
        tx_exons[tx].append((int(f[3]),int(f[4]))); tx_meta[tx]=(f[0],f[6],gn)
    genes=defaultdict(lambda: dict(chrom=None,strand=None,exons=[],term=[]))
    for tx,exons in tx_exons.items():
        chrom,strand,gn=tx_meta[tx]; g=genes[gn]; g["chrom"]=chrom; g["strand"]=strand
        g["exons"]+=exons
        g["term"].append(max(exons,key=lambda e:e[1]) if strand=="+" else min(exons,key=lambda e:e[0]))
    def union(ivs):
        if not ivs: return []
        ivs=sorted(ivs); out=[list(ivs[0])]
        for s,e in ivs[1:]:
            if s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
            else: out.append([s,e])
        return [(s,e) for s,e in out]
    for gn,g in genes.items():
        ex=union(g["exons"]); g["exons"]=ex; g["term"]=union(g["term"])
        introns=[]
        for i in range(len(ex)-1): introns.append((ex[i][1]+1, ex[i+1][0]-1))
        g["introns"]=[(s,e) for s,e in introns if e>s]
        g["span"]=(ex[0][0],ex[-1][1]) if ex else (0,0)
    return genes

def in_any(pos,ivs):
    for s,e in ivs:
        if s<=pos<=e: return True
    return False

def load_atlas_pos(atlas):
    raw=pd.read_csv(atlas,sep="\t",header=0,dtype=str,low_memory=False)
    rc={c.lower().lstrip("#"):c for c in raw.columns}
    def col(*n):
        for x in n:
            if x in rc: return rc[x]
    rep=col("rep","repsite"); chrom=col("chrom")
    d=pd.DataFrame({"chrom":raw[chrom].astype(str),
                    "pos":pd.to_numeric(raw[rep],errors="coerce")-1})
    d["chrom"]=d["chrom"].where(d["chrom"].str.startswith("chr"),"chr"+d["chrom"])
    d=d.dropna()
    return {c:np.sort(gg["pos"].to_numpy()) for c,gg in d.groupby("chrom")}

def near_atlas(chrom,pos,atlas,tol):
    if chrom not in atlas: return False
    P=atlas[chrom]; i=np.searchsorted(P,pos)
    for j in (i-1,i):
        if 0<=j<len(P) and abs(P[j]-pos)<=tol: return True
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--per_gene_dir",required=True)
    ap.add_argument("--gtf",required=True); ap.add_argument("--atlas",required=True)
    ap.add_argument("--pc_genes_bed",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup",required=True); ap.add_argument("--me5_lookup",required=True)
    ap.add_argument("--rf_min",type=float,default=0.5,help="min RF prob = complete grammar")
    ap.add_argument("--atlas_tol",type=int,default=250)
    ap.add_argument("--win",type=int,default=500)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--n_bins",type=int,default=8)
    ap.add_argument("--max_per_gene",type=int,default=20,help="cap high-RF intronic windows per gene")
    ap.add_argument("--max_genes",type=int,default=0,help="0=all (subset for a quick run)")
    ap.add_argument("--out",default="suppressed_vs_used")
    a=ap.parse_args()
    off=a.win//2
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan)
    from compute_gene_baselines import derive_axes
    import splice_extra_features as XF
    from splice_common import compute_splice_row, SPLICE_COLS

    genes_gtf=gencode_introns(a.gtf)
    atlas=load_atlas_pos(a.atlas)
    pc=pd.read_csv(a.pc_genes_bed,sep="\t",header=None,
                   names=["chrom","start","end","name","score","strand"])
    pc["gene"]=pc["name"].astype(str).str.split("_").str[-1]; pc=pc.drop_duplicates("gene").set_index("gene")
    genes=[g for g in genes_gtf if g in pc.index and genes_gtf[g]["chrom"] in TEST_CHROMS and genes_gtf[g]["introns"]]
    if a.max_genes>0: genes=genes[:a.max_genes]
    print(f"[genes] {len(genes):,} test-chrom genes with introns",file=sys.stderr)

    # collect high-rf intronic windows, label used/silent
    recs=[]
    for g in tqdm(genes,desc="collect"):
        p=os.path.join(a.per_gene_dir,f"{g}.parquet")
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=["window_start_genomic","rf_prob"]).dropna(subset=["rf_prob"])
        if len(d)==0: continue
        gg=genes_gtf[g]; chrom=gg["chrom"]
        hi=d[d["rf_prob"]>=a.rf_min]
        if len(hi)==0: continue
        hi=hi.sort_values("rf_prob",ascending=False).head(a.max_per_gene)
        for _,r in hi.iterrows():
            pos=int(r["window_start_genomic"])+off
            if not in_any(pos,gg["introns"]): continue
            used=near_atlas(chrom,pos,atlas,a.atlas_tol)
            host=None
            for s,e in gg["introns"]:
                if s<=pos<=e: host=(s,e); break
            if host is None: continue
            dist_from_5ss = (pos-host[0]) if gg["strand"]=="+" else (host[1]-pos)
            recs.append(dict(gene=g,chrom=chrom,strand=gg["strand"],
                             pos=pos,window_start=int(r["window_start_genomic"]),
                             rf=float(r["rf_prob"]),used=int(used),
                             dist_from_5ss=int(dist_from_5ss)))
    S=pd.DataFrame(recs)
    if len(S)==0: sys.exit("[err] no high-RF intronic windows; lower --rf_min")
    nu=int(S.used.sum()); ns=int((1-S.used).sum())
    print(f"[pop] {len(S):,} high-RF intronic windows: {nu:,} used, {ns:,} silent",file=sys.stderr)

    # splice features at each
    fa=GenomeFasta(a.reference_fasta); u1=load_u1_lookup(a.u1_lookup); me5=load_me5_lookup(a.me5_lookup)
    _,score3,_,m3=init_maxentscan()
    rows=[]
    for _,r in tqdm(S.iterrows(),total=len(S),desc="splice"):
        c=r["pos"]; strand=r["strand"]; chrom=r["chrom"]
        if strand=="+": w0=max(0,c-a.upstream_nt); w1=c+a.downstream_nt
        else:           w0=max(0,c-a.downstream_nt); w1=c+a.upstream_nt
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: seq=None
        rows.append(compute_splice_row(seq,a.upstream_nt,a.downstream_nt,a.n_bins,me5,score3,m3,u1,derive_axes,XF))
    F=pd.DataFrame(rows,columns=SPLICE_COLS); S=pd.concat([S.reset_index(drop=True),F],axis=1)
    for c in SPLICE_COLS: S[c]=pd.to_numeric(S[c],errors="coerce")
    S.to_parquet(f"{a.out}_sites.parquet",index=False)

    # rf-matched comparison: bin by rf, compare splice features within bins
    S["rf_bin"]=pd.cut(S["rf"],bins=np.linspace(a.rf_min,1.0,6),include_lowest=True)
    print(f"\n[RF-matched used vs silent] effect = mean(silent) - mean(used), per feature",file=sys.stderr)
    print(f"  (positive on u1_suppression_* => MORE telescripting field around SILENT sites = suppression)",file=sys.stderr)
    print(f"{'feature':<26}{'silent-used':>12}{'95% CI':>22}{'verdict':>10}",file=sys.stderr)
    eff_rows=[]
    for f in SPLICE_COLS:
        du=[]; ds=[]
        for b,gb in S.groupby("rf_bin",observed=True):
            u=gb[gb.used==1][f].dropna(); s=gb[gb.used==0][f].dropna()
            if len(u)<5 or len(s)<5: continue
            m=pd.concat([u,s]).mean()
            du.append(u-m); ds.append(s-m)
        if not du: continue
        u_all=np.concatenate([x.values for x in du]); s_all=np.concatenate([x.values for x in ds])
        diff,lo,hi=boot_diff(u_all,s_all)
        verdict="**" if (lo>0 or hi<0) else ""
        eff_rows.append(dict(feature=f,silent_minus_used=diff,ci_lo=lo,ci_hi=hi,sig=verdict))
        print(f"{f:<26}{diff:>+12.4f}   [{lo:>+.4f},{hi:>+.4f}]{verdict:>8}",file=sys.stderr)
    pd.DataFrame(eff_rows).to_csv(f"{a.out}_effects.tsv",sep="\t",index=False)

    donor_feats=[f for f in ["spl_local_me5","spl_interv_me5_max","spl_upstream_mean_u1",
                             "spl_u1_suppression_up"] if f in S.columns]
    print(f"\n[telescripting gradient] silent-used donor strength vs distance from upstream 5'ss",file=sys.stderr)
    S["dbin"]=pd.cut(S["dist_from_5ss"],bins=[0,500,1500,4000,10000,1e9],
                     labels=["0-0.5k","0.5-1.5k","1.5-4k","4-10k",">10k"])
    grad_rows=[]
    for f in donor_feats:
        line=[]
        for b,gb in S.groupby("dbin",observed=True):
            u=gb[gb.used==1][f].dropna(); s=gb[gb.used==0][f].dropna()
            if len(u)<10 or len(s)<10:
                line.append((str(b),np.nan,0)); continue
            line.append((str(b),float(s.mean()-u.mean()),len(s)+len(u)))
        cells=" ".join(f"{b}:{d:+.3f}(n={n})" if np.isfinite(d) else f"{b}:NA" for b,d,n in line)
        print(f"  {f:<24} {cells}",file=sys.stderr)
        for b,d,n in line: grad_rows.append(dict(feature=f,dist_bin=b,silent_minus_used=d,n=n))
    pd.DataFrame(grad_rows).to_csv(f"{a.out}_gradient.tsv",sep="\t",index=False)
    print("  telescripting predicts a POSITIVE effect that SHRINKS with distance "
          "(protection strongest near the 5'ss).",file=sys.stderr)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    Sd=S.dropna(subset=SPLICE_COLS)
    y=1-Sd["used"].to_numpy()
    rf_only=StandardScaler().fit_transform(Sd[["rf"]])
    spl=StandardScaler().fit_transform(Sd[SPLICE_COLS])
    import numpy as _np
    Xrf=rf_only; Xboth=_np.hstack([rf_only,spl])
    def cv_auc(X,y,k=5,seed=0):
        rng=_np.random.default_rng(seed); idx=rng.permutation(len(y)); folds=_np.array_split(idx,k); au=[]
        for i in range(k):
            te=folds[i]; tr=_np.concatenate([folds[j] for j in range(k) if j!=i])
            if len(_np.unique(y[tr]))<2 or len(_np.unique(y[te]))<2: continue
            m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(X[tr],y[tr])
            au.append(roc_auc_score(y[te],m.predict_proba(X[te])[:,1]))
        return float(_np.mean(au)) if au else float("nan")
    a_rf=cv_auc(Xrf,y); a_both=cv_auc(Xboth,y)
    print(f"\n[predict SILENCE | 5-fold AUC]  RF-only={a_rf:.3f}  RF+splice={a_both:.3f}  "
          f"delta={a_both-a_rf:+.3f}",file=sys.stderr)
    print(" If RF+splice > RF-only, the splicing/U1 context carries suppression info that",file=sys.stderr)
    print(" RF (complete-grammar score) does NOT -- i.e. the suppressor is non-grammatical.",file=sys.stderr)
    print(f"[out] {a.out}_sites.parquet, {a.out}_effects.tsv",file=sys.stderr)

if __name__=="__main__": main()