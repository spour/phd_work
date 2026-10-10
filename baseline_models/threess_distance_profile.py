#!/usr/bin/env python3
# (multi-source, geometry-fixed) strongest upstream maxentscan 3'ss
import argparse, sys, os, shutil
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x

def _scalar(s):
    if s is None: return None
    if np.isscalar(s):
        return float(s)
    try:
        arr=np.asarray(s,dtype=float)
        if arr.size==0: return None
        return float(np.nanmax(arr))
    except Exception:
        return None

def _call_score3(score3, win, m3):
    try:
        return _scalar(score3(win))
    except TypeError:
        try: return _scalar(score3(win, m3))
        except Exception: return None
    except Exception:
        return None

def scan_best_3ss(seq, lo, hi, score3, m3):
    n=len(seq)
    lo=max(0,lo); hi=min(n,hi)
    if hi-lo<23: return np.nan
    region=seq[lo:hi].upper()
    ra=np.frombuffer(region.encode("ascii","replace"),dtype=np.uint8)
    ag=np.where((ra[:-1]==65)&(ra[1:]==71))[0]+2
    best=-1e9; found=False
    for i in ag:
        p=lo+int(i); s0=p-20
        if s0<0 or p+3>n: continue
        v=_call_score3(score3, seq[s0:p+3].upper(), m3)
        if v is not None and v>best: best=v; found=True
    return float(best) if found else np.nan

def edges(max_nt, step):
    e=list(range(0,int(max_nt)+1,int(step)))
    if e[-1]<max_nt: e.append(int(max_nt))
    return e

# source loaders: each returns dataframe[gene?,chrom,center,strand,cls]
def from_shortlist(path):
    df=pd.read_parquet(path)
    if "n_internal" in df.columns: df=df[df.n_internal>0]
    df=df.copy()
    df["cls"]=np.where(df["is_canonical"].astype(bool),"contested CPA","internal competitor")
    return df[["chrom","center","strand","cls"]]

def from_cpa_bed(path):
    cols=["chrom","start","end","symbol","enst","strand"]
    raw=pd.read_csv(path,sep="\t",header=None)
    extra=raw.shape[1]-6
    names=cols+[f"x{i}" for i in range(extra)] if extra>0 else cols
    raw.columns=names
    raw["center"]=((raw["start"].astype(int)+raw["end"].astype(int))//2)
    out=[]
    a=raw.copy(); a["cls"]="all CPA"; out.append(a[["chrom","center","strand","cls"]])
    const_col=[c for c in raw.columns if "constitut" in str(c).lower()]
    if const_col:
        cc=const_col[0]; sub=raw[raw[cc].astype(str).str.lower().isin(["1","true","yes"])].copy()
        if len(sub): sub["cls"]="constitutive CPA"; out.append(sub[["chrom","center","strand","cls"]])
    return pd.concat(out,ignore_index=True)

def from_atlas_bed(path, ann_keep=None):
    cols=["chrom","start","end","cid","tpm","strand","repsite","supp","nprot","ann","pas"]
    raw=pd.read_csv(path,sep="\t",header=None,names=cols,dtype=str)
    pos1=raw["cid"].astype(str).str.split(":").str[1]
    raw["center"]=pd.to_numeric(pos1,errors="coerce")-1
    raw=raw.dropna(subset=["center"]); raw["center"]=raw["center"].astype(int)
    raw["chrom"]=raw["chrom"].where(raw["chrom"].str.startswith("chr"),"chr"+raw["chrom"])
    raw["tpm"]=pd.to_numeric(raw["tpm"],errors="coerce")
    if ann_keep:
        keep=set(x.strip().upper() for x in ann_keep.split(","))
        before=len(raw)
        raw=raw[raw["ann"].astype(str).str.upper().isin(keep)]
        print(f"[atlas] ann filter {sorted(keep)}: {before:,} -> {len(raw):,} sites",file=sys.stderr)
        label="atlas "+"/".join(sorted(keep))
    else:
        label="all atlas"
    raw["cls"]=label
    return raw[["chrom","center","strand","cls","tpm"]]

def from_internal_exons(gtf, n, max_nt, seed=0):
    import re as _re
    tx={}
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"): continue
            f=line.rstrip("\n").split("\t")
            if len(f)<9 or f[2]!="exon": continue
            if 'transcript_type "protein_coding"' not in f[8] and 'transcript_biotype "protein_coding"' not in f[8]:
                continue
            m=_re.search(r'transcript_id "([^"]+)"', f[8])
            if not m: continue
            tx.setdefault(m.group(1),[]).append((f[0],int(f[3]),int(f[4]),f[6]))
    internal=[]
    for t,exs in tx.items():
        if len(exs)<3: continue
        exs=sorted(exs,key=lambda e:e[1])
        strand=exs[0][3]
        mid = exs[1:-1]
        for (chrom,s,e,st) in mid:
            if e-s < 50: continue
            internal.append((chrom,s,e,st))
    if not internal: sys.exit("[err] no internal exons found in GTF")
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(internal),size=n)
    rows=[]
    for i in idx:
        chrom,s,e,st=internal[i]
        center = e if st=="+" else s
        rows.append(dict(chrom=str(chrom),center=int(center),strand=st,cls="random internal exon"))
    return pd.DataFrame(rows)

def from_random_genebody(genes_bed, n, max_nt, seed=0):
    g=pd.read_csv(genes_bed,sep="\t",header=None,
                  names=["chrom","start","end","name","score","strand"])
    g=g[(g.end-g.start)>(2*max_nt+1000)].reset_index(drop=True)
    if len(g)==0: sys.exit("[err] no genes long enough for random sampling")
    rng=np.random.default_rng(seed)
    rows=[]
    idx=rng.integers(0,len(g),size=n)
    for i in idx:
        r=g.iloc[i]
        lo=int(r.start)+max_nt; hi=int(r.end)-max_nt
        if hi<=lo: continue
        pos=int(rng.integers(lo,hi))
        rows.append(dict(chrom=str(r.chrom),center=pos,strand=r.strand,cls="random (gene body)"))
    return pd.DataFrame(rows)

def paired_within_gene_3ss(fa, shortlist_path, score3, m3, max_nt, pad, out_prefix, seed=0):
    df=pd.read_parquet(shortlist_path)
    if "gene" not in df.columns:
        print("[paired] shortlist has no 'gene' column; cannot pair. Skipping.",file=sys.stderr)
        return
    rng=np.random.default_rng(seed)
    def acc_at(chrom,center,strand):
        if strand=="+": w0=max(0,center-max_nt); w1=center+pad
        else:           w0=max(0,center-pad);     w1=center+max_nt
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: return np.nan
        if not seq: return np.nan
        L=len(seq); cand_idx=L-pad
        return scan_best_3ss(seq, max(0,cand_idx-max_nt), cand_idx-pad, score3, m3)
    rows=[]; null_diffs=[]
    for gene,gg in df.groupby("gene"):
        can=gg[gg["is_canonical"]==True]; com=gg[gg["is_canonical"]==False]
        if len(can)==0 or len(com)==0: continue
        cr=can.iloc[0]
        c=acc_at(str(cr["chrom"]),int(cr["center"]),cr["strand"])
        comps=[acc_at(str(rr["chrom"]),int(rr["center"]),rr["strand"]) for _,rr in com.iterrows()]
        comps=[x for x in comps if np.isfinite(x)]
        if not np.isfinite(c) or not comps: continue
        rows.append(dict(gene=gene, canon_3ss=float(c), comp_3ss=float(np.mean(comps))))
        if len(comps)>=2:
            i,j=rng.choice(len(comps),size=2,replace=False)
            if rng.random()<0.5: i,j=j,i
            null_diffs.append(comps[i]-comps[j])
    P=pd.DataFrame(rows)
    if len(P)==0:
        print("[paired] no paired genes found",file=sys.stderr); return
    P.to_csv(f"{out_prefix}_paired.tsv",sep="\t",index=False)
    pd.DataFrame({"null_diff":null_diffs}).to_csv(f"{out_prefix}_paired_null.tsv",sep="\t",index=False)
    from numpy import median
    def wilson(k,n,z=1.96):
        if n==0: return (np.nan,np.nan)
        phat=k/n; den=1+z*z/n
        c=(phat+z*z/(2*n))/den; h=(z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n)))/den
        return (c-h,c+h)
    def report(d, label):
        d=np.asarray(d,float); d=d[np.isfinite(d)]
        k=int(np.sum(d>0)); nnz=int(np.sum(d!=0))
        frac=k/nnz if nnz>0 else np.nan
        lo,hi=wilson(k,nnz)
        from math import erfc
        z=(k-0.5*nnz)/np.sqrt(0.25*nnz) if nnz>0 else 0.0
        p=0.5*erfc(z/np.sqrt(2))
        return frac,lo,hi,float(median(d)),p
    d_real=(P["canon_3ss"]-P["comp_3ss"]).to_numpy()
    fr,lo,hi,med,p = report(d_real,"real")
    nf,nlo,nhi,nmed,npv = report(null_diffs,"null")
    print(f"\n=== WITHIN-GENE paired 3'ss: canonical vs competitor (n={len(P)} genes) ===",file=sys.stderr)
    print(f"  acceptor hypothesis predicts CANONICAL > COMPETITOR (stronger defining upstream acceptor)",file=sys.stderr)
    print(f"  REAL  median(canon-comp)={med:+.3f}  canon>comp {fr:.0%} [{lo:.0%}-{hi:.0%}]  p={p:.1e}",file=sys.stderr)
    print(f"  NULL  (random competitor pair)        {nf:.0%} [{nlo:.0%}-{nhi:.0%}]  (should be ~50%)",file=sys.stderr)
    print(f"  {'supports' if med>0 and fr>0.5 else 'does NOT support'} the acceptor hypothesis",file=sys.stderr)
    print(f"  [out] {out_prefix}_paired.tsv, {out_prefix}_paired_null.tsv",file=sys.stderr)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scripts_dir",required=True)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--shortlist",default=None)
    ap.add_argument("--cpa_bed",default=None)
    ap.add_argument("--atlas_bed",default=None)
    ap.add_argument("--atlas_ann",default=None,
                    help="comma-sep atlas annotation classes to keep as ONE curve "
                         "(e.g. 'TE' for terminal exon, 'IN' for intronic). Omit = all classes.")
    ap.add_argument("--atlas_ann_split",default=None,
                    help="comma-sep atlas annotation classes, each as a SEPARATE curve "
                         "(e.g. 'TE,IN,EX' to compare terminal vs internal vs other exonic).")
    ap.add_argument("--genes_bed",default=None,
                    help="BED6 of pc gene bodies for the random null (e.g. pc_genes.bed)")
    ap.add_argument("--random_n",type=int,default=0,
                    help="if >0 and --genes_bed given, add a 'random (gene body)' null class")
    ap.add_argument("--internal_exon_gtf",default=None,
                    help="GTF for the 'random internal exon' hard null (samples inside "
                         "annotated internal exons; tests terminal-specificity)")
    ap.add_argument("--internal_exon_n",type=int,default=0,
                    help="if >0 with --internal_exon_gtf, add the internal-exon null class")
    ap.add_argument("--max_nt",type=float,required=True)
    ap.add_argument("--step",type=int,default=250)
    ap.add_argument("--pad",type=int,default=10)
    ap.add_argument("--paired",action="store_true",
                    help="ALSO run within-gene paired canonical-vs-competitor 3'ss test "
                         "(cancels gene-position confound)")
    ap.add_argument("--scheme",default="both",choices=["both","cumulative","shells"])
    ap.add_argument("--max_peaks",type=int,default=0,help="0=all; else subsample per class")
    ap.add_argument("--out",default="threess_profile")
    a=ap.parse_args()
    sys.path.insert(0,a.scripts_dir)
    from extract_splicing_features import GenomeFasta, init_maxentscan
    _,score3,_,m3=init_maxentscan()
    fa=GenomeFasta(a.reference_fasta)

    parts=[]
    if a.shortlist: parts.append(from_shortlist(a.shortlist))
    if a.cpa_bed:   parts.append(from_cpa_bed(a.cpa_bed))
    if a.atlas_bed:
        if a.atlas_ann_split:
            for ann in [x.strip() for x in a.atlas_ann_split.split(",")]:
                parts.append(from_atlas_bed(a.atlas_bed, ann_keep=ann))
        else:
            parts.append(from_atlas_bed(a.atlas_bed, ann_keep=a.atlas_ann))
    if a.random_n>0:
        if not a.genes_bed: sys.exit("[err] --random_n requires --genes_bed")
        parts.append(from_random_genebody(a.genes_bed,a.random_n,int(a.max_nt)))
    if a.internal_exon_n>0:
        if not a.internal_exon_gtf: sys.exit("[err] --internal_exon_n requires --internal_exon_gtf")
        parts.append(from_internal_exons(a.internal_exon_gtf,a.internal_exon_n,int(a.max_nt)))
    if not parts: sys.exit("[err] pass at least one of --shortlist/--cpa_bed/--atlas_bed")
    for i,p in enumerate(parts):
        if "tpm" not in p.columns: parts[i]=p.assign(tpm=np.nan)
    df=pd.concat(parts,ignore_index=True).dropna(subset=["center"])
    df["center"]=df["center"].astype(int)
    if a.max_peaks>0:
        df=pd.concat([s.sample(min(len(s),a.max_peaks),random_state=0)
                      for _,s in df.groupby("cls")],ignore_index=True)
    print("[data] peaks per class:",file=sys.stderr)
    for cls,n in df.cls.value_counts().items(): print(f"   {cls:<20} {n:,}",file=sys.stderr)
    print(f"  max_nt={a.max_nt:.0f} step={a.step} pad={a.pad}",file=sys.stderr)

    ED=edges(a.max_nt,a.step); grid=ED[1:]; pad=a.pad; maxw=int(a.max_nt)
    from collections import defaultdict
    acc=defaultdict(list)
    per_peak=[]

    n_seq_ok=0; n_found_any=0
    NEARW=500
    for _,r in tqdm(df.iterrows(),total=len(df),desc="3ss"):
        c=int(r["center"]); strand=r["strand"]; chrom=str(r["chrom"]); cls=r["cls"]
        tpm=r["tpm"] if "tpm" in r else np.nan
        if strand=="+": w0=max(0,c-maxw); w1=c+pad
        else:           w0=max(0,c-pad);  w1=c+maxw
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: seq=None
        if seq is None or len(seq)<50: continue
        n_seq_ok+=1
        L=len(seq)
        cand_idx=L-pad
        found_here=False
        near=scan_best_3ss(seq,cand_idx-NEARW,cand_idx-pad,score3,m3)
        if np.isfinite(tpm) and np.isfinite(near):
            per_peak.append(dict(cls=cls,tpm=float(tpm),best_3ss_near=float(near)))
        if a.scheme in ("both","cumulative"):
            for W in grid:
                v=scan_best_3ss(seq,cand_idx-W,cand_idx-pad,score3,m3)
                acc[(cls,"cumulative",W)].append(v)
                if np.isfinite(v): found_here=True
        if a.scheme in ("both","shells"):
            for j in range(len(ED)-1):
                d_lo=ED[j]; d_hi=ED[j+1]
                v=scan_best_3ss(seq,cand_idx-d_hi,cand_idx-max(d_lo,pad),score3,m3)
                acc[(cls,"shells",d_hi)].append(v)
                if np.isfinite(v): found_here=True
        if found_here: n_found_any+=1
    print(f"[scan] {n_seq_ok:,} sequences fetched; {n_found_any:,} had >=1 acceptor found",
          file=sys.stderr)
    if n_found_any==0:
        sys.exit("[err] still all-NaN -> run diag_3ss.py; geometry or score3 still off")

    rows=[]
    for (cls,scheme,x),vals in acc.items():
        v=np.array(vals,float); v=v[np.isfinite(v)]
        if len(v)==0:
            rows.append(dict(cls=cls,scheme=scheme,x_nt=x,mean_score3=np.nan,se=np.nan,
                             median_score3=np.nan,n=0)); continue
        rows.append(dict(cls=cls,scheme=scheme,x_nt=x,mean_score3=float(v.mean()),
            se=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.0,
            median_score3=float(np.median(v)),n=int(len(v))))
    R=pd.DataFrame(rows).sort_values(["scheme","cls","x_nt"])
    R.to_csv(f"{a.out}_long.tsv",sep="\t",index=False)
    print(f"[out] {a.out}_long.tsv",file=sys.stderr)

    # tpm vs near-window 3'ss strength correlation (atlas classes have tpm)
    if per_peak:
        PP=pd.DataFrame(per_peak)
        PP.to_csv(f"{a.out}_tpm_vs_3ss.tsv",sep="\t",index=False)
        from scipy.stats import spearmanr, pearsonr
        print(f"\n=== usage (tpm) vs near-window strongest 3'ss ===",file=sys.stderr)
        for cls,g in PP.groupby("cls"):
            g=g[np.isfinite(g.tpm)&np.isfinite(g.best_3ss_near)]
            if len(g)<20: continue
            lt=np.log10(g.tpm.clip(lower=1e-6))
            rho,p_s=spearmanr(g.tpm,g.best_3ss_near)
            rp,p_p=pearsonr(lt,g.best_3ss_near)
            print(f"  {cls:<18} n={len(g):<6} Spearman rho={rho:+.3f} (p={p_s:.1e})  "
                  f"Pearson(log tpm)={rp:+.3f}",file=sys.stderr)
        print(f"[out] {a.out}_tpm_vs_3ss.tsv  (cls, tpm, best_3ss_near)",file=sys.stderr)
    print(f"\n=== mean strongest 3'ss by upstream distance ===",file=sys.stderr)
    for scheme in sorted(R.scheme.unique()):
        print(f"  [{scheme}]",file=sys.stderr)
        for cls in sorted(R[R.scheme==scheme].cls.unique()):
            s2=R[(R.scheme==scheme)&(R.cls==cls)].sort_values("x_nt")
            head=", ".join(f"{int(x)}:{m:.2f}" for x,m in zip(s2.x_nt.head(6),s2.mean_score3.head(6)))
            print(f"    {cls:<20} {head} ...",file=sys.stderr)

    # figure (hardened: png to /tmp then copy
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["pdf.fonttype"]=42
        PAL={"all CPA":"#C65B3C","constitutive CPA":"#8C3B22","contested CPA":"#E08A6F",
             "all atlas":"#3E6B8B","internal competitor":"#5B7C6F","random (gene body)":"#999999",
             "atlas TE":"#3E6B8B","atlas IN":"#7BA3C0","atlas EX":"#B0C4D4",
             "random internal exon":"#C9A14A"}
        schemes=sorted(R.scheme.unique())
        fig,axes=plt.subplots(1,len(schemes),figsize=(6.4*len(schemes),4.8),squeeze=False)
        for si,scheme in enumerate(schemes):
            ax=axes[0][si]; sub=R[R.scheme==scheme]
            for cls in sub.cls.unique():
                s2=sub[sub.cls==cls].sort_values("x_nt")
                col=PAL.get(cls,"#888888")
                is_null = cls in ("random (gene body)","random internal exon")
                ls="--" if is_null else "-"
                mk="" if is_null else "o"
                ax.plot(s2.x_nt,s2.mean_score3,ls+mk,ms=3,color=col,label=cls,
                        linewidth=1.6 if is_null else 1.2)
                ax.fill_between(s2.x_nt,s2.mean_score3-s2.se,s2.mean_score3+s2.se,
                                color=col,alpha=0.15,linewidth=0)
            ax.set_title(scheme,fontsize=11)
            ax.set_xlabel("upstream distance (nt)"); ax.set_ylabel("strongest 3'ss (MaxEntScan)")
            ax.spines[["top","right"]].set_visible(False)
            ax.legend(frameon=False,fontsize=8,loc="best")
        fig.suptitle("Strongest upstream 3' splice site vs distance, by peak class",fontsize=12,y=1.02)
        fig.tight_layout()
        tmp_png="/tmp/_threess_profile.png"; fig.savefig(tmp_png,dpi=200,bbox_inches="tight")
        shutil.copy(tmp_png,f"{a.out}.png")
        try:
            tmp_pdf="/tmp/_threess_profile.pdf"; fig.savefig(tmp_pdf,bbox_inches="tight")
            shutil.copy(tmp_pdf,f"{a.out}.pdf")
        except Exception as e:
            print(f"[warn] PDF write failed ({e}); PNG written, replot from {a.out}_long.tsv",file=sys.stderr)
        print(f"[out] {a.out}.png",file=sys.stderr)
    except Exception as e:
        print(f"[warn] figure failed ({e}); data is in {a.out}_long.tsv, replot locally",file=sys.stderr)

    if a.paired:
        if not a.shortlist:
            print("[paired] needs --shortlist; skipping",file=sys.stderr)
        else:
            paired_within_gene_3ss(fa, a.shortlist, score3, m3, int(a.max_nt), a.pad, a.out, seed=0)

if __name__=="__main__": main()