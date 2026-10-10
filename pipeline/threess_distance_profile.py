#!/usr/bin/env python3
# (self-contained: no extract_splicing_features import) strongest upstream maxentscan 3'ss
import argparse, sys, os, shutil
import numpy as np, pandas as pd

_venv = os.environ.get("VIRTUAL_ENV")
if _venv:
    _sp = os.path.join(_venv, "lib",
                       f"python{sys.version_info.major}.{sys.version_info.minor}",
                       "site-packages")
    if os.path.isdir(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

try:
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix5, load_matrix3
except Exception as e:
    sys.exit(f"[fatal] cannot import maxentpy: {type(e).__name__}: {e}\n"
             f"        VIRTUAL_ENV={_venv}\n"
             f"        sys.path[:3]={sys.path[:3]}")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x,**k): return x


class GenomeFasta:
    def __init__(self, path):
        self.path = path
        try:
            import pysam
            self.fa = pysam.FastaFile(path); self.engine = "pysam"
            self.refs = set(self.fa.references)
        except Exception:
            from pyfaidx import Fasta
            self.fa = Fasta(path, sequence_always_upper=True); self.engine = "pyfaidx"
            self.refs = set(self.fa.keys())

    def fetch(self, chrom, start, end, strand="+"):
        c = chrom if chrom in self.refs else chrom.lstrip("chr")
        if c not in self.refs: return None
        if self.engine == "pysam":
            seq = self.fa.fetch(c, start, end).upper()
        else:
            seq = str(self.fa[c][start:end]).upper()
        if strand == "-":
            seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        return seq


def init_maxentscan():
    m5 = load_matrix5(); m3 = load_matrix3()
    def score5(seq):
        if len(seq) != 9 or "N" in seq: return -99.0
        try: return float(maxent.score5(seq, matrix=m5))
        except Exception: return -99.0
    def score3(seq):
        if len(seq) != 23 or "N" in seq: return -99.0
        try: return float(maxent.score3(seq, matrix=m3))
        except Exception: return -99.0
    return score5, score3, m5, m3


def _scalar(s):
    if s is None: return None
    if np.isscalar(s): return float(s)
    try:
        arr=np.asarray(s,dtype=float)
        if arr.size==0: return None
        return float(np.nanmax(arr))
    except Exception: return None

def _call_score3(score3, win, m3):
    try: return _scalar(score3(win))
    except TypeError:
        try: return _scalar(score3(win, m3))
        except Exception: return None
    except Exception: return None

def scan_best_3ss(seq, lo, hi, score3, m3):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
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

def scan_best_5ss(seq, lo, hi, me5_lookup):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
    if hi-lo<9: return np.nan
    best=np.nan
    for i in range(lo,hi-9+1):
        if seq[i+3]!="G" or seq[i+4]!="T": continue
        k=seq[i:i+9].upper()
        if "N" in k: continue
        v=me5_lookup.get(k)
        if v is not None:
            fv=float(v)
            if np.isnan(best) or fv>best: best=fv
    return best

def edges(max_nt, step):
    e=list(range(0,int(max_nt)+1,int(step)))
    if e[-1]<max_nt: e.append(int(max_nt))
    return e

def from_shortlist(path):
    df=pd.read_parquet(path)
    if "n_internal" in df.columns: df=df[df.n_internal>0]
    df=df[~df["is_canonical"].astype(bool)].copy()
    df["cls"]="internal competitor"
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
        raw=raw[raw["ann"].astype(str).str.upper().isin(keep)]
        label="atlas "+"/".join(sorted(keep))
    else: label="all atlas"
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
        exs=sorted(exs,key=lambda e:e[1]); mid=exs[1:-1]
        for (chrom,s,e,st) in mid:
            if e-s<50: continue
            internal.append((chrom,s,e,st))
    if not internal: sys.exit("[err] no internal exons found in GTF")
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(internal),size=n)
    rows=[]
    for i in idx:
        chrom,s,e,st=internal[i]
        center = e if st=="+" else s
        rows.append(dict(chrom=str(chrom),center=int(center),strand=st,cls="random internal exon"))
    return pd.DataFrame(rows)

def from_random_genebody(genes_bed, n, max_nt, seed=0):
    g=pd.read_csv(genes_bed,sep="\t",header=None,names=["chrom","start","end","name","score","strand"])
    g=g[(g.end-g.start)>(2*max_nt+1000)].reset_index(drop=True)
    if len(g)==0: sys.exit("[err] no genes long enough for random sampling")
    rng=np.random.default_rng(seed); rows=[]
    idx=rng.integers(0,len(g),size=n)
    for i in idx:
        r=g.iloc[i]; lo=int(r.start)+max_nt; hi=int(r.end)-max_nt
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
    cum_windows=[w for w in (250,500,1000,1500,2000,3000,4000,6200) if w<=max_nt]
    if max_nt not in cum_windows: cum_windows.append(int(max_nt))
    shell_step=250
    shell_windows=[(s,s+shell_step) for s in range(0,int(max_nt),shell_step)]
    def all_acc(chrom,center,strand):
        if strand=="+": w0=max(0,center-int(max_nt)); w1=center+pad
        else:           w0=max(0,center-pad); w1=center+int(max_nt)
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: return (np.array([]),np.array([]))
        if not seq: return (np.array([]),np.array([]))
        seq=seq.upper(); n=len(seq); cand_idx=n-pad
        ra=np.frombuffer(seq.encode("ascii","replace"),dtype=np.uint8)
        ags=np.where((ra[:-1]==65)&(ra[1:]==71))[0]+2
        dists=[]; scores=[]
        for i in ags:
            p=int(i)
            if p-20<0 or p+3>n: continue
            if p>cand_idx-pad: continue
            v=_call_score3(score3, seq[p-20:p+3], m3)
            if v is not None:
                dists.append(cand_idx-p)
                scores.append(float(v))
        return (np.array(dists,float), np.array(scores,float))
    def best_in(dist,score,lo,hi):
        if dist.size==0: return np.nan
        m=(dist>=lo)&(dist<hi)
        return float(np.max(score[m])) if m.any() else np.nan
    _cache={}
    def scan_site(chrom,center,strand):
        key=(chrom,center,strand)
        if key not in _cache: _cache[key]=all_acc(chrom,center,strand)
        return _cache[key]
    real_diffs={agg:{w:[] for w in cum_windows} for agg in ("mean","max","min","pair")}
    null_diffs_bydist={w:[] for w in cum_windows}
    shell_real={agg:{w:[] for w in shell_windows} for agg in ("mean","max","min","pair")}
    shell_null={w:[] for w in shell_windows}
    rows=[]; null_rows=[]
    for gene,gg in df.groupby("gene"):
        can=gg[gg["is_canonical"]==True]; com=gg[gg["is_canonical"]==False]
        if len(can)==0 or len(com)==0: continue
        cr=can.iloc[0]
        com_list=[(str(rr["chrom"]),int(rr["center"]),rr["strand"]) for _,rr in com.iterrows()]
        cD,cS=scan_site(str(cr["chrom"]),int(cr["center"]),cr["strand"])
        comp_scans=[scan_site(ch,ce,st) for ch,ce,st in com_list]
        for w in cum_windows:
            c=best_in(cD,cS,0,w)
            comps=[best_in(dd,ss,0,w) for dd,ss in comp_scans]; comps=[x for x in comps if np.isfinite(x)]
            if np.isfinite(c) and comps:
                real_diffs["mean"][w].append(c-float(np.mean(comps)))
                real_diffs["max"][w].append(c-float(np.max(comps)))
                real_diffs["min"][w].append(c-float(np.min(comps)))
                for cv in comps: real_diffs["pair"][w].append(c-float(cv))
            if len(comps)>=2:
                i,j=rng.choice(len(comps),size=2,replace=False)
                if rng.random()<0.5: i,j=j,i
                null_diffs_bydist[w].append(comps[i]-comps[j])
        for (lo,hi) in shell_windows:
            c=best_in(cD,cS,lo,hi)
            comps=[best_in(dd,ss,lo,hi) for dd,ss in comp_scans]; comps=[x for x in comps if np.isfinite(x)]
            if np.isfinite(c) and comps:
                shell_real["mean"][(lo,hi)].append(c-float(np.mean(comps)))
                shell_real["max"][(lo,hi)].append(c-float(np.max(comps)))
                shell_real["min"][(lo,hi)].append(c-float(np.min(comps)))
                for cv in comps: shell_real["pair"][(lo,hi)].append(c-float(cv))
            if len(comps)>=2:
                i,j=rng.choice(len(comps),size=2,replace=False)
                if rng.random()<0.5: i,j=j,i
                shell_null[(lo,hi)].append(comps[i]-comps[j])
        wmax=cum_windows[-1]
        c=best_in(cD,cS,0,wmax)
        comps=[best_in(dd,ss,0,wmax) for dd,ss in comp_scans]; comps=[x for x in comps if np.isfinite(x)]
        if np.isfinite(c) and comps:
            rows.append(dict(gene=gene, canon_3ss=float(c),
                             comp_3ss=float(np.mean(comps)),
                             comp_max=float(np.max(comps)),
                             comp_min=float(np.min(comps)),
                             n_comp=len(comps)))
            if len(comps)>=2:
                i,j=rng.choice(len(comps),size=2,replace=False)
                if rng.random()<0.5: i,j=j,i
                null_rows.append(dict(null_diff=comps[i]-comps[j]))
    P=pd.DataFrame(rows)
    if len(P)==0:
        print("[paired] no paired genes found",file=sys.stderr); return
    P.to_csv(f"{out_prefix}_paired.tsv",sep="\t",index=False)
    pd.DataFrame(null_rows).to_csv(f"{out_prefix}_paired_null.tsv",sep="\t",index=False)
    from numpy import median
    def wilson(k,n,z=1.96):
        if n==0: return (np.nan,np.nan)
        phat=k/n; den=1+z*z/n
        c=(phat+z*z/(2*n))/den; h=(z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n)))/den
        return (c-h,c+h)
    def summ(diffs):
        d=np.asarray(diffs,float); d=d[np.isfinite(d)]
        if len(d)==0: return dict(median=np.nan,med_lo=np.nan,med_hi=np.nan,frac_pos=np.nan,frac_lo=np.nan,frac_hi=np.nan,n=0,p=1.0)
        k=int(np.sum(d>0)); nnz=int(np.sum(d!=0))
        frac=k/nnz if nnz>0 else np.nan; lo,hi=wilson(k,nnz)
        from math import erfc
        z=(k-0.5*nnz)/np.sqrt(0.25*nnz) if nnz>0 else 0.0
        p=0.5*erfc(z/np.sqrt(2))
        rng2=np.random.default_rng(0); B=1000
        bmed=np.median(rng2.choice(d,size=(B,len(d)),replace=True),axis=1)
        return dict(median=float(median(d)),med_lo=float(np.percentile(bmed,2.5)),med_hi=float(np.percentile(bmed,97.5)),
                    frac_pos=frac,frac_lo=lo,frac_hi=hi,n=len(d),p=p)
    sweep=[]
    print(f"\n=== WITHIN-GENE paired 3'ss by DISTANCE (n={len(P)} genes) ===",file=sys.stderr)
    print(f"  canonical vs MEAN / MAX(strongest) / MIN(weakest) competitor, per window",file=sys.stderr)
    aggs=[("pair","canon>each (per-pair)"),("mean","canon>mean"),("max","canon>MAX (strict)"),("min","canon>min")]
    for w in cum_windows:
        nu=summ(null_diffs_bydist[w])
        line=f"    0-{w:<5}"
        for agg,lab in aggs:
            r=summ(real_diffs[agg][w])
            sweep.append(dict(scheme="cumulative",kind="real",agg=agg,d0=0,d1=w,**r))
            line+=f"  {lab} {r['frac_pos']:.0%}[{r['frac_lo']:.0%}-{r['frac_hi']:.0%}]"
        sweep.append(dict(scheme="cumulative",kind="random",agg="null",d0=0,d1=w,**nu))
        line+=f"  null {nu['frac_pos']:.0%}"
        print(line,file=sys.stderr)
    for (lo,hi) in shell_windows:
        nu=summ(shell_null[(lo,hi)])
        for agg,_ in aggs:
            r=summ(shell_real[agg][(lo,hi)])
            sweep.append(dict(scheme="shell",kind="real",agg=agg,d0=lo,d1=hi,**r))
        sweep.append(dict(scheme="shell",kind="random",agg="null",d0=lo,d1=hi,**nu))
    pd.DataFrame(sweep).to_csv(f"{out_prefix}_paired_bydist.tsv",sep="\t",index=False)
    print(f"  [out] {out_prefix}_paired.tsv, _paired_null.tsv, _paired_bydist.tsv",file=sys.stderr)


def load_me5_lookup(path):
    arr=np.load(path, allow_pickle=True)
    return arr.item() if arr.dtype==object else dict(arr)

def paired_intervening_5ss(fa, shortlist_path, score3, m3, me5_lookup, max_nt, pad, out_prefix, seed=0):
    df=pd.read_parquet(shortlist_path)
    if "gene" not in df.columns:
        print("[interv5] no gene column; skipping",file=sys.stderr); return
    rng=np.random.default_rng(seed)
    cum_windows=[w for w in (500,1000,1500,2000,3000,4000,6200) if w<=max_nt]
    if max_nt not in cum_windows: cum_windows.append(int(max_nt))
    def interv_5ss(chrom,center,strand,win):
        if strand=="+": w0=max(0,center-win); w1=center+pad
        else:           w0=max(0,center-pad); w1=center+win
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: return np.nan
        if not seq: return np.nan
        seq=seq.upper(); L=len(seq); site=L-pad
        lo3=max(0,site-win); hi3=site-pad
        if hi3-lo3<23: return np.nan
        region=seq[lo3:hi3]
        ra=np.frombuffer(region.encode("ascii","replace"),dtype=np.uint8)
        ag=np.where((ra[:-1]==65)&(ra[1:]==71))[0]+2
        best3=None; best3v=-1e9
        for i in ag:
            p=lo3+int(i); s0=p-20
            if s0<0 or p+3>L: continue
            try:
                v=score3(seq[s0:p+3]); v=float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
            except Exception: continue
            if v>best3v: best3v=v; best3=p
        if best3 is None: return np.nan
        return scan_best_5ss(seq, best3, site-pad, me5_lookup)
    real={w:[] for w in cum_windows}; null={w:[] for w in cum_windows}
    for gene,gg in df.groupby("gene"):
        can=gg[gg["is_canonical"]==True]; com=gg[gg["is_canonical"]==False]
        if len(can)==0 or len(com)==0: continue
        cr=can.iloc[0]
        com_list=[(str(rr["chrom"]),int(rr["center"]),rr["strand"]) for _,rr in com.iterrows()]
        for w in cum_windows:
            c=interv_5ss(str(cr["chrom"]),int(cr["center"]),cr["strand"],w)
            comps=[interv_5ss(ch,ce,st,w) for ch,ce,st in com_list]; comps=[x for x in comps if np.isfinite(x)]
            if np.isfinite(c) and comps:
                real[w].append(float(np.mean(comps))-c)
            if len(comps)>=2:
                i,j=rng.choice(len(comps),size=2,replace=False)
                if rng.random()<0.5: i,j=j,i
                null[w].append(comps[i]-comps[j])
    from numpy import median
    def wilson(k,n,z=1.96):
        if n==0: return (np.nan,np.nan)
        phat=k/n; den=1+z*z/n; c=(phat+z*z/(2*n))/den
        h=(z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n)))/den; return (c-h,c+h)
    def summ(d):
        d=np.asarray(d,float); d=d[np.isfinite(d)]
        if len(d)==0: return dict(median=np.nan,med_lo=np.nan,med_hi=np.nan,frac_pos=np.nan,frac_lo=np.nan,frac_hi=np.nan,n=0,p=1.0)
        k=int(np.sum(d>0)); nnz=int(np.sum(d!=0)); frac=k/nnz if nnz>0 else np.nan; lo,hi=wilson(k,nnz)
        from math import erfc; z=(k-0.5*nnz)/np.sqrt(0.25*nnz) if nnz>0 else 0.0; p=0.5*erfc(z/np.sqrt(2))
        rng2=np.random.default_rng(0); bmed=np.median(rng2.choice(d,size=(1000,len(d)),replace=True),axis=1)
        return dict(median=float(median(d)),med_lo=float(np.percentile(bmed,2.5)),med_hi=float(np.percentile(bmed,97.5)),
                    frac_pos=frac,frac_lo=lo,frac_hi=hi,n=len(d),p=p)
    rows=[]
    print("\n=== WITHIN-GENE paired INTERVENING 5ss (terminal-exon integrity) ===",file=sys.stderr)
    print("  predict CANONICAL < COMPETITOR; frac_pos = %% genes competitor has STRONGER intervening 5ss",file=sys.stderr)
    for w in cum_windows:
        r=summ(real[w]); nu=summ(null[w])
        rows.append(dict(scheme="cumulative",kind="real",agg="mean",d0=0,d1=w,**r))
        rows.append(dict(scheme="cumulative",kind="random",agg="null",d0=0,d1=w,**nu))
        print("    0-%-5d comp>canon %.0f%% [%.0f-%.0f] med=%+.2f  null %.0f%%  (n=%d)" % (
            w, r["frac_pos"]*100, r["frac_lo"]*100, r["frac_hi"]*100, r["median"], nu["frac_pos"]*100, r["n"]),file=sys.stderr)
    pd.DataFrame(rows).to_csv(out_prefix+"_interv5ss_paired_bydist.tsv",sep="\t",index=False)
    print("  [out] "+out_prefix+"_interv5ss_paired_bydist.tsv",file=sys.stderr)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--me5_lookup",default=None,help="for intervening 5ss paired test")
    ap.add_argument("--shortlist",default=None)
    ap.add_argument("--cpa_bed",default=None)
    ap.add_argument("--atlas_bed",default=None)
    ap.add_argument("--atlas_ann",default=None)
    ap.add_argument("--atlas_ann_split",default=None)
    ap.add_argument("--genes_bed",default=None)
    ap.add_argument("--random_n",type=int,default=0)
    ap.add_argument("--internal_exon_gtf",default=None)
    ap.add_argument("--internal_exon_n",type=int,default=0)
    ap.add_argument("--max_nt",type=float,required=True)
    ap.add_argument("--step",type=int,default=250)
    ap.add_argument("--pad",type=int,default=10)
    ap.add_argument("--paired",action="store_true")
    ap.add_argument("--scheme",default="both",choices=["both","cumulative","shells"])
    ap.add_argument("--max_peaks",type=int,default=0,help="0=all; else subsample per class")
    ap.add_argument("--out",default="threess_profile")
    a=ap.parse_args()
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
    acc=defaultdict(list); per_peak=[]
    n_seq_ok=0; n_found_any=0; NEARW=500
    for _,r in tqdm(df.iterrows(),total=len(df),desc="3ss"):
        c=int(r["center"]); strand=r["strand"]; chrom=str(r["chrom"]); cls=r["cls"]
        tpm=r["tpm"] if "tpm" in r else np.nan
        if strand=="+": w0=max(0,c-maxw); w1=c+pad
        else:           w0=max(0,c-pad);  w1=c+maxw
        try: seq=fa.fetch(chrom,w0,w1,strand=strand)
        except Exception: seq=None
        if seq is None or len(seq)<50: continue
        n_seq_ok+=1; L=len(seq); cand_idx=L-pad; found_here=False
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
    print(f"[scan] {n_seq_ok:,} sequences fetched; {n_found_any:,} had >=1 acceptor found",file=sys.stderr)
    if n_found_any==0: sys.exit("[err] still all-NaN -> geometry or score3 still off")
    rows=[]
    for (cls,scheme,x),vals in acc.items():
        v=np.array(vals,float); v=v[np.isfinite(v)]
        if len(v)==0:
            rows.append(dict(cls=cls,scheme=scheme,x_nt=x,mean_score3=np.nan,se=np.nan,median_score3=np.nan,n=0)); continue
        rows.append(dict(cls=cls,scheme=scheme,x_nt=x,mean_score3=float(v.mean()),
            se=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.0,
            median_score3=float(np.median(v)),n=int(len(v))))
    R=pd.DataFrame(rows).sort_values(["scheme","cls","x_nt"])
    R.to_csv(f"{a.out}_long.tsv",sep="\t",index=False)
    print(f"[out] {a.out}_long.tsv",file=sys.stderr)
    if per_peak:
        PP=pd.DataFrame(per_peak); PP.to_csv(f"{a.out}_tpm_vs_3ss.tsv",sep="\t",index=False)
        try:
            from scipy.stats import spearmanr, pearsonr
            print(f"\n=== usage (tpm) vs near-window strongest 3'ss ===",file=sys.stderr)
            for cls,g in PP.groupby("cls"):
                g=g[np.isfinite(g.tpm)&np.isfinite(g.best_3ss_near)]
                if len(g)<20: continue
                lt=np.log10(g.tpm.clip(lower=1e-6))
                rho,p_s=spearmanr(g.tpm,g.best_3ss_near); rp,p_p=pearsonr(lt,g.best_3ss_near)
                print(f"  {cls:<18} n={len(g):<6} Spearman rho={rho:+.3f} (p={p_s:.1e})  Pearson(log tpm)={rp:+.3f}",file=sys.stderr)
        except Exception: pass
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        matplotlib.rcParams["pdf.fonttype"]=42
        PAL={"all CPA":"#C65B3C","constitutive CPA":"#8C3B22","internal competitor":"#5B7C6F",
             "random (gene body)":"#999999","random internal exon":"#C9A14A"}
        schemes=sorted(R.scheme.unique())
        fig,axes=plt.subplots(1,len(schemes),figsize=(6.4*len(schemes),4.8),squeeze=False)
        for si,scheme in enumerate(schemes):
            ax=axes[0][si]; sub=R[R.scheme==scheme]
            for cls in sub.cls.unique():
                s2=sub[sub.cls==cls].sort_values("x_nt"); col=PAL.get(cls,"#888888")
                is_null = cls in ("random (gene body)","random internal exon")
                ls="--" if is_null else "-"; mk="" if is_null else "o"
                ax.plot(s2.x_nt,s2.mean_score3,ls+mk,ms=3,color=col,label=cls,linewidth=1.6 if is_null else 1.2)
                ax.fill_between(s2.x_nt,s2.mean_score3-s2.se,s2.mean_score3+s2.se,color=col,alpha=0.15,linewidth=0)
            ax.set_title(scheme,fontsize=11); ax.set_xlabel("upstream distance (nt)"); ax.set_ylabel("strongest 3'ss (MaxEntScan)")
            ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,fontsize=8,loc="best")
        fig.suptitle("Strongest upstream 3' splice site vs distance, by peak class",fontsize=12,y=1.02)
        fig.tight_layout()
        tmp_png="/tmp/_threess_profile.png"; fig.savefig(tmp_png,dpi=200,bbox_inches="tight"); shutil.copy(tmp_png,f"{a.out}.png")
        try:
            tmp_pdf="/tmp/_threess_profile.pdf"; fig.savefig(tmp_pdf,bbox_inches="tight"); shutil.copy(tmp_pdf,f"{a.out}.pdf")
        except Exception as e:
            print(f"[warn] PDF write failed ({e})",file=sys.stderr)
        print(f"[out] {a.out}.png",file=sys.stderr)
    except Exception as e:
        print(f"[warn] figure failed ({e}); data in {a.out}_long.tsv",file=sys.stderr)
    if a.paired:
        if not a.shortlist:
            print("[paired] needs --shortlist; skipping",file=sys.stderr)
        else:
            paired_within_gene_3ss(fa, a.shortlist, score3, m3, int(a.max_nt), a.pad, a.out, seed=0)
            if a.me5_lookup:
                me5_lu=load_me5_lookup(a.me5_lookup)
                paired_intervening_5ss(fa, a.shortlist, score3, m3, me5_lu, int(a.max_nt), a.pad, a.out, seed=0)

if __name__=="__main__": main()


# print(f"\n within-gene paired 3'ss: canonical vs competitor (n={len(p)} genes)

# print(f"\n usage (tpm) vs near-window strongest 3'ss ",file=sys.stderr)

