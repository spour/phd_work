#!/usr/bin/env python3
# mirror of threess_distance_profile.py, but for the donor side
import argparse, sys, os
import numpy as np, pandas as pd

# downstream donor scanners
def scan_me5(seq, lo, hi, score5):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
    if hi-lo<9: return np.array([])
    out=[]
    for i in range(lo, hi-9+1):
        w=seq[i:i+9].upper()
        if "N" in w or w[3:5]!="GT": out.append(np.nan); continue
        try: out.append(float(score5(w)))
        except Exception: out.append(np.nan)
    return np.array(out,float)

def scan_u1(seq, lo, hi, u1_lookup):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
    if hi-lo<9: return np.array([])
    out=[]
    for i in range(lo, hi-9+1):
        k=seq[i:i+9].upper()
        if "N" in k: out.append(np.nan); continue
        v=u1_lookup.get(k)
        out.append(-float(v) if v is not None else np.nan)
    return np.array(out,float)

def load_u1_lookup(path):
    arr=np.load(path, allow_pickle=True)
    return arr.item() if arr.dtype==object else dict(arr)

# per-class source builders (centers in transcribed orientation)
def from_shortlist(path):
    df=pd.read_parquet(path)
    out=[]
    for _,r in df.iterrows():
        cls = "contested CPA" if r.get("is_canonical",False) else "internal competitor"
        out.append(dict(chrom=str(r["chrom"]),center=int(r["center"]),strand=r["strand"],cls=cls))
    return pd.DataFrame(out)

def from_cpa_bed(path):
    rows=[]
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"): continue
            f=line.rstrip("\n").split("\t")
            chrom,s,e=f[0],int(f[1]),int(f[2]); strand=f[5] if len(f)>5 else "+"
            rows.append(dict(chrom=str(chrom),center=(s+e)//2,strand=strand,cls="all CPA"))
    return pd.DataFrame(rows)

def from_atlas(path, ann_keep=None):
    rows=[]
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"): continue
            f=line.rstrip("\n").split("\t")
            chrom,s,e=f[0],int(f[1]),int(f[2]); strand=f[5] if len(f)>5 else "+"
            cid=f[3] if len(f)>3 else ""
            ann=f[9] if len(f)>9 else ""
            try:
                parts=cid.split(":"); mid=int(parts[1]) if len(parts)>1 else (s+e)//2
                center=mid-1
            except Exception: center=(s+e)//2
            cls=f"atlas {ann}" if ann else "atlas"
            if ann_keep and ann not in ann_keep: continue
            rows.append(dict(chrom=str(chrom),center=int(center),strand=strand,cls=cls))
    return pd.DataFrame(rows)

def from_internal_exons(gtf, n, seed=0):
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
    if not internal: sys.exit("[err] no internal exons in GTF")
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(internal),size=n)
    rows=[]
    for i in idx:
        chrom,s,e,st=internal[i]
        center = e if st=="+" else s
        rows.append(dict(chrom=str(chrom),center=int(center),strand=st,cls="random internal exon"))
    return pd.DataFrame(rows)

def from_random_genebody(genes_bed, n, seed=0):
    g=[]
    with open(genes_bed) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"): continue
            f=line.rstrip("\n").split("\t")
            g.append((f[0],int(f[1]),int(f[2]),f[5] if len(f)>5 else "+"))
    rng=np.random.default_rng(seed); rows=[]
    for _ in range(n):
        chrom,s,e,st=g[rng.integers(0,len(g))]
        if e-s<200: continue
        center=int(rng.integers(s+50,e-50))
        rows.append(dict(chrom=str(chrom),center=center,strand=st,cls="random (gene body)"))
    return pd.DataFrame(rows)

def paired_within_gene(fa, shortlist_path, metric, score5, u1, win, pad, thr, out_prefix, seed=0):
    df=pd.read_parquet(shortlist_path)
    if "gene" not in df.columns:
        print("[paired] shortlist has no 'gene' column; cannot pair. Skipping.",file=sys.stderr)
        return
    rng=np.random.default_rng(seed)
    def sig_arr(chrom,center,strand):
        if strand=="+": gstart,gend=center-win-pad, center+pad
        else:           gstart,gend=center-pad, center+win+pad
        try: seq=fa.fetch(chrom,gstart,gend,strand=strand)
        except Exception: return (None,None)
        site_idx = win+pad if strand=="+" else win+pad-1
        if metric=="me5": arr=scan_me5(seq, max(0,site_idx-win), site_idx, score5)
        else:             arr=scan_u1 (seq, max(0,site_idx-win), site_idx, u1)
        if arr.size==0: return (None,None)
        dist = site_idx - np.arange(arr.size)
        ok=np.isfinite(arr)
        return (arr[ok], dist[ok])
    def load_in(arr,dist,d0,d1):
        if arr is None: return np.nan
        m=(dist>=d0)&(dist<d1)&(arr>=thr)
        return float(np.sum(arr[m]))/max(d1-d0,1)*100.0
    def frac_pos_in(arr,dist,d0,d1):
        if arr is None: return np.nan
        m=(dist>=d0)&(dist<d1)
        if not m.any(): return np.nan
        return float(np.sum(arr[m]>=thr))/max(d1-d0,1)*100.0

    step=250
    cum_windows=[(0,250),(0,500),(0,1000),(0,1500),(0,2000)]
    cum_windows=[(0,e) for (_,e) in cum_windows if e<=win]
    shell_windows=[(s,s+step) for s in range(0,win,step)]

    rows=[]; null_rows=[]
    cum_diffs={w:[] for w in cum_windows}
    shell_diffs={w:[] for w in shell_windows}
    cum_null={w:[] for w in cum_windows}
    shell_null={w:[] for w in shell_windows}
    for gene,gg in df.groupby("gene"):
        can=gg[gg["is_canonical"]==True]; com=gg[gg["is_canonical"]==False]
        if len(can)==0 or len(com)==0: continue
        cr=can.iloc[0]
        c_arr,c_dist=sig_arr(str(cr["chrom"]),int(cr["center"]),cr["strand"])
        if c_arr is None: continue
        comp_arrs=[]
        for _,rr in com.iterrows():
            a,dd=sig_arr(str(rr["chrom"]),int(rr["center"]),rr["strand"])
            if a is not None: comp_arrs.append((a,dd))
        if not comp_arrs: continue
        c_tot=load_in(c_arr,c_dist,0,win)
        c_dn =frac_pos_in(c_arr,c_dist,0,win)
        c_mx =float(np.nanmax(c_arr)) if c_arr.size else np.nan
        comp_tot=np.nanmean([load_in(a,dd,0,win) for a,dd in comp_arrs])
        comp_dn =np.nanmean([frac_pos_in(a,dd,0,win) for a,dd in comp_arrs])
        comp_mx =np.nanmean([float(np.nanmax(a)) for a,dd in comp_arrs])
        rows.append(dict(gene=gene, canon_max=c_mx,comp_max=comp_mx,
            canon_density=c_dn,comp_density=comp_dn, canon_total=c_tot,comp_total=comp_tot))
        for w in cum_windows:
            cc=load_in(c_arr,c_dist,*w); co=np.nanmean([load_in(a,dd,*w) for a,dd in comp_arrs])
            if np.isfinite(cc) and np.isfinite(co): cum_diffs[w].append(co-cc)
        for w in shell_windows:
            cc=load_in(c_arr,c_dist,*w); co=np.nanmean([load_in(a,dd,*w) for a,dd in comp_arrs])
            if np.isfinite(cc) and np.isfinite(co): shell_diffs[w].append(co-cc)
        if len(comp_arrs)>=2:
            ii,jj=rng.choice(len(comp_arrs),size=2,replace=False)
            if rng.random()<0.5: ii,jj=jj,ii
            aA,dA=comp_arrs[ii]; aB,dB=comp_arrs[jj]
            for w in cum_windows:
                lo=load_in(aA,dA,*w); hi=load_in(aB,dB,*w)
                if np.isfinite(lo) and np.isfinite(hi): cum_null[w].append(hi-lo)
            for w in shell_windows:
                lo=load_in(aA,dA,*w); hi=load_in(aB,dB,*w)
                if np.isfinite(lo) and np.isfinite(hi): shell_null[w].append(hi-lo)
        if len(comp_arrs)>=2:
            ii,jj=rng.choice(len(comp_arrs),size=2,replace=False)
            if rng.random()<0.5: ii,jj=jj,ii
            a=load_in(comp_arrs[ii][0],comp_arrs[ii][1],0,win)
            b=load_in(comp_arrs[jj][0],comp_arrs[jj][1],0,win)
            if np.isfinite(a) and np.isfinite(b):
                null_rows.append(dict(gene=gene, null_total=b-a,
                                      null_density=np.nan, null_max=np.nan))
    P=pd.DataFrame(rows); N=pd.DataFrame(null_rows)
    if len(P)==0:
        print("[paired] no paired genes found",file=sys.stderr); return
    P.to_csv(f"{out_prefix}_paired.tsv",sep="\t",index=False)
    N.to_csv(f"{out_prefix}_paired_null.tsv",sep="\t",index=False)

    from numpy import median
    def wilson(k,n,z=1.96):
        if n==0: return (np.nan,np.nan)
        phat=k/n; den=1+z*z/n
        centre=(phat+z*z/(2*n))/den
        half=(z*np.sqrt(phat*(1-phat)/n + z*z/(4*n*n)))/den
        return (centre-half, centre+half)
    def summ(diffs):
        d=np.array(diffs,float); d=d[np.isfinite(d)]
        if len(d)==0: return dict(median=np.nan,med_lo=np.nan,med_hi=np.nan,
                                  frac_pos=np.nan,frac_lo=np.nan,frac_hi=np.nan,n=0,p=1.0)
        med=float(median(d))
        k=int(np.sum(d>0)); nnz=int(np.sum(d!=0))
        frac=float(k/nnz) if nnz>0 else np.nan
        from math import erfc
        z=(k-0.5*nnz)/np.sqrt(0.25*nnz) if nnz>0 else 0.0
        p=0.5*erfc(z/np.sqrt(2))
        lo,hi=wilson(k,nnz)
        B=2000
        bs=rng.choice(d,size=(B,len(d)),replace=True)
        bmed=np.median(bs,axis=1)
        med_lo,med_hi=float(np.percentile(bmed,2.5)),float(np.percentile(bmed,97.5))
        return dict(median=med,med_lo=med_lo,med_hi=med_hi,
                    frac_pos=frac,frac_lo=lo,frac_hi=hi,n=len(d),p=p)
    sweep=[]
    print(f"\n=== WITHIN-GENE paired by DISTANCE [{metric}] (n={len(P)} genes) ===",file=sys.stderr)
    print(f"  real = canonical vs competitor;  null = two RANDOM peaks in same gene",file=sys.stderr)
    print(f"  CUMULATIVE windows (0..X):",file=sys.stderr)
    for w in cum_windows:
        r=summ(cum_diffs[w]); nu=summ(cum_null[w])
        sweep.append(dict(scheme="cumulative",kind="real",d0=w[0],d1=w[1],**r))
        sweep.append(dict(scheme="cumulative",kind="random",d0=w[0],d1=w[1],**nu))
        print(f"    0-{w[1]:<5} real {r['frac_pos']:.0%} [{r['frac_lo']:.0%}-{r['frac_hi']:.0%}] "
              f"med={r['median']:+.2f} p={r['p']:.1e}   null {nu['frac_pos']:.0%}",file=sys.stderr)
    print(f"  PER-SHELL windows:",file=sys.stderr)
    for w in shell_windows:
        r=summ(shell_diffs[w]); nu=summ(shell_null[w])
        sweep.append(dict(scheme="shell",kind="real",d0=w[0],d1=w[1],**r))
        sweep.append(dict(scheme="shell",kind="random",d0=w[0],d1=w[1],**nu))
        print(f"    {w[0]}-{w[1]:<5} real {r['frac_pos']:.0%} [{r['frac_lo']:.0%}-{r['frac_hi']:.0%}] "
              f"med={r['median']:+.2f} p={r['p']:.1e}   null {nu['frac_pos']:.0%}",file=sys.stderr)
    pd.DataFrame(sweep).to_csv(f"{out_prefix}_paired_bydist.tsv",sep="\t",index=False)
    print(f"  [out] {out_prefix}_paired.tsv, _paired_null.tsv, _paired_bydist.tsv",file=sys.stderr)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scripts_dir",default=None)
    ap.add_argument("--shortlist"); ap.add_argument("--cpa_bed"); ap.add_argument("--atlas_bed")
    ap.add_argument("--atlas_ann_split",default=None,help="comma list e.g. TE,IN")
    ap.add_argument("--genes_bed"); ap.add_argument("--random_n",type=int,default=0)
    ap.add_argument("--internal_exon_gtf"); ap.add_argument("--internal_exon_n",type=int,default=0)
    ap.add_argument("--reference_fasta",required=True)
    ap.add_argument("--u1_lookup"); ap.add_argument("--me5_lookup")
    ap.add_argument("--metric",choices=["me5","u1"],required=True)
    ap.add_argument("--upstream_nt",type=int,default=2000,help="kept for CLI compat; scan window is --max_nt")
    ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--max_nt",type=int,default=2000,help="max UPSTREAM distance to profile")
    ap.add_argument("--step",type=int,default=100)
    ap.add_argument("--density_thresh",type=float,default=None,
                    help="signal threshold for the DENSITY panel (fraction of positions "
                         "above it). Default: 0 for me5 (real donor), 6.0 for u1 (-dG).")
    ap.add_argument("--pad",type=int,default=10)
    ap.add_argument("--max_peaks",type=int,default=20000)
    ap.add_argument("--paired",action="store_true",
                    help="ALSO run the within-gene paired canonical-vs-competitor test "
                         "(cancels gene-position confound; the honest suppression test)")
    ap.add_argument("--out",default="fivess_profile")
    a=ap.parse_args()
    if a.density_thresh is None:
        a.density_thresh = 0.0 if a.metric=="me5" else 6.0
    if a.scripts_dir: sys.path.insert(0,a.scripts_dir)

    from extract_splicing_features import GenomeFasta, init_maxentscan
    fa=GenomeFasta(a.reference_fasta)
    score5=None; u1=None
    if a.metric=="me5":
        ms=init_maxentscan()
        if isinstance(ms,(tuple,list)): score5=ms[0]
        elif isinstance(ms,dict): score5=ms.get("score5")
        if score5 is None: sys.exit("[err] could not resolve score5")
    else:
        if not a.u1_lookup: sys.exit("[err] --metric u1 needs --u1_lookup")
        u1=load_u1_lookup(a.u1_lookup)

    parts=[]
    if a.shortlist: parts.append(from_shortlist(a.shortlist))
    if a.cpa_bed: parts.append(from_cpa_bed(a.cpa_bed))
    if a.atlas_bed:
        keep=set(a.atlas_ann_split.split(",")) if a.atlas_ann_split else None
        parts.append(from_atlas(a.atlas_bed, ann_keep=keep))
    if a.random_n>0:
        if not a.genes_bed: sys.exit("[err] --random_n needs --genes_bed")
        parts.append(from_random_genebody(a.genes_bed,a.random_n))
    if a.internal_exon_n>0:
        if not a.internal_exon_gtf: sys.exit("[err] --internal_exon_n needs --internal_exon_gtf")
        parts.append(from_internal_exons(a.internal_exon_gtf,a.internal_exon_n))
    if not parts: sys.exit("[err] no sources given")
    sites=pd.concat(parts,ignore_index=True)

    capped=[]
    for cls,g in sites.groupby("cls"):
        if len(g)>a.max_peaks: g=g.sample(a.max_peaks,random_state=0)
        capped.append(g)
    sites=pd.concat(capped,ignore_index=True)
    print("[data] peaks per class:",file=sys.stderr)
    for cls,g in sites.groupby("cls"): print(f"   {cls:<22}{len(g):>7,}",file=sys.stderr)

    win=a.max_nt
    edges=np.arange(0, a.max_nt+a.step, a.step)
    from collections import defaultdict
    shell_sum=defaultdict(lambda: np.zeros(len(edges)-1))
    shell_cnt=defaultdict(lambda: np.zeros(len(edges)-1))
    cum_sum =defaultdict(lambda: np.zeros(len(edges)-1))
    cum_cnt =defaultdict(lambda: np.zeros(len(edges)-1))
    dens_sum=defaultdict(lambda: np.zeros(len(edges)-1))
    dens_cnt=defaultdict(lambda: np.zeros(len(edges)-1))
    cumd_sum=defaultdict(lambda: np.zeros(len(edges)-1))
    cumd_cnt=defaultdict(lambda: np.zeros(len(edges)-1))
    tot_sum =defaultdict(lambda: np.zeros(len(edges)-1))
    tot_cnt =defaultdict(lambda: np.zeros(len(edges)-1))
    thr=a.density_thresh

    for _,r in sites.iterrows():
        chrom,center,strand,cls=r["chrom"],int(r["center"]),r["strand"],r["cls"]
        if strand=="+": gstart,gend=center-win-a.pad, center+a.pad
        else:           gstart,gend=center-a.pad, center+win+a.pad
        try: seq=fa.fetch(chrom,gstart,gend,strand=strand)
        except Exception: continue
        site_idx = win+a.pad if strand=="+" else win+a.pad-1
        if a.metric=="me5": arr=scan_me5(seq, max(0,site_idx-win), site_idx, score5)
        else:               arr=scan_u1 (seq, max(0,site_idx-win), site_idx, u1)
        if arr.size==0: continue
        pos = (site_idx - np.arange(arr.size))
        for b in range(len(edges)-1):
            lo,hi=edges[b],edges[b+1]
            shell_w=max(hi-lo,1)
            m=(pos>=lo)&(pos<hi)&np.isfinite(arr)
            if m.any():
                shell_sum[cls][b]+=np.nanmax(arr[m]); shell_cnt[cls][b]+=1
                dens_sum[cls][b]+=float(np.sum(arr[m]>=thr))/shell_w*100.0; dens_cnt[cls][b]+=1
                pos_sig=arr[m][arr[m]>=thr]
                tot_sum[cls][b]+=float(np.sum(pos_sig))/shell_w*100.0; tot_cnt[cls][b]+=1
            mc=(pos<hi)&np.isfinite(arr)
            if mc.any():
                cum_sum[cls][b]+=np.nanmax(arr[mc]); cum_cnt[cls][b]+=1
                cumd_sum[cls][b]+=float(np.sum(arr[mc]>=thr))/max(hi,1)*100.0; cumd_cnt[cls][b]+=1

    xs=(edges[:-1]+edges[1:])/2
    rows=[]
    for cls in shell_cnt:
        for b in range(len(xs)):
            sh = shell_sum[cls][b]/shell_cnt[cls][b] if shell_cnt[cls][b]>0 else np.nan
            cu = cum_sum[cls][b]/cum_cnt[cls][b] if cum_cnt[cls][b]>0 else np.nan
            dsh= dens_sum[cls][b]/dens_cnt[cls][b] if dens_cnt[cls][b]>0 else np.nan
            dcu= cumd_sum[cls][b]/cumd_cnt[cls][b] if cumd_cnt[cls][b]>0 else np.nan
            tsh= tot_sum[cls][b]/tot_cnt[cls][b] if tot_cnt[cls][b]>0 else np.nan
            rows.append(dict(cls=cls,x_nt=float(xs[b]),scheme="shells",mean_score=sh,
                             density=dsh,total=tsh,n=int(shell_cnt[cls][b])))
            rows.append(dict(cls=cls,x_nt=float(xs[b]),scheme="cumulative",mean_score=cu,
                             density=dcu,total=np.nan,n=int(cum_cnt[cls][b])))
    out=pd.DataFrame(rows)
    out.to_csv(f"{a.out}_long.tsv",sep="\t",index=False)

    lab = "5'ss MaxEntScan (upstream donor)" if a.metric=="me5" else "U1 hybridization -dG (upstream)"
    dlab = ("real 5'ss donors per 100nt (upstream)" if a.metric=="me5"
            else f"U1 sites (-dG\u2265{a.density_thresh:g}) per 100nt (upstream)")
    print(f"\n=== UPSTREAM {a.metric}: MAX (strongest) and DENSITY (field) by distance ===",file=sys.stderr)
    for scheme in ("shells","cumulative"):
        print(f"  [{scheme}] MAX:",file=sys.stderr)
        sub=out[out.scheme==scheme]
        for cls in sorted(sub.cls.unique()):
            s2=sub[sub.cls==cls].sort_values("x_nt")
            vals=", ".join(f"{int(x)}:{v:.2f}" for x,v in zip(s2.x_nt,s2.mean_score) if np.isfinite(v))
            print(f"    {cls:<22}{vals[:80]} ...",file=sys.stderr)
    print(f"  [shells] DENSITY (field, the suppression-relevant quantity):",file=sys.stderr)
    sub=out[out.scheme=="shells"]
    for cls in sorted(sub.cls.unique()):
        s2=sub[sub.cls==cls].sort_values("x_nt")
        vals=", ".join(f"{int(x)}:{v:.3f}" for x,v in zip(s2.x_nt,s2.density) if np.isfinite(v))
        print(f"    {cls:<22}{vals[:80]} ...",file=sys.stderr)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        PAL={"all CPA":"#C65B3C","contested CPA":"#E08A6F","internal competitor":"#5B7C6F",
             "atlas TE":"#3E6B8B","atlas IN":"#7BA3C0","random (gene body)":"#999999",
             "random internal exon":"#C9A14A"}
        def draw(ax, sub, ycol, ylab, title):
            for cls in sorted(sub.cls.unique()):
                s2=sub[sub.cls==cls].sort_values("x_nt")
                col=PAL.get(cls,"#888888")
                isnull=cls in ("random (gene body)","random internal exon")
                ax.plot(s2.x_nt,s2[ycol],("--" if isnull else "-o"),ms=3,color=col,
                        label=cls,linewidth=1.6 if isnull else 1.2)
            ax.set_title(title); ax.set_xlabel("upstream distance (nt)"); ax.set_ylabel(ylab)
        fig,axes=plt.subplots(1,4,figsize=(22,5))
        draw(axes[0], out[out.scheme=="cumulative"], "mean_score", lab, "cumulative (max)")
        draw(axes[1], out[out.scheme=="shells"],     "mean_score", lab, "shells (max)")
        draw(axes[2], out[out.scheme=="shells"],     "density",    dlab, "shells (DENSITY / field)")
        tlab=("total real-donor load /100nt" if a.metric=="me5" else f"total U1 load (-dG\u2265{a.density_thresh:g}) /100nt")
        draw(axes[3], out[out.scheme=="shells"],     "total",      tlab, "shells (TOTAL load / sum)")
        axes[3].legend(fontsize=7,frameon=False)
        fig.suptitle(f"Upstream donor/U1 vs distance [{a.metric}]: max, density, total")
        fig.tight_layout()
        png=f"{a.out}.png"; fig.savefig("/tmp/_5ss.png",dpi=160); import shutil; shutil.copy("/tmp/_5ss.png",png)
        try: fig.savefig(f"{a.out}.pdf")
        except Exception: pass
        print(f"\n[out] {png}, {a.out}_long.tsv",file=sys.stderr)
    except Exception as e:
        print(f"[warn] plot failed ({e}); TSV written",file=sys.stderr)

    if a.paired:
        if not a.shortlist:
            print("[paired] needs --shortlist; skipping",file=sys.stderr)
        else:
            paired_within_gene(fa, a.shortlist, a.metric, score5, u1,
                               a.max_nt, a.pad, a.density_thresh, a.out)

if __name__=="__main__": main()