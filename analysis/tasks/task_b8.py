#!/usr/bin/env python3
# b8 - the gc problem
import sys, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; U1NPY="../train/splice_train/data/u1_lookup.npy"
META="fig3_ucube_meta.tsv"; WIN=500; PAS_W=50; DN_W=50

# u1 score lookup (negated dg), identical to builder
B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
arr=np.load(U1NPY,allow_pickle=True); d=arr.item() if arr.dtype==object else dict(arr)
kb_=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
SCORE=np.empty(4**9,np.float64); SCORE[(B[kb_]@POW)]=-np.fromiter(d.values(),np.float64,len(d))
def score_window(tok):
    if tok.shape[0]<9: return np.empty(0)
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    out=np.full(sw.shape[0],np.nan); out[good]=SCORE[sw[good]@POW]; return out

class Faidx:
    _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
    def __init__(s,fa):
        s.fh=open(fa,"rb"); s.idx={}
        for L in open(fa+".fai"):
            p=L.split(); s.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
    def fetch(s,c,a,b,st="+"):
        c=c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln,off,lb,lw=s.idx[c]; a=max(0,a); b=min(b,Ln)
        if b<=a: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st=="-" else q
G=Faidx(FASTA)

PAS12=["AATAAA","ATTAAA","TATAAA","AGTAAA","AAGAAA","AATACA",
       "CATAAA","GATAAA","AATATA","AATGAA","TTTAAA","ACTAAA"]
PAS_STRONG=("AATAAA","ATTAAA")

Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; gi=Z["gene_idx"]
u1d525=Z["dens_5.25"]; excess525=Z["excess_525"]
NP=len(isc)

rows=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["chrom"],int(r["center"]),r["strand"]))
assert len(rows)==NP

u1_mean=np.full(NP,np.nan); u1_sum=np.full(NP,np.nan); u1_max=np.full(NP,np.nan)
gc_up=np.full(NP,np.nan)
pas_any=np.zeros(NP); pas_strong=np.zeros(NP); pas_count=np.zeros(NP)
t_dn=np.full(NP,np.nan); gu_dn=np.full(NP,np.nan); at_dn=np.full(NP,np.nan)
for i,(c,ctr,st) in enumerate(rows):
    seq = G.fetch(c,ctr,ctr+WIN,"-") if st=="-" else G.fetch(c,ctr-WIN,ctr,"+")
    if seq:
        tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
        sc=score_window(tok); v=sc[np.isfinite(sc)]
        if v.size: u1_mean[i]=v.mean(); u1_sum[i]=v.sum(); u1_max[i]=v.max()
        tg=tok[tok>=0]
        if tg.size: gc_up[i]=((tg==1)|(tg==2)).mean()
        pas_seq=seq[-PAS_W:]
        cnt=sum(pas_seq.count(h) for h in PAS12)
        pas_count[i]=cnt; pas_any[i]=1.0 if cnt>0 else 0.0
        pas_strong[i]=1.0 if any(h in pas_seq for h in PAS_STRONG) else 0.0
    dn = G.fetch(c,ctr-DN_W,ctr,"-") if st=="-" else G.fetch(c,ctr,ctr+DN_W,"+")
    if dn:
        dt=B[np.frombuffer(dn.encode("ascii"),np.uint8)]; dg=dt[dt>=0]
        if dg.size:
            t_dn[i]=(dg==3).mean(); at_dn[i]=((dg==0)|(dg==3)).mean()
            di=dg[:-1]*4+dg[1:]
            gu_dn[i]=np.isin(di,(11,14)).mean() if di.size else np.nan
    if (i+1)%20000==0: log(f"  scan {i+1:,}/{NP:,}")

# within-gene engine (closed-form null + gene bootstrap on excess)
order=np.argsort(gi,kind="stable"); gs=gi[order]
cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=order[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1: gmemb.append(idx); gcanon.append(int(c[0]))
NG=len(gmemb)
comp_flat=np.concatenate([m[m!=c] for m,c in zip(gmemb,gcanon)])
comp_gene=np.concatenate([np.full((m!=c).sum(),i) for i,(m,c) in enumerate(zip(gmemb,gcanon))])
canon_arr=np.array(gcanon)
def obs_pergene(val):
    vc=val[canon_arr][comp_gene]; vk=val[comp_flat]
    win=vk>vc; loss=vk<vc; tie=vk==vc
    w=np.bincount(comp_gene,win,NG); l=np.bincount(comp_gene,loss,NG)
    t=np.bincount(comp_gene,tie,NG); n=np.bincount(comp_gene,None,NG)
    info=w+l; wr=np.where(info>0,w/np.maximum(info,1),np.nan); half=(w+0.5*t)/n
    return wr,half,t.sum()/n.sum()
def null_pergene(val):
    nu=np.full(NG,np.nan)
    for i,m in enumerate(gmemb):
        v=val[m]; sv=np.sort(v); mm=len(v)
        less=np.searchsorted(sv,v,"left"); gr=mm-np.searchsorted(sv,v,"right"); info=(less+gr)>0
        if info.any(): nu[i]=(gr[info]/(less[info]+gr[info])).mean()
    return nu
def report(val,nm):
    wr,half,tf=obs_pergene(val); nu=null_pergene(val); ok=np.isfinite(wr)
    exg=(wr-nu)[ok]; b=exg[rng.integers(0,len(exg),size=(2000,len(exg)))].mean(1)
    return dict(nm=nm,obs=wr[ok].mean(),null=nu[ok].mean(),exc=exg.mean(),
                lo=np.percentile(b,2.5),hi=np.percentile(b,97.5),
                half=np.nanmean(half),ties=tf,genes=int(ok.sum()))
def line(r): return (f"  {r['nm']:<26} obs {r['obs']:.4f}  null {r['null']:.4f}  "
                     f"excess {r['exc']:+.4f}  [{r['lo']:+.3f},{r['hi']:+.3f}]  "
                     f"ties={r['ties']:.3f}  ties.5={r['half']:.4f}  n={r['genes']}")

print("\n============  B8.1  continuous U1 encodings vs GC (same within-gene axis)  ============")
print("  competitor>canonical => excess>0 ; GC is the +0.1198 benchmark\n")
for val,nm in [(gc_up,"GC content (benchmark)"),
               (u1d525,"U1 density @5.25 (thresh)"),
               (u1_mean,"U1 mean score (cont.)"),
               (u1_sum,"U1 sum energy (cont.)"),
               (u1_max,"U1 max score (cont.)"),
               (excess525,"u1_excess @5.25")]:
    print(line(report(val,nm)))

cands={"u1_mean":u1_mean,"u1_sum":u1_sum,"u1_max":u1_max}
bestnm=max(cands,key=lambda k:report(cands[k],k)["exc"]); bestU1=cands[bestnm]
print(f"\n  best continuous U1 encoding by excess: {bestnm}")

# b8.2 reciprocal matched-binarisation, null 0.5 (label symmetry within stratum)
print("\n============  B8.2  reciprocal U1 vs GC, MATCHED BINARISATION (no OLS)  ============")
def cond_winrate(Aval,Sbin):
    pg=np.full(NG,np.nan)
    for i,(m,c) in enumerate(zip(gmemb,gcanon)):
        ks=m[m!=c]; matched=ks[Sbin[ks]==Sbin[c]]
        if matched.size==0: continue
        dd=Aval[matched]-Aval[c]; nz=dd!=0
        if nz.any(): pg[i]=(dd[nz]>0).mean()
    ok=np.isfinite(pg); v=pg[ok]
    b=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)
    return dict(obs=v.mean(),lo=np.percentile(b,2.5),hi=np.percentile(b,97.5),genes=int(ok.sum()))
p_u1=np.mean(u1d525>0)
gc_thr=np.nanquantile(gc_up,1-p_u1); GC_hi=(gc_up>=gc_thr).astype(float)
U1_hi525=(u1d525>0).astype(float)
best_thr=np.nanquantile(bestU1,1-p_u1); U1_hi_best=(bestU1>=best_thr).astype(float)
print(f"  matched prevalence p(U1@5.25>0)={p_u1:.3f} ; GC binarised at >={gc_thr:.3f} ; {bestnm}>={best_thr:.3g}")
for Aval,Sbin,nm in [(u1d525,GC_hi,"U1@5.25 | GC (GC-matched)"),
                     (bestU1,GC_hi,f"{bestnm} | GC (GC-matched)"),
                     (excess525,GC_hi,"u1_excess | GC (GC-matched)"),
                     (gc_up,U1_hi525,"GC | U1@5.25 (U1-matched)"),
                     (gc_up,U1_hi_best,f"GC | {bestnm} (U1-matched)")]:
    r=cond_winrate(Aval,Sbin)
    print(f"  {nm:<32} win-rate {r['obs']:.4f}  [{r['lo']:.3f},{r['hi']:.3f}]  null 0.500  n={r['genes']} genes")

# b8.3 corr(u1_excess, gc)
print("\n============  B8.3  corr(u1_excess, GC) per peak  ============")
ok=np.isfinite(excess525)&np.isfinite(gc_up)
print(f"  corr(u1_excess, GC) = {np.corrcoef(excess525[ok],gc_up[ok])[0,1]:+.3f}")
print(f"  corr(U1@5.25 raw, GC) = {np.corrcoef(u1d525[ok],gc_up[ok])[0,1]:+.3f}   "
      f"corr({bestnm}, GC) = {np.corrcoef(bestU1[np.isfinite(bestU1)&ok],gc_up[np.isfinite(bestU1)&ok])[0,1]:+.3f}")

print("\n============  B8.4  PAS architecture alternative  ============")
print("  real ends (canonical) expected to have MORE/stronger PAS => excess<0\n")
for val,nm in [(pas_any,"any PAS hexamer (0/1)"),
               (pas_strong,"strong PAS AATAAA/ATTAAA"),
               (pas_count,"PAS hexamer count"),
               (at_dn,"A/T frac downstream 50"),
               (t_dn,"U(T) frac downstream 50"),
               (gu_dn,"GU/UG dinuc downstream 50")]:
    print(line(report(val,nm)))
print(f"\n  prevalence: strong-PAS present canonical {pas_strong[canon_arr].mean():.3f} vs "
      f"competitor {pas_strong[comp_flat].mean():.3f}")
print(f"  corr(GC, strong-PAS)={np.corrcoef(gc_up[np.isfinite(gc_up)],pas_strong[np.isfinite(gc_up)])[0,1]:+.3f} ; "
      f"corr(U1@5.25, strong-PAS)={np.corrcoef(u1d525,pas_strong)[0,1]:+.3f}")
print("\n  do U1(best)/u1_excess survive conditioning on strong-PAS presence (matched)?")
for Aval,nm in [(u1d525,"U1@5.25 | strong-PAS matched"),
                (bestU1,f"{bestnm} | strong-PAS matched"),
                (excess525,"u1_excess | strong-PAS matched")]:
    r=cond_winrate(Aval,pas_strong)
    print(f"  {nm:<34} win-rate {r['obs']:.4f}  [{r['lo']:.3f},{r['hi']:.3f}]  null 0.500  n={r['genes']}")
print("\n[done]")
