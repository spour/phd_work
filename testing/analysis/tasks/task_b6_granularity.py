#!/usr/bin/env python3
# b6.1: is the maxent5|u1 collapse (+0.077 -> +0.001) real subsumption
import sys, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; WIN=500

Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; gi=Z["gene_idx"]
u1d525=Z["dens_5.25"]; u1c8=Z["dens_8.0"]*0.5
zc=np.load("fig3_ucube.npz"); mc=zc["me5cube"]; edges=zc["edges"]; mthr=list(zc["me5_thrs"])
lo,hi=int(np.searchsorted(edges,-500)),int(np.searchsorted(edges,0))
me5c8=mc[:,lo:hi,3+mthr.index(8.0)].sum(1)
NP=len(isc)

# gc per window (light rescan: base composition only)
B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
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
        bp=lambda p: off+(p//lb)*lw+(p%lb)
        s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st=="-" else q
G=Faidx(FASTA)
rows=[]
with open("fig3_ucube_meta.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["chrom"],int(r["center"]),r["strand"]))
gc=np.full(NP,np.nan)
for i,(c,ctr,st) in enumerate(rows):
    seq=G.fetch(c,ctr,ctr+WIN,"-") if st=="-" else G.fetch(c,ctr-WIN,ctr,"+")
    if not seq: continue
    t=B[np.frombuffer(seq.encode("ascii"),np.uint8)]; t=t[t>=0]
    if t.size: gc[i]=((t==1)|(t==2)).mean()
    if (i+1)%20000==0: log(f"  gc {i+1:,}")

# within-gene engine (closed-form null)
order=np.argsort(gi,kind="stable"); gs=gi[order]
cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=order[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1: gmemb.append(idx);gcanon.append(int(c[0]))
NG=len(gmemb)
comp_flat=np.concatenate([m[m!=c] for m,c in zip(gmemb,gcanon)])
comp_gene=np.concatenate([np.full((m!=c).sum(),i) for i,(m,c) in enumerate(zip(gmemb,gcanon))])
canon_arr=np.array(gcanon)
def obs_ng(val):
    vc=val[canon_arr][comp_gene]; vk=val[comp_flat]
    win=vk>vc; loss=vk<vc
    w=np.bincount(comp_gene,win,NG); l=np.bincount(comp_gene,loss,NG); info=w+l
    return np.where(info>0,w/np.maximum(info,1),np.nan)
def null_ng(val):
    nu=np.full(NG,np.nan)
    for i,m in enumerate(gmemb):
        v=val[m]; sv=np.sort(v); mm=len(v)
        less=np.searchsorted(sv,v,"left"); great=mm-np.searchsorted(sv,v,"right")
        info=(less+great)>0
        if info.any(): nu[i]=(great[info]/(less[info]+great[info])).mean()
    return nu
def report(val,nm):
    wr=obs_ng(val); nu=null_ng(val); ok=np.isfinite(wr)
    exg=(wr-nu)[ok]; b=exg[rng.integers(0,len(exg),size=(2000,len(exg)))].mean(1)
    return dict(nm=nm,obs=wr[ok].mean(),null=nu[ok].mean(),exc=exg.mean(),
                lo=np.percentile(b,2.5),hi=np.percentile(b,97.5),genes=int(ok.sum()))
def resid(y,x):
    ok=np.isfinite(y)&np.isfinite(x); bta=np.polyfit(x[ok],y[ok],1); return y-(bta[0]*x+bta[1])

shufU1=u1d525.copy(); rng.shuffle(shufU1)
print("\n============  B6.1a  method + gene retention (residualisation) ============")
for val,nm in [(resid(u1d525,me5c8),"U1@5.25 | MaxEnt5"),(resid(me5c8,u1d525),"MaxEnt5 | U1@5.25")]:
    r=report(val,nm); print(f"  {nm:<22} genes retained {r['genes']:,}  (residual is continuous, full set kept)")

print("\n============  B6.1b  negative controls: MaxEnt5 conditioned on dense non-U1 features ============")
print(f"  MaxEnt5 corr with: U1@5.25 r={np.corrcoef(me5c8,u1d525)[0,1]:+.3f}   "
      f"shuffledU1 r={np.corrcoef(me5c8,shufU1)[0,1]:+.3f}   GC r={np.corrcoef(me5c8[np.isfinite(gc)],gc[np.isfinite(gc)])[0,1]:+.3f}")
for val,nm in [(me5c8,"MaxEnt5 (marginal)"),(resid(me5c8,u1d525),"MaxEnt5 | REAL U1@5.25"),
               (resid(me5c8,shufU1),"MaxEnt5 | shuffled U1"),(resid(me5c8,gc),"MaxEnt5 | GC content")]:
    r=report(val,nm); print(f"  {nm:<24} obs {r['obs']:.4f}  null {r['null']:.4f}  excess {r['exc']:+.4f}  [{r['lo']:+.3f},{r['hi']:+.3f}]")

print("\n============  B6.1c  symmetric granularity: U1 count>=8 vs MaxEnt5 count>=8 ============")
for val,nm in [(u1c8,"U1>=8 (marginal)"),(resid(u1c8,me5c8),"U1>=8 | MaxEnt5>=8"),
               (me5c8,"MaxEnt5>=8 (marginal)"),(resid(me5c8,u1c8),"MaxEnt5>=8 | U1>=8")]:
    r=report(val,nm); print(f"  {nm:<22} obs {r['obs']:.4f}  null {r['null']:.4f}  excess {r['exc']:+.4f}  [{r['lo']:+.3f},{r['hi']:+.3f}]")
print("  presence/absence (>=1 hit):")
u1p=(u1c8>0).astype(float); me5p=(me5c8>0).astype(float)
for val,nm in [(u1p,"U1 present (marginal)"),(resid(u1p,me5p),"U1 present | MaxEnt5 present"),
               (me5p,"MaxEnt5 present (marginal)"),(resid(me5p,u1p),"MaxEnt5 present | U1 present")]:
    r=report(val,nm); print(f"  {nm:<28} obs {r['obs']:.4f}  null {r['null']:.4f}  excess {r['exc']:+.4f}  [{r['lo']:+.3f},{r['hi']:+.3f}]")
