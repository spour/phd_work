#!/usr/bin/env python3
# b7 + alignment/validation
import sys, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)

Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; gi=Z["gene_idx"]
dens4=Z["dens_4.0"]; u1d525=Z["dens_5.25"]
zc=np.load("fig3_ucube.npz"); mc=zc["me5cube"]; edges=zc["edges"]; mthr=list(zc["me5_thrs"])
lo,hi=int(np.searchsorted(edges,-500)),int(np.searchsorted(edges,0))
me5c8=mc[:,lo:hi,3+mthr.index(8.0)].sum(1)
NP=len(isc)

# alignment: u1_grid_cache @4 vs my dens_4.0
zg=np.load("u1_grid_cache.npz",allow_pickle=True); cols=list(zg["cols"]); M=zg["M"]
grid4=M[:,cols.index("up_500|density_per_kb@4")].astype(float)
meta_gene=[];
with open("fig3_ucube_meta.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"): meta_gene.append(r["gene"])
grid_gene=[str(g) for g in zg["gene"]] if "gene" in zg else None
print("=== ALIGNMENT / SCAN VALIDATION ===")
print(f"  rows: u1_grid_cache {M.shape[0]:,}  u1_window_cache {NP:,}  meta {len(meta_gene):,}")
if grid_gene is not None:
    gm=np.mean([a==b for a,b in zip(grid_gene,meta_gene)])
    print(f"  gene-label match (grid vs meta, by row): {gm:.6f}")
d=np.abs(dens4-grid4)
print(f"  dens_4.0 vs grid @4: exact-match {np.mean(d<1e-6):.6f}  corr {np.corrcoef(dens4,grid4)[0,1]:.6f}  max|diff| {d.max():.4g} mean|diff| {d.mean():.4g}")

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
def pergene(val):
    vc=val[canon_arr][comp_gene]; vk=val[comp_flat]
    win=vk>vc; loss=vk<vc; tie=vk==vc
    w=np.bincount(comp_gene,win,NG); l=np.bincount(comp_gene,loss,NG); t=np.bincount(comp_gene,tie,NG); n=np.bincount(comp_gene,None,NG)
    info=w+l; wr=np.where(info>0,w/np.maximum(info,1),np.nan); half=(w+0.5*t)/n
    return wr,half
def null_ng(val):
    nu=np.full(NG,np.nan)
    for i,m in enumerate(gmemb):
        v=val[m]; sv=np.sort(v); mm=len(v)
        less=np.searchsorted(sv,v,"left"); gr=mm-np.searchsorted(sv,v,"right"); info=(less+gr)>0
        if info.any(): nu[i]=(gr[info]/(less[info]+gr[info])).mean()
    return nu
wr4,half4=pergene(dens4); ok=np.isfinite(wr4)
print(f"  dens_4.0 through within-gene code: ties=0.5 {np.nanmean(half4):.4f} (known 0.6187) ; sign-test {wr4[ok].mean():.4f} (known 0.6200)")

# b7.1 dilution: pair-level split
print("\n=== B7.1  dilution / tie-destruction test (pair level) ===")
vc_me5=me5c8[canon_arr][comp_gene]; vk_me5=me5c8[comp_flat]
me5_diff=vk_me5!=vc_me5; me5_bothzero=(vk_me5==0)&(vc_me5==0)
print(f"  pairs total {len(comp_flat):,} ; raw MaxEnt5 informative {int(me5_diff.sum()):,} ; both-zero {me5_bothzero.mean():.4f}")
print(f"  marginal MaxEnt5 pooled tie-excl win-rate (raw-informative pairs): {(vk_me5[me5_diff]>vc_me5[me5_diff]).mean():.4f}\n")
def resid(y,x):
    ok=np.isfinite(y)&np.isfinite(x); b=np.polyfit(x[ok],y[ok],1); return y-(b[0]*x+b[1])
shuf=u1d525.copy(); rng.shuffle(shuf)
import os
armfeats=[("REAL U1@5.25",resid(me5c8,u1d525)),("shuffled U1",resid(me5c8,shuf))]
print(f"{'arm':<16}{'n_info':>9}{'subsetA(raw me5 diff)':>24}{'subsetB(raw me5 equal)':>24}")
for nm,f in armfeats:
    fc=f[canon_arr][comp_gene]; fk=f[comp_flat]; fdiff=fk!=fc
    A=fdiff&me5_diff; Bm=fdiff&(~me5_diff)
    wrA=(fk[A]>fc[A]).mean(); wrB=(fk[Bm]>fc[Bm]).mean()
    print(f"{nm:<16}{int(fdiff.sum()):>9,}   A n={int(A.sum()):>7,} wr={wrA:.4f}   B n={int(Bm.sum()):>7,} wr={wrB:.4f}")
print("  (dilution predicts subset A ~ marginal 0.70, subset B ~0.5 for shuffled; real-U1 B<0.5)")

# b7.2 gc within-gene
print("\n=== B7.2  is GC content itself competitor-associated within gene? ===")
FASTA="../train/data/hg38.fa"; WIN=500
B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
class Faidx:
    _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
    def __init__(s,fa):
        s.fh=open(fa,"rb"); s.idx={}
        for L in open(fa+".fai"):
            p=L.split(); s.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
    def fetch(s,c,a,b,st="+"):
        c=c if c in s.idx else c.lstrip("chr");
        if c not in s.idx: return None
        Ln,off,lb,lw=s.idx[c]; a=max(0,a); b=min(b,Ln)
        if b<=a: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st=="-" else q
G=Faidx(FASTA); rows=[]
with open("fig3_ucube_meta.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["chrom"],int(r["center"]),r["strand"]))
gc=np.full(NP,np.nan)
for i,(c,ctr,st) in enumerate(rows):
    seq=G.fetch(c,ctr,ctr+WIN,"-") if st=="-" else G.fetch(c,ctr-WIN,ctr,"+")
    if not seq: continue
    t=B[np.frombuffer(seq.encode("ascii"),np.uint8)]; t=t[t>=0]
    if t.size: gc[i]=((t==1)|(t==2)).mean()
wrg,halfg=pergene(gc); nu=null_ng(gc); okg=np.isfinite(wrg)
exc=(wrg-nu)[okg]
print(f"  GC content within-gene: obs {wrg[okg].mean():.4f}  null {nu[okg].mean():.4f}  excess {exc.mean():+.4f}  (competitors higher GC if >0)")
print(f"  corr(GC, MaxEnt5 count>=8) = {np.corrcoef(gc[np.isfinite(gc)],me5c8[np.isfinite(gc)])[0,1]:+.3f} ; corr(GC, U1@5.25) = {np.corrcoef(gc[np.isfinite(gc)],u1d525[np.isfinite(gc)])[0,1]:+.3f}")
