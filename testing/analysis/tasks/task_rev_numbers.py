#!/usr/bin/env python3
# reversal: exact numbers for the reframed layer-2 subtitles/caption, from caches only
import numpy as np
from collections import defaultdict
rng=np.random.default_rng(42)
comp=np.load("peak_compartment.npy"); gc=np.load("gc_up500.npy")
Z=np.load("u1_window_cache.npz"); u1=Z["dens_5.25"]; exc=Z["excess_525"]; isc=Z["isc"]; gi=Z["gene_idx"]
LAB=["TERM","INTERNAL","INTRON","OTHER"]

order=np.argsort(gi,kind="stable"); gs=gi[order]
cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=order[cut[j]:cut[j+1]]; cc=idx[isc[idx]==1]; kk=idx[isc[idx]==0]
    if len(cc)==1 and len(kk)>=1: gmemb.append(idx); gcanon.append(int(cc[0]))
NG=len(gmemb)
comp_flat=np.concatenate([m[m!=c] for m,c in zip(gmemb,gcanon)])
comp_gene=np.concatenate([np.full((m!=c).sum(),i) for i,(m,c) in enumerate(zip(gmemb,gcanon))])
canon_arr=np.array(gcanon); compK=comp[comp_flat]; compC=comp[canon_arr][comp_gene]

def stratum(Aval,sel):
    vc=Aval[canon_arr][comp_gene]; vk=Aval[comp_flat]
    win=(vk>vc)&sel; tie=(vk==vc)&sel
    w=np.bincount(comp_gene,win,NG); t=np.bincount(comp_gene,tie,NG); nn=np.bincount(comp_gene,sel.astype(float),NG)
    half=np.where(nn>0,(w+0.5*t)/nn,np.nan)
    return half
def ga(stat):
    ok=np.isfinite(stat); v=stat[ok]
    if len(v)<5: return (np.nan,np.nan,np.nan,int(ok.sum()))
    b=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)
    return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5),int(ok.sum())

print("="*70); print("MARGINAL within-gene win-rate (all pairs), null 0.500"); print("="*70)
allsel=np.ones(len(comp_flat),bool)
for A,nm in [(u1,"U1@5.25"),(exc,"u1_excess"),(gc,"GC")]:
    hm,lo,hi,ng=ga(stratum(A,allsel))
    print(f"  {nm:<11} {hm:.4f} [{lo:.4f},{hi:.4f}] excess {hm-0.5:+.4f}")

print("\n"+"="*70); print("TERM-TERM (both windows terminal/3'UTR), null 0.500"); print("="*70)
same=compK==compC; termterm=same&(compK==0)
print(f"  informative TERM-TERM pairs: {int(termterm.sum()):,}  ({int((same).sum()):,} same-compartment total)")
for A,nm in [(u1,"U1@5.25"),(exc,"u1_excess"),(gc,"GC")]:
    hm,lo,hi,ng=ga(stratum(A,termterm))
    star="CI excludes 0.5" if (lo>0.5 or hi<0.5) else "CI includes 0.5"
    print(f"  {nm:<11} {hm:.4f} [{lo:.4f},{hi:.4f}] excess {hm-0.5:+.4f}  {star}  ({ng} genes)")

print("\n"+"="*70); print("POOLED same-compartment (all four compartments), null 0.500"); print("="*70)
for A,nm in [(u1,"U1@5.25"),(exc,"u1_excess"),(gc,"GC")]:
    hm,lo,hi,ng=ga(stratum(A,same))
    print(f"  {nm:<11} {hm:.4f} [{lo:.4f},{hi:.4f}] excess {hm-0.5:+.4f}  ({ng} genes)")

print("\n"+"="*70); print("GC covariate numbers"); print("="*70)
can=isc==1; cmp_=isc==0
print(f"  between-class GC diff (comp-canon): {np.nanmean(gc[cmp_])-np.nanmean(gc[can]):+.4f}")
print(f"  compartment GC means: "+" ".join(f"{LAB[j]} {np.nanmean(gc[comp==j]):.3f}" for j in range(4)))
print(f"  compartment U1@5.25 means: "+" ".join(f"{LAB[j]} {np.nanmean(u1[comp==j]):.1f}" for j in range(4)))
print(f"  compartment distribution canonical: "+" ".join(f"{LAB[j]} {(comp[can]==j).mean()*100:.0f}%" for j in range(4)))
print(f"  compartment distribution competitor: "+" ".join(f"{LAB[j]} {(comp[cmp_]==j).mean()*100:.0f}%" for j in range(4)))

print("\n"+"="*70); print("DECOY specificity limit (competitor vs matched intronic decoy)"); print("="*70)
D=np.load("decoy_set.npz"); dgene=D["gene"]; du1=D["u1"]; dexc=D["exc"]
comp_by_gene=defaultdict(list)
for k,g in zip(comp_flat,comp_gene): comp_by_gene[g].append(k)
dec_by_gene=defaultdict(list)
for j,g in enumerate(dgene): dec_by_gene[int(g)].append(j)
for Aname,compA,decA in [("u1_excess",exc,dexc),("U1@5.25",u1,du1)]:
    rates=[]
    for g,ks in comp_by_gene.items():
        ds=dec_by_gene.get(g,[])
        if not ds: continue
        cv=compA[ks]; dv=decA[ds]
        cv=cv[np.isfinite(cv)]; dv=dv[np.isfinite(dv)]
        if not cv.size or not dv.size: continue
        w=(cv[:,None]>dv[None,:]).sum()+0.5*(cv[:,None]==dv[None,:]).sum()
        rates.append(w/(cv.size*dv.size))
    rates=np.asarray(rates)
    print(f"  {Aname:<11} competitor>decoy win-rate {rates.mean():.4f} (excess {rates.mean()-0.5:+.4f}) over {len(rates)} genes")
