#!/usr/bin/env python3
# b15 - numerical hygiene reissue of b10, clean
import sys, csv, re
import numpy as np
from collections import defaultdict
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(42)
GTF="../train/data/annotation/gencode.v49.annotation.gtf"; META="fig3_ucube_meta.tsv"; WIN=500

TID=re.compile(r'transcript_id "([^"]+)"')
tx=defaultdict(lambda:{"chrom":None,"strand":None,"exon":[],"utr":[],"cds":[]}); n=0
with open(GTF) as fh:
    for line in fh:
        if line[0]=="#": continue
        f=line.split("\t"); ft=f[2]
        if ft not in ("exon","UTR","CDS"): continue
        m=TID.search(f[8])
        if not m: continue
        t=tx[m.group(1)]; t["chrom"]=f[0]; t["strand"]=f[6]
        iv=(int(f[3])-1,int(f[4]))
        t["exon" if ft=="exon" else ("utr" if ft=="UTR" else "cds")].append(iv)
        n+=1
        if n%4_000_000==0: log(f"  gtf {n:,}")
exon_by=defaultdict(list); tx_by=defaultdict(list); term_by=defaultdict(list)
for t in tx.values():
    if not t["exon"]: continue
    st=t["strand"]; key=(t["chrom"],st); ex=t["exon"]
    exon_by[key].extend(ex); tx_by[key].append((min(a for a,_ in ex),max(b for _,b in ex)))
    term_by[key].append(max(ex,key=lambda iv:iv[1]) if st=="+" else min(ex,key=lambda iv:iv[0]))
    if t["cds"] and t["utr"]:
        cmin=min(a for a,_ in t["cds"]); cmax=max(b for _,b in t["cds"])
        for a,b in t["utr"]:
            if (st=="+" and a>=cmax) or (st=="-" and b<=cmin): term_by[key].append((a,b))
def merge(ivs):
    if not ivs: return (np.empty(0,np.int64),np.empty(0,np.int64))
    ivs=sorted(ivs); cs=[];ce=[]
    for a,b in ivs:
        if cs and a<=ce[-1]: ce[-1]=max(ce[-1],b)
        else: cs.append(a);ce.append(b)
    return np.array(cs,np.int64),np.array(ce,np.int64)
TERM={k:merge(v) for k,v in term_by.items()}; EXON={k:merge(v) for k,v in exon_by.items()}; TXS={k:merge(v) for k,v in tx_by.items()}
def ov(a,b,cse):
    cs,ce=cse
    if cs.size==0: return 0.0
    lo=np.searchsorted(ce,a,"right"); hi=np.searchsorted(cs,b,"left")
    if hi<=lo: return 0.0
    s=np.maximum(cs[lo:hi],a); e=np.minimum(ce[lo:hi],b); return float(np.maximum(e-s,0).sum())

rows=[];isc=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["chrom"],int(r["center"]),r["strand"])); isc.append(int(r["is_canonical"]))
isc=np.array(isc); NP=len(rows)
LAB=["TERM","INTERNAL","INTRON","OTHER"]
comp=np.full(NP,3,np.int8); other_sub=np.zeros(NP,np.int8)
empt=(np.empty(0,np.int64),np.empty(0,np.int64))
for i,(c,ctr,st) in enumerate(rows):
    a,b=(ctr-WIN,ctr) if st=="+" else (ctr,ctr+WIN); key=(c,st)
    term=ov(a,b,TERM.get(key,empt)); exon=ov(a,b,EXON.get(key,empt)); txv=ov(a,b,TXS.get(key,empt))
    internal=max(exon-term,0.0); intron=max(txv-exon,0.0); inter=max(WIN-txv,0.0)
    vals=np.array([term,internal,intron,inter]); j=int(vals.argmax())
    if vals[j]>=0.8*WIN: comp[i]=j
    else:
        comp[i]=3
        other_sub[i]= 0 if inter>=0.5*WIN else (1 if (exon>0 and intron>0) else 2)
np.save("peak_compartment.npy",comp)

Z=np.load("u1_window_cache.npz"); u1d525=Z["dens_5.25"]; excess525=Z["excess_525"]; gi=Z["gene_idx"]
gc=np.load("gc_up500.npy")
o=np.argsort(gi,kind="stable"); gs=gi[o]; cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=o[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1: gmemb.append(idx);gcanon.append(int(c[0]))
NG=len(gmemb); canon_arr=np.array(gcanon)
comp_flat=np.concatenate([m[m!=c] for m,c in zip(gmemb,gcanon)])
comp_gene=np.concatenate([np.full((m!=c).sum(),i) for i,(m,c) in enumerate(zip(gmemb,gcanon))])
compK=comp[comp_flat]; compC=comp[canon_arr][comp_gene]

def stat(Aval,sel):
    vc=Aval[canon_arr][comp_gene]; vk=Aval[comp_flat]
    win=(vk>vc)&sel; loss=(vk<vc)&sel; tie=(vk==vc)&sel
    w=np.bincount(comp_gene,win,NG); l=np.bincount(comp_gene,loss,NG); t=np.bincount(comp_gene,tie,NG); nn=np.bincount(comp_gene,sel.astype(float),NG)
    info=w+l; half=np.where(nn>0,(w+0.5*t)/nn,np.nan); sign=np.where(info>0,w/np.maximum(info,1),np.nan)
    return half,sign,int(info.sum())
def ga(x):
    ok=np.isfinite(x); v=x[ok]
    if len(v)<5: return (np.nan,np.nan,np.nan,int(ok.sum()))
    b=v[rng.integers(0,len(v),size=(4000,len(v)))].mean(1)
    return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5),int(ok.sum())

FEATS=[("GC",gc),("U1@5.25",u1d525),("u1_excess",excess525)]
same=compK==compC
print("\n============  B15  reissued B10 within-gene, ties=0.5 estimator, null=0.500  ============")
print("  win-rate | CI(win-rate) | excess=wr-0.500 | CI(excess) | sign-test | genes | info-pairs\n")
def emit(tag,sel):
    print(f"  --- {tag} ---")
    for nm,A in FEATS:
        half,sign,ip=stat(A,sel); hm,hlo,hhi,ng=ga(half); sm,slo,shi,_=ga(sign)
        if np.isfinite(hm):
            print(f"    {nm:<10} wr {hm:.4f} [{hlo:.4f},{hhi:.4f}] | exc {hm-0.5:+.4f} [{hlo-0.5:+.4f},{hhi-0.5:+.4f}] | sign {sm:.4f} | {ng}g {ip:,}p")
        else:
            print(f"    {nm:<10} too few genes ({ng})")
emit("marginal (all pairs)",np.ones(len(comp_flat),bool))
for j,nm in enumerate(LAB): emit(f"both {nm}",same&(compK==j))
emit("pooled same-compartment (incl OTHER)",same)
emit("pooled same-compartment (excl OTHER)",same&(compK!=3))

print("\n============  B15.3  INTRON stratum vs pooled (does its CI exclude the pooled estimate?)  ============")
for nm,A in FEATS:
    h_i,_,_=stat(A,same&(compK==2)); hi_m,hi_lo,hi_hi,_=ga(h_i)
    h_p,_,_=stat(A,same); hp_m,_,_,_=ga(h_p)
    incl = hi_lo<=hp_m<=hi_hi
    print(f"  {nm:<10} INTRON wr {hi_m:.4f} [{hi_lo:.4f},{hi_hi:.4f}] ; pooled {hp_m:.4f} -> pooled {'INSIDE' if incl else 'OUTSIDE'} INTRON CI")

print("\n============  B15.4  OTHER compartment breakdown  ============")
oth=comp==3
print(f"  OTHER: {oth.mean()*100:.1f}% of peaks ({int(oth.sum()):,}). Canonical {(comp[isc==1]==3).mean()*100:.1f}% / competitor {(comp[isc==0]==3).mean()*100:.1f}%")
for s,nm in [(0,"mostly intergenic (>=50% beyond any transcript)"),(1,"exon/intron boundary-spanning"),(2,"no compartment >=80%, other")]:
    m=oth&(other_sub==s); print(f"    {nm:<42} {m.sum()/max(oth.sum(),1)*100:5.1f}%  ({int(m.sum()):,})")

print("\n============  B15.5  INTERNAL stratum  ============")
ii=same&(compK==1)
print(f"  both-INTERNAL informative pairs: {int(ii.sum())} (canonical INTERNAL is ~0% -> no usable pairs, confirmed)")
print("\n[done]")
