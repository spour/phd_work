#!/usr/bin/env python3
# b10 - the compartment confound (blocking)
import sys, csv, re
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; GTF="../train/data/annotation/gencode.v49.annotation.gtf"
META="fig3_ucube_meta.tsv"; WIN=500

# parse gencode: per-transcript exon/utr/cds
TID=re.compile(r'transcript_id "([^"]+)"')
tx=defaultdict(lambda: {"chrom":None,"strand":None,"exon":[],"utr":[],"cds":[]})
n=0
with open(GTF) as fh:
    for line in fh:
        if line[0]=="#": continue
        f=line.split("\t")
        ft=f[2]
        if ft not in ("exon","UTR","CDS"): continue
        m=TID.search(f[8])
        if not m: continue
        t=tx[m.group(1)]; t["chrom"]=f[0]; t["strand"]=f[6]
        iv=(int(f[3])-1,int(f[4]))
        t["exon" if ft=="exon" else ("utr" if ft=="UTR" else "cds")].append(iv)
        n+=1
        if n%2_000_000==0: log(f"  gtf {n:,} feats, {len(tx):,} tx")
log(f"[gtf] {len(tx):,} transcripts parsed")

exon_by=defaultdict(list); txspan_by=defaultdict(list); term_by=defaultdict(list)
for t in tx.values():
    if not t["exon"]: continue
    c=t["chrom"]; st=t["strand"]; key=(c,st)
    ex=t["exon"]
    exon_by[key].extend(ex)
    txspan_by[key].append((min(a for a,_ in ex), max(b for _,b in ex)))
    term_by[key].append(max(ex,key=lambda iv:iv[1]) if st=="+" else min(ex,key=lambda iv:iv[0]))
    if t["cds"] and t["utr"]:
        cmin=min(a for a,_ in t["cds"]); cmax=max(b for _,b in t["cds"])
        for a,b in t["utr"]:
            if (st=="+" and a>=cmax) or (st=="-" and b<=cmin): term_by[key].append((a,b))

def merge(ivs):
    if not ivs: return np.empty(0,np.int64),np.empty(0,np.int64)
    ivs=sorted(ivs); cs=[];ce=[]
    for a,b in ivs:
        if cs and a<=ce[-1]: ce[-1]=max(ce[-1],b)
        else: cs.append(a);ce.append(b)
    return np.array(cs,np.int64),np.array(ce,np.int64)
TERM={k:merge(v) for k,v in term_by.items()}
EXON={k:merge(v) for k,v in exon_by.items()}
TXSPAN={k:merge(v) for k,v in txspan_by.items()}
log("[gtf] disjoint interval sets built")

def ov(a,b,cse):
    cs,ce=cse
    if cs.size==0: return 0.0
    lo=np.searchsorted(ce,a,"right"); hi=np.searchsorted(cs,b,"left")
    if hi<=lo: return 0.0
    s=np.maximum(cs[lo:hi],a); e=np.minimum(ce[lo:hi],b)
    return float(np.maximum(e-s,0).sum())

# classify each peak window
gene=[];isc=[];rows=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        gene.append(r["gene"]); isc.append(int(r["is_canonical"]))
        rows.append((r["chrom"],int(r["center"]),r["strand"]))
NP=len(rows)
LAB=["TERM","INTERNAL","INTRON","OTHER"]
comp=np.full(NP,3,np.int8)
empt=(np.empty(0,np.int64),np.empty(0,np.int64))
for i,(c,ctr,st) in enumerate(rows):
    a,b=(ctr-WIN,ctr) if st=="+" else (ctr,ctr+WIN)
    key=(c,st)
    term=ov(a,b,TERM.get(key,empt)); exon=ov(a,b,EXON.get(key,empt)); tx_=ov(a,b,TXSPAN.get(key,empt))
    internal=max(exon-term,0.0); intron=max(tx_-exon,0.0); inter=max(WIN-tx_,0.0)
    vals=np.array([term,internal,intron,inter]); j=int(vals.argmax())
    comp[i]= j if vals[j]>=0.8*WIN else 3
    if (i+1)%20000==0: log(f"  classify {i+1:,}")
np.save("peak_compartment.npy",comp)

# gc per peak (same upstream window as builder)
Bm=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): Bm[ch]=v
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
G=Faidx(FASTA); gc=np.full(NP,np.nan)
for i,(c,ctr,st) in enumerate(rows):
    seq=G.fetch(c,ctr,ctr+WIN,"-") if st=="-" else G.fetch(c,ctr-WIN,ctr,"+")
    if seq:
        tg=Bm[np.frombuffer(seq.encode("ascii"),np.uint8)]; tg=tg[tg>=0]
        if tg.size: gc[i]=((tg==1)|(tg==2)).mean()

Z=np.load("u1_window_cache.npz"); u1d525=Z["dens_5.25"]; excess525=Z["excess_525"]; gi=Z["gene_idx"]
isc=np.array(isc)

# within-gene structures
order=np.argsort(gi,kind="stable"); gs=gi[order]
cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=order[cut[j]:cut[j+1]]; cc=idx[isc[idx]==1]; kk=idx[isc[idx]==0]
    if len(cc)==1 and len(kk)>=1: gmemb.append(idx);gcanon.append(int(cc[0]))
NG=len(gmemb)
comp_flat=np.concatenate([m[m!=c] for m,c in zip(gmemb,gcanon)])
comp_gene=np.concatenate([np.full((m!=c).sum(),i) for i,(m,c) in enumerate(zip(gmemb,gcanon))])
canon_arr=np.array(gcanon)
compK=comp[comp_flat]; compC=comp[canon_arr][comp_gene]

def stratum(Aval,sel):
    vc=Aval[canon_arr][comp_gene]; vk=Aval[comp_flat]
    win=(vk>vc)&sel; loss=(vk<vc)&sel; tie=(vk==vc)&sel
    w=np.bincount(comp_gene,win,NG); l=np.bincount(comp_gene,loss,NG)
    t=np.bincount(comp_gene,tie,NG); nn=np.bincount(comp_gene,sel.astype(float),NG)
    info=w+l
    half=np.where(nn>0,(w+0.5*t)/nn,np.nan); sign=np.where(info>0,w/np.maximum(info,1),np.nan)
    return half,sign,nn,info
def ga(stat):
    ok=np.isfinite(stat); v=stat[ok]
    if len(v)<5: return (np.nan,np.nan,np.nan,int(ok.sum()))
    b=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)
    return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5),int(ok.sum())

print("\n============  B10.1  compartment distribution (window pA-500..pA)  ============")
can=isc==1; cmp_=isc==0
print(f"{'compartment':<10}{'canonical %':>13}{'competitor %':>14}")
for j,nm in enumerate(LAB):
    print(f"{nm:<10}{(comp[can]==j).mean()*100:>12.1f}%{(comp[cmp_]==j).mean()*100:>13.1f}%")

print("\n============  B10.2  GC and U1@5.25 by compartment (classes pooled)  ============")
print(f"{'compartment':<10}{'n':>8}{'mean GC':>10}{'mean U1@5.25':>14}")
for j,nm in enumerate(LAB):
    s=comp==j
    print(f"{nm:<10}{int(s.sum()):>8,}{np.nanmean(gc[s]):>10.4f}{np.nanmean(u1d525[s]):>14.4f}")
print(f"  overall between-class GC diff (comp-canon): {np.nanmean(gc[cmp_])-np.nanmean(gc[can]):+.4f}")

print("\n============  B10.3  within-gene test restricted to SAME-compartment pairs  ============")
print("  ties=0.5 estimator (exact 0.5 null) + sign test ; excess = obs-0.5 ; competitor>canon => >0.5\n")
def run_stratum(name,sel):
    print(f"  --- {name}: informative pairs {int((sel).sum()):,} ---")
    for Aval,fn in [(gc,"GC"),(u1d525,"U1@5.25"),(excess525,"u1_excess")]:
        half,sign,nn,info=stratum(Aval,sel)
        hm,hlo,hhi,ng=ga(half); sm,slo,shi,_=ga(sign)
        tot=int(info.sum())
        if np.isfinite(hm):
            print(f"    {fn:<10} ties.5 {hm:.4f} [{hlo:.4f},{hhi:.4f}] exc {hm-0.5:+.4f} | "
                  f"sign {sm:.4f} [{slo:.4f},{shi:.4f}] | {ng} genes, {tot:,} info pairs")
        else:
            print(f"    {fn:<10} too few genes ({ng})")
same=compK==compC
for j,nm in enumerate(LAB):
    run_stratum(f"both {nm}", same&(compK==j))
run_stratum("pooled same-compartment", same)

print("\n============  B10.4  compartment-matched conditional win-rate (matched binarisation)  ============")
print("  win-rate among same-compartment pairs, pooled; null 0.500\n")
for Aval,nm in [(u1d525,"U1@5.25 | compartment"),(excess525,"u1_excess | compartment"),(gc,"GC | compartment")]:
    half,sign,nn,info=stratum(Aval,same)
    hm,hlo,hhi,ng=ga(half)
    print(f"  {nm:<26} ties.5 {hm:.4f} [{hlo:.4f},{hhi:.4f}]  excess {hm-0.5:+.4f}  ({ng} genes)")
print("\n  marginal (all pairs) for reference:")
for Aval,nm in [(u1d525,"U1@5.25"),(excess525,"u1_excess"),(gc,"GC")]:
    half,sign,nn,info=stratum(Aval,np.ones(len(comp_flat),bool))
    hm,hlo,hhi,ng=ga(half)
    print(f"  {nm:<26} ties.5 {hm:.4f} [{hlo:.4f},{hhi:.4f}]  excess {hm-0.5:+.4f}")
print("\n[out] peak_compartment.npy")
