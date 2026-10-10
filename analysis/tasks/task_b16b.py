#!/usr/bin/env python3
# b16b - firm the negative further: (a) repeat-matched sensitivity
import sys, csv, re
import numpy as np
from collections import defaultdict
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(7)
FASTA="../train/data/hg38.fa"; GTF="../train/data/annotation/gencode.v49.annotation.gtf"
ATLAS="../train/data/atlas.chr.bed"; META="fig3_ucube_meta.tsv"; U1NPY="../train/splice_train/data/u1_lookup.npy"
WIN=500; STEP=25; KDEC=3; PRIMARY=5.25; MAXCAND=600; DPA_CAP=500; NSUB=1000

B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
a=np.load(U1NPY,allow_pickle=True); d=a.item() if a.dtype==object else dict(a)
kb=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9); ktok=B[kb]; kidx=ktok@POW
SCORE=np.empty(4**9); SCORE[kidx]=-np.fromiter(d.values(),np.float64,len(d))
o=np.argsort(kidx); tbc=ktok[o]; init=np.zeros((4**9,4)); init[np.arange(4**9),tbc[:,0]]=1.0
steps=tbc[:,:-1]*4+tbc[:,1:]; dcn=np.zeros((4**9,16))
for i in range(16): dcn[:,i]=(steps==i).sum(1)
FEAT=np.hstack([init,dcn]); Ht=np.where(SCORE>=PRIMARY)[0]; Fh=FEAT[Ht]
def swscore(tok):
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    out=np.full(sw.shape[0],np.nan); out[good]=SCORE[sw[good]@POW]; return out
class Faidx:
    _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
    def __init__(s,fa):
        s.fh=open(fa,"rb"); s.idx={}
        for L in open(fa+".fai"):
            p=L.split(); s.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
    def fetch(s,c,a,b,st,up=True):
        c=c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln,off,lb,lw=s.idx[c]; a=max(0,a); b=min(b,Ln)
        if b<=a: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a]
        return q
G=Faidx(FASTA)
def u1exc_and_low(pos,ch,st):
    if st=="-": rb=G.fetch(ch,pos,pos+WIN,st)
    else: rb=G.fetch(ch,pos-WIN,pos,st)
    if not rb: return (np.nan,np.nan,np.nan)
    arr=np.frombuffer(rb,np.uint8); low=float(((arr>=97)&(arr<=122)).mean())
    up=arr.copy()
    s=bytes(arr).decode("ascii").upper()
    if st=="-": s=s.translate(Faidx._T)[::-1]
    tok=B[np.frombuffer(s.encode("ascii"),np.uint8)]
    sc=swscore(tok); valid=np.isfinite(sc); npos=int(valid.sum()); v=sc[valid]
    if npos==0: return (np.nan,np.nan,low)
    cnt=float((v>=PRIMARY).sum()); dens=cnt/(WIN/1000.0)
    tg=tok[tok>=0]; exc=np.nan
    if tg.size>=10:
        bc=np.bincount(tg,minlength=4).astype(float); stc=np.bincount(tg[:-1]*4+tg[1:],minlength=16).astype(float).reshape(4,4)
        p0=(bc+0.5)/(bc.sum()+2.0); T=(stc+0.25)/(stc.sum(1,keepdims=True)+1.0)
        Wl=np.concatenate([np.log(p0),np.log(T).ravel()]); q=float(np.exp(Fh@Wl).sum()); exc=(cnt-npos*q)/(WIN/1000.0)
    return (dens,exc,low)

# peaks + within-gene
Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; gi=Z["gene_idx"]; u1d525=Z["dens_5.25"]; excess525=Z["excess_525"]
rows=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["gene"],r["chrom"],int(r["center"]),r["strand"]))
gname=np.array([x[0] for x in rows]); chrom=np.array([x[1] for x in rows]); center=np.array([x[2] for x in rows]); strand=np.array([x[3] for x in rows])
o2=np.argsort(gi,kind="stable"); gs=gi[o2]; cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=o2[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1: gmemb.append(idx);gcanon.append(int(c[0]))
NG=len(gmemb)

# (a) repeat-matched sensitivity on existing decoy_set
D=np.load("decoy_set.npz"); dpos=D["pos"]; dgene=D["gene"]; dcomp=D["comp"]; dexc=D["exc"]; du1=D["u1"]
log("rescoring low-complexity at existing decoys + competitors...")
dlow=np.empty(len(dpos))
for i in range(len(dpos)):
    g=dgene[i]; ch=chrom[gcanon[g]]; st=strand[gcanon[g]]
    _,_,dlow[i]=u1exc_and_low(int(dpos[i]),ch,st)
compset=np.unique(dcomp); clow={}
for c in compset:
    _,_,clow[c]=u1exc_and_low(int(center[c]),chrom[c],strand[c])
def wr_matched(realval,decval,maxdlow):
    half=np.full(NG,np.nan)
    for g in range(NG):
        m=dgene==g
        if not m.any(): continue
        dv=decval[m]; cm=dcomp[m]; dl=dlow[m]
        keep=np.isfinite(dv)
        if maxdlow is not None:
            keep=keep & np.array([abs(dl[j]-clow[cm[j]])<=maxdlow for j in range(len(cm))])
        if keep.sum()==0: continue
        cv=realval[cm[keep]]; dd=cv-dv[keep]; nz=dd!=0
        half[g]=((dd>0).sum()+0.5*(dd==0).sum())/keep.sum()
    ok=np.isfinite(half); v=half[ok]; b=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)
    return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5),int(ok.sum())
print("\n============  B16b(a)  repeat-matched sensitivity (B13.2 competitor vs decoy)  ============")
print(f"  competitor mean low {np.nanmean([clow[c] for c in compset]):.3f} vs decoy mean low {np.nanmean(dlow):.3f}")
for nm,rv,dv in [("u1_excess",excess525,dexc),("U1@5.25",u1d525,du1)]:
    m0=wr_matched(rv,dv,None); m1=wr_matched(rv,dv,0.10)
    print(f"  {nm:<10} all-decoys wr {m0[0]:.4f} [{m0[1]:.4f},{m0[2]:.4f}] ({m0[3]}g) | |dlow|<0.10 wr {m1[0]:.4f} [{m1[1]:.4f},{m1[2]:.4f}] ({m1[3]}g)")

# (b) exclusion-radius stability (subsample redraw)
TID=re.compile(r'transcript_id "([^"]+)"'); GNM=re.compile(r'gene_name "([^"]+)"')
tx=defaultdict(lambda:{"g":None,"st":None,"exon":[]}); n=0
with open(GTF) as fh:
    for line in fh:
        if line[0]=="#": continue
        f=line.split("\t")
        if f[2]!="exon": continue
        m=TID.search(f[8]); gm=GNM.search(f[8])
        if not m or not gm: continue
        t=tx[m.group(1)]; t["g"]=gm.group(1); t["st"]=f[6]; t["exon"].append((int(f[3])-1,int(f[4])))
        n+=1
def merge(ivs):
    ivs=sorted(ivs); cs=[];ce=[]
    for a,b in ivs:
        if cs and a<=ce[-1]: ce[-1]=max(ce[-1],b)
        else: cs.append(a);ce.append(b)
    return list(zip(cs,ce))
ge=defaultdict(list); gstr={}
for t in tx.values():
    if t["exon"]: ge[t["g"]].extend(t["exon"]); gstr[t["g"]]=t["st"]
g_int={}; g_acc={}; g_don={}
for g,ex in ge.items():
    mex=merge(ex); st=gstr[g]; intr=[(mex[i][1],mex[i+1][0]) for i in range(len(mex)-1) if mex[i+1][0]>mex[i][1]]
    g_int[g]=intr; g_acc[g]=np.array([b for _,b in intr] if st=="+" else [a for a,_ in intr],np.int64)
    g_don[g]=np.array([a for a,_ in intr] if st=="+" else [b for _,b in intr],np.int64)
atl=defaultdict(list)
with open(ATLAS) as fh:
    for line in fh:
        p=line.split("\t")
        if len(p)<6: continue
        atl[(p[0],p[5])].append((int(p[1])+int(p[2]))//2)
ATL={k:np.array(sorted(v),np.int64) for k,v in atl.items()}
def near(pos,ch,st,rad):
    arr=ATL.get((ch,st))
    if arr is None: return False
    j=np.searchsorted(arr,pos)
    for jj in (j-1,j):
        if 0<=jj<arr.size and abs(int(arr[jj])-pos)<=rad: return True
    return False
def nn(p,arr): return float(np.abs(arr-p).min()) if arr.size else 1e9

sub=rng.choice(NG,min(NSUB,NG),replace=False)
print(f"\n============  B16b(b)  exclusion-radius stability (B13.2, {len(sub)} gene subsample)  ============")
for RAD in (1000,2000,5000):
    half_e=np.full(NG,np.nan); half_u=np.full(NG,np.nan)
    for g in sub:
        idx=gmemb[g]; c=gcanon[g]; comps=idx[idx!=c]; gnm=gname[c]; ch=chrom[c]; st=strand[c]; cpA=center[c]
        if gnm not in g_int: continue
        acc=g_acc[gnm]; don=g_don[gnm]
        cand=[p for (a,b) in g_int[gnm] for p in range(a+1,b,STEP) if not near(p,ch,st,RAD)]
        if not cand: continue
        cand=np.array(cand,np.int64)
        if cand.size>MAXCAND: cand=cand[rng.choice(cand.size,MAXCAND,replace=False)]
        cdpa=np.abs(cand-cpA).astype(float); cdac=np.array([nn(p,acc) for p in cand]); cddo=np.array([nn(p,don) for p in cand])
        spa=cdpa.std()+1e-9; sac=cdac.std()+1e-9; sdo=cddo.std()+1e-9; used=np.zeros(cand.size,bool)
        ce=[]; cu=[]; cc=[]
        for comp in comps:
            cp=center[comp]; qpa=abs(cp-cpA); qac=nn(cp,acc); qdo=nn(cp,don)
            dist=((cdpa-qpa)/spa)**2+2*((cdac-qac)/sac)**2+2*((cddo-qdo)/sdo)**2
            dist[np.abs(cdpa-qpa)>DPA_CAP]=np.inf; dist[used]=np.inf
            take=np.argsort(dist)[:KDEC]; take=take[np.isfinite(dist[take])]
            if take.size==0: continue
            used[take]=True
            for ti in take:
                de,dxe,_=(None,None,None)
                dens,exc,_=u1exc_and_low(int(cand[ti]),ch,st); ce.append(exc); cu.append(dens); cc.append(comp)
        if not cc: continue
        ce=np.array(ce); cu=np.array(cu); cc=np.array(cc)
        for arr,store in [(ce,half_e),(cu,half_u)]:
            ok=np.isfinite(arr)
            if ok.sum()==0: continue
            realv=excess525 if store is half_e else u1d525
            dd=realv[cc[ok]]-arr[ok]; nz=dd!=0
            store[g]=((dd>0).sum()+0.5*(dd==0).sum())/ok.sum()
    def rep(h):
        ok=np.isfinite(h); v=h[ok]; b=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)
        return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5),int(ok.sum())
    e=rep(half_e); u=rep(half_u)
    print(f"  excl {RAD:>4}nt : u1_excess wr {e[0]:.4f} [{e[1]:.4f},{e[2]:.4f}] | U1@5.25 wr {u[0]:.4f} [{u[1]:.4f},{u[2]:.4f}]  ({e[3]}g)")
print("\n[done]")
