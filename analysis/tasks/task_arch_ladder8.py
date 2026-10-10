#!/usr/bin/env python3
# regenerate arch_ladder.tsv with terminal-like at the primary >=8 donor-absence cutoff
import sys, csv, math
import numpy as np
from collections import defaultdict
sys.path.insert(0,"scripts")
from extract_splicing_features import init_maxentscan, load_me5_lookup
def log(*a): print(*a,file=sys.stderr,flush=True)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
POS="fig3_new_positional_features.tsv"; PAD=12; CAP=30000; UP2K=2000; BASE=0.09308
s5,s3,m5,m3=init_maxentscan()
Bm=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): Bm[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
me5d=load_me5_lookup(ME5NPY)
ks=np.frombuffer("".join(me5d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
ME5=np.full(4**9,-99.0); ME5[Bm[ks]@POW]=np.fromiter(me5d.values(),np.float64,len(me5d))
def sc3(seq):
    if not seq or len(seq)!=23 or "N" in seq: return -99.0
    v=s3(seq); return float(np.nanmax(np.asarray(v,float))) if hasattr(v,"__len__") else float(v)
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
R=list(csv.DictReader(open(POS),delimiter="\t"))
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
chrom=[r["chrom"] for r in R]; center=[int(r["center"]) for r in R]; strand=[r["strand"] for r in R]
acc=np.array([float(r["acc"]) if r["acc"] not in("","nan") else np.nan for r in R])
apos=np.array([float(r["anchor_3ss_pos"]) if r["anchor_3ss_pos"] not in("","nan") else np.nan for r in R])
dpa=np.array([float(r["d_anchor_to_pa"]) if r["d_anchor_to_pa"] not in("","nan") else np.nan for r in R])
NP=len(R); max5=np.full(NP,-99.0); max3=np.full(NP,-99.0)
for i in range(NP):
    ce=center[i]; st=strand[i]; ch=chrom[i]
    if np.isfinite(apos[i]) and np.isfinite(dpa[i]) and dpa[i]>=9:
        d=int(min(dpa[i],CAP)); a,b=(ce-d-PAD,ce+PAD) if st=="+" else (ce-PAD,ce+d+PAD)
        seq=G.fetch(ch,a,b,st)
        if seq and len(seq)>=9:
            tok=Bm[np.frombuffer(seq.encode("ascii"),np.uint8)]
            sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
            if good.any():
                sc=ME5[sw[good]@POW]; idx=np.where(good)[0]+3-PAD; m=(idx>=0)&(idx<d)
                max5[i]=float(sc[m].max()) if m.any() else float(sc.max())
    a,b=(ce-UP2K-3,ce+3) if st=="+" else (ce-3,ce+UP2K+3)
    seq=G.fetch(ch,a,b,st)
    if seq and len(seq)>=23:
        arr=np.frombuffer(seq.encode("ascii"),np.uint8); ag=np.where((arr[:-1]==65)&(arr[1:]==71))[0]
        best=-99.0
        for p in ag:
            s0=p-18
            if s0<0 or s0+23>len(seq): continue
            v=sc3(seq[s0:s0+23]); best=v if v>best else best
        max3[i]=best
    if (i+1)%10000==0: log(f"  scan {i+1}/{NP}")
log("scan done")
strong=acc>=8; term8=strong&(max5<8); top3=acc>=(max3-1e-6)
def wilson(k,n,z=1.959963984540054):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
gp=defaultdict(list)
for i in range(NP): gp[gene[i]].append(i)
GEN=[g for g,idx in gp.items() if len(idx)>=2 and any(isc[j]==1 for j in idx)]
GIDX={g:np.array(gp[g]) for g in GEN}
chance=float(np.mean([(isc[GIDX[g]]==1).mean() for g in GEN]))
zero=np.zeros(NP)
def top1(score,tb):
    return np.array([1.0 if isc[GIDX[g][np.lexsort((tb[GIDX[g]],score[GIDX[g]]))[-1]]]==1 else 0.0 for g in GEN])
def boot(h,n=2000):
    rs=np.random.default_rng(1); b=h[rs.integers(0,len(h),size=(n,len(h)))].mean(1); return h.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
def pooled(mask):
    n=int(mask.sum()); k=int((mask&(isc==1)).sum()); p=k/n if n else float("nan"); lo,hi=wilson(k,n); return n,p,lo,hi
dec=np.clip(np.digitize(acc,np.nanquantile(acc,np.linspace(0,1,11)[1:-1])),0,9)
ptop=pooled(dec==9); pbot=pooled(dec==0)
rows=[]
rows.append(("a. base rate",BASE,0,0,0,chance,chance,chance,"random (chance)"))
mb=boot(top1(acc,zero)); rows.append(("b. acceptor decile",ptop[1],ptop[0],pbot[1],pbot[0],*mb,f"top/bot ratio {ptop[1]/max(pbot[1],1e-9):.2f}"))
pc=pooled(strong); mc=boot(top1(strong.astype(float),acc)); rows.append(("c. acceptor>=8",pc[1],pc[0],pc[2],pc[3],*mc,"acceptor-only"))
pd_=pooled(term8); md=boot(top1(term8.astype(float),acc)); rows.append(("d. terminal-like @8 (rule1)",pd_[1],pd_[0],pd_[2],pd_[3],*md,"acc>=8 & no 5'ss>=8"))
te=term8&top3; pe=pooled(te); me_=boot(top1(te.astype(float),acc)); rows.append(("e. + anchor is top 3'ss/2kb",pe[1],pe[0],pe[2],pe[3],*me_,f"top3ss frac {top3.mean():.3f}"))
with open("arch_ladder.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t")
    w.writerow(["rule","p_real_or_topdecile","n_or_topn","p_bot_or_x","n_bot_or_x","top1_acc","top1_lo","top1_hi","note"])
    for r in rows: w.writerow([r[0],f"{r[1]:.5f}",r[2],f"{r[3]:.5f}",r[4],f"{r[5]:.5f}",f"{r[6]:.5f}",f"{r[7]:.5f}",r[8]])
log(f"ladder rebuilt at >=8 (chance {chance:.4f}):")
for r in rows: log(f"  {r[0]:32s} top-1={r[5]:.4f} [{r[6]:.4f},{r[7]:.4f}]  {r[8]}")
print("[out] arch_ladder.tsv (terminal-like at >=8)")
