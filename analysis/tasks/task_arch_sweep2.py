#!/usr/bin/env python3
# fix 1: extend the donor-absence sweep past 8
import sys, csv, math
import numpy as np
from collections import defaultdict
sys.path.insert(0,"scripts")
from extract_splicing_features import load_me5_lookup
def log(*a): print(*a,file=sys.stderr,flush=True)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
POS="fig3_new_positional_features.tsv"; PAD=12; CAP=30000

Bm=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): Bm[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
me5d=load_me5_lookup(ME5NPY)
ks=np.frombuffer("".join(me5d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
ME5=np.full(4**9,-99.0); ME5[Bm[ks]@POW]=np.fromiter(me5d.values(),np.float64,len(me5d))
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
NP=len(R)
max5=np.full(NP,-99.0)
for i in range(NP):
    if not (np.isfinite(apos[i]) and np.isfinite(dpa[i]) and dpa[i]>=9): continue
    ce=center[i]; st=strand[i]; d=int(min(dpa[i],CAP))
    a,b=(ce-d-PAD,ce+PAD) if st=="+" else (ce-PAD,ce+d+PAD)
    seq=G.fetch(chrom[i],a,b,st)
    if not seq or len(seq)<9: continue
    tok=Bm[np.frombuffer(seq.encode("ascii"),np.uint8)]
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    if not good.any(): continue
    sc=ME5[sw[good]@POW]; idx=np.where(good)[0]+3-PAD; m=(idx>=0)&(idx<d)
    max5[i]=float(sc[m].max()) if m.any() else float(sc.max())
    if (i+1)%15000==0: log(f"  {i+1}/{NP}")
log("max5 done")

strong=acc>=8
gp=defaultdict(list)
for i in range(NP): gp[gene[i]].append(i)
GEN=[g for g,idx in gp.items() if len(idx)>=2 and any(isc[j]==1 for j in idx)]
GIDX={g:np.array(gp[g]) for g in GEN}
chance=float(np.mean([(isc[GIDX[g]]==1).mean() for g in GEN]))
def wilson(k,n,z=1.959963984540054):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
def top1(score,tb):
    hits=[]
    for g in GEN:
        idx=GIDX[g]; order=np.lexsort((tb[idx],score[idx])); hits.append(1.0 if isc[idx[order[-1]]]==1 else 0.0)
    return np.array(hits)
def boot(h,n=2000):
    rs=np.random.default_rng(1); b=h[rs.integers(0,len(h),size=(n,len(h)))].mean(1)
    return h.mean(),np.percentile(b,2.5),np.percentile(b,97.5)

h_c=top1(strong.astype(float),acc); acc_only=boot(h_c)
CUTS=[2,3,4.13,5,6,8,9,10,11,12,14,16]
rows=[]
for c in CUTS:
    tt=strong&(max5<c); n=int(tt.sum()); k=int((tt&(isc==1)).sum())
    p=k/n if n else float("nan"); lo,hi=wilson(k,n)
    m,l2,h2=boot(top1(tt.astype(float),acc))
    rows.append((c,n,p,lo,hi,m,l2,h2))
with open("arch_absence_sweep.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t")
    w.writerow(["donor_absence_cutoff","n_terminal","p_real","ci_lo","ci_hi","top1_acc","top1_lo","top1_hi"])
    for c,n,p,lo,hi,m,l2,h2 in rows:
        w.writerow([f"{c}",n,f"{p:.5f}",f"{lo:.5f}",f"{hi:.5f}",f"{m:.5f}",f"{l2:.5f}",f"{h2:.5f}"])
    w.writerow([]); w.writerow([f"# chance_top1={chance:.5f}",f"acceptor_only_acc>=8_top1={acc_only[0]:.5f}"])
best=max(rows,key=lambda r:r[5])
log(f"chance top-1 (gene-avg 1/n_peaks) = {chance:.4f} ; acceptor-only (acc>=8) top-1 = {acc_only[0]:.4f}")
log("EXTENDED ABSENCE SWEEP:")
for c,n,p,lo,hi,m,l2,h2 in rows:
    mark=" <- top-1 peak" if c==best[0] else ""
    log(f"  cutoff {c:<5} n={n:6d} P(real)={p:.4f} [{lo:.4f},{hi:.4f}]  top-1={m:.4f} [{l2:.4f},{h2:.4f}]{mark}")
print(f"[out] arch_absence_sweep.tsv ; chance={chance:.4f} ; top-1 peak at cutoff {best[0]} (top-1 {best[5]:.4f}, P(real) {best[2]:.4f})")
