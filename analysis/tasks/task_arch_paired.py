#!/usr/bin/env python3
# fixes 2-3: paired top-1/top-3 comparison of terminal-like@8 vs acceptor-only on the same
import sys, csv
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
NP=len(R); max5=np.full(NP,-99.0)
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
    if (i+1)%15000==0: log(f"  max5 {i+1}/{NP}")
log("max5 done")

term8=((acc>=8)&(max5<8)).astype(float)
gp=defaultdict(list)
for i in range(NP): gp[gene[i]].append(i)
GEN=[np.array(gp[g]) for g in gp if len(gp[g])>=2 and any(isc[j]==1 for j in gp[g])]
NG=len(GEN)

def topk_hit(idx,score,tb,k):
    order=np.lexsort((tb[idx],score[idx]))[::-1]
    return 1.0 if (isc[idx[order[:k]]]==1).any() else 0.0
zero=np.zeros(NP)
def hits(rule,k):
    if rule=="acc": return np.array([topk_hit(idx,acc,zero,k) for idx in GEN])
    return np.array([topk_hit(idx,term8,acc,k) for idx in GEN])
def bootdiff(a,b,n=5000):
    d=a-b; rs=np.random.default_rng(1); idx=rs.integers(0,len(d),size=(n,len(d)))
    bs=d[idx].mean(1); return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

print("PAIRED top-1 / top-3: terminal-like@8 vs acceptor-only, same %d genes"%NG)
for k in (1,3):
    ha=hits("acc",k); ht=hits("term",k)
    dm,lo,hi=bootdiff(ht,ha)
    tw=int(((ht==1)&(ha==0)).sum()); aw=int(((ht==0)&(ha==1)).sum()); tie=int((ht==ha).sum())
    print(f"  top-{k}: acceptor-only {ha.mean():.4f} ; terminal@8 {ht.mean():.4f} ; "
          f"paired diff {dm:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"         genes: terminal wins {tw} ; acceptor wins {aw} ; same pick/tie {tie}")

# permutation check of the 0.270 chance rate
chance_analytic=float(np.mean([(isc[idx]==1).mean() for idx in GEN]))
rs=np.random.default_rng(7); NPERM=1000
perm=np.empty(NPERM)
for p in range(NPERM):
    h=0
    for idx in GEN:
        lab=rs.permutation(isc[idx])
        order=np.lexsort((zero[idx],acc[idx]))[::-1]
        h+=1 if lab[order[0]]==1 else 0
    perm[p]=h/len(GEN)
print(f"\nCHANCE top-1: analytic (gene-avg n_canon/n_peaks) = {chance_analytic:.4f}")
print(f"  within-gene label permutation: mean {perm.mean():.4f} [{np.percentile(perm,2.5):.4f},{np.percentile(perm,97.5):.4f}] ({NPERM} perms)")
print("[out] (printed only)")
