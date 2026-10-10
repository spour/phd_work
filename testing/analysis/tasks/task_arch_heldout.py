#!/usr/bin/env python3
# held-out validation of the donor-absence cutoff choice
import sys, csv
import numpy as np
from collections import defaultdict
sys.path.insert(0,"scripts")
from extract_splicing_features import load_me5_lookup
def log(*a): print(*a,file=sys.stderr,flush=True)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
POS="fig3_new_positional_features.tsv"; PAD=12; CAP=30000
CUTS=[4,5,6,7,8,9,10,11,12]

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

gp=defaultdict(list)
for i in range(NP): gp[gene[i]].append(i)
GEN=[np.array(gp[g]) for g in gp if len(gp[g])>=2 and any(isc[j]==1 for j in gp[g])]
NG=len(GEN); zero=np.zeros(NP); strong=acc>=8
def top1(idx,score,tb):
    return 1.0 if isc[idx[np.lexsort((tb[idx],score[idx]))[-1]]]==1 else 0.0
acc_hit=np.array([top1(idx,acc,zero) for idx in GEN])
term_hit={c:np.array([top1(idx,((acc>=8)&(max5<c)).astype(float),acc) for idx in GEN]) for c in CUTS}
log("precomputed hits")

# full-data reference + flip decomposition (at cutoff 8)
ht8=term_hit[8]; d8=ht8-acc_hit
flips=int((d8!=0).sum()); tw=int((d8>0).sum()); aw=int((d8<0).sum())
full_gain=d8.mean()
chance=float(np.mean([(isc[idx]==1).mean() for idx in GEN]))
print(f"FULL-DATA (cutoff 8): paired gain {full_gain:+.4f} ; flips {flips}/{NG} ({100*flips/NG:.1f}%) ; "
      f"terminal correct in {tw}/{flips} ({100*tw/flips:.1f}%)")
print(f"  decomposition above chance {chance:.4f}: acceptor +{acc_hit.mean()-chance:.4f} ; donor +{full_gain:+.4f}")

# 20 held-out splits
gains=[]; chosen=[]
for seed in range(20):
    rs=np.random.default_rng(1000+seed); perm=rs.permutation(NG); half=NG//2
    tr=perm[:half]; te=perm[half:]
    best_c=None; best_g=-1e9
    for c in CUTS:
        g=(term_hit[c][tr]-acc_hit[tr]).mean()
        if g>best_g: best_g=g; best_c=c
    te_gain=(term_hit[best_c][te]-acc_hit[te]).mean()
    gains.append(te_gain); chosen.append(best_c)
gains=np.array(gains)
print(f"\nHELD-OUT (20 splits): mean test paired gain {gains.mean():+.4f} "
      f"[{gains.mean()-gains.std():+.4f},{gains.mean()+gains.std():+.4f}] (+/-1 sd) ; "
      f"min {gains.min():+.4f} max {gains.max():+.4f}")
from collections import Counter
print(f"  cutoff chosen on train halves: {dict(sorted(Counter(chosen).items()))}")
print(f"  full-data gain {full_gain:+.4f} vs held-out mean {gains.mean():+.4f} -> "
      f"{'bias negligible' if abs(gains.mean()-full_gain)<0.004 else 'shrinks materially'}")
