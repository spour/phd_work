#!/usr/bin/env python3
# b3.3: sequence diversity of score>=8.0 u1 hits in the pa-500..pa windows
import sys, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
FASTA="../train/data/hg38.fa"; U1NPY="../train/splice_train/data/u1_lookup.npy"
META="fig3_ucube_meta.tsv"; WIN=500; THR=8.0

B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
arr=np.load(U1NPY,allow_pickle=True); d=arr.item() if arr.dtype==object else dict(arr)
kb=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
kidx=B[kb]@POW; SCORE=np.empty(4**9,np.float64); SCORE[kidx]=-np.fromiter(d.values(),np.float64,len(d))
STRONG=SCORE>=THR
NUC=np.array(list("ACGT"))
def decode(code):
    b=[];
    for _ in range(9): b.append(code&3); code>>=2
    return "".join(NUC[x] for x in b[::-1])

class Faidx:
    _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
    def __init__(self,fa):
        self.fh=open(fa,"rb"); self.idx={}
        for line in open(fa+".fai"):
            p=line.split(); self.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
    def fetch(self,c,s,e,st="+"):
        c=c if c in self.idx else c.lstrip("chr")
        if c not in self.idx: return None
        L,off,lb,lw=self.idx[c]; s=max(0,s); e=min(e,L)
        if e<=s: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb)
        self.fh.seek(bp(s)); raw=self.fh.read(bp(e)-bp(s))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:e-s].decode("ascii").upper()
        return q.translate(self._T)[::-1] if st=="-" else q
G=Faidx(FASTA)

rows=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["chrom"],int(r["center"]),r["strand"]))
log(f"[meta] {len(rows):,} peaks ; counting >= {THR} 9-mers")

count=np.zeros(4**9,np.int64)
for i,(c,ctr,st) in enumerate(rows):
    seq=G.fetch(c,ctr,ctr+WIN,"-") if st=="-" else G.fetch(c,ctr-WIN,ctr,"+")
    if not seq: continue
    tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
    if tok.shape[0]<9: continue
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    codes=sw[good]@POW; codes=codes[STRONG[codes]]
    if codes.size: count+=np.bincount(codes,minlength=4**9)
    if (i+1)%20000==0: log(f"  {i+1:,}")

nz=np.where(count>0)[0]; cnt=count[nz]; o=np.argsort(cnt)[::-1]
nz=nz[o]; cnt=cnt[o]; total=cnt.sum()
cum=np.cumsum(cnt)/total
n50=int(np.searchsorted(cum,0.50)+1); n90=int(np.searchsorted(cum,0.90)+1)
print(f"\n=== B3.3  sequence diversity of >= {THR} hits (pA-500..pA) ===")
print(f"total >= {THR} hits {total:,} ; distinct 9-mers {len(nz):,}")
print(f"9-mers for 50% of hits: {n50}    for 90%: {n90}\n")
print(f"{'rank':>4} {'9-mer':>10} {'count':>10} {'share':>8} {'cum':>8}")
for i in range(min(10,len(nz))):
    print(f"{i+1:>4} {decode(int(nz[i])):>10} {cnt[i]:>10,} {cnt[i]/total:>8.4f} {cum[i]:>8.4f}")
with open("u1_strong_9mer_diversity.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["rank","kmer","count","share","cum_share"])
    for i in range(len(nz)):
        w.writerow([i+1,decode(int(nz[i])),int(cnt[i]),f"{cnt[i]/total:.6f}",f"{cum[i]:.6f}"])
print(f"\n[summary] top10 cumulative share {cum[min(9,len(nz)-1)]:.4f}")
print("[out] u1_strong_9mer_diversity.tsv")
