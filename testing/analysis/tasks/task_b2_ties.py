#!/usr/bin/env python3
# task b2: is the win-rate decline across the ladder a tie (sparsity) artifact?
import sys, csv
import numpy as np

FASTA="../train/data/hg38.fa"; U1NPY="../train/splice_train/data/u1_lookup.npy"
META="fig3_ucube_meta.tsv"; WIN=500; LADDER=[4.0,5.25,6.5,7.0,8.0]; PRIMARY=5.25
def log(*a): print(*a,file=sys.stderr,flush=True)

B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
arr=np.load(U1NPY,allow_pickle=True); d=arr.item() if arr.dtype==object else dict(arr)
kbytes=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
ktok=B[kbytes]; kidx=ktok@POW
SCORE=np.empty(4**9,np.float64); SCORE[kidx]=-np.fromiter(d.values(),np.float64,len(d))

order=np.argsort(kidx); tbc=ktok[order]
init=np.zeros((4**9,4)); init[np.arange(4**9),tbc[:,0]]=1.0
steps=tbc[:,:-1]*4+tbc[:,1:]; dc=np.zeros((4**9,16))
for i in range(16): dc[:,i]=(steps==i).sum(1)
FEAT=np.hstack([init,dc])

def score_window(tok):
    if tok.shape[0]<9: return np.empty(0)
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    out=np.full(sw.shape[0],np.nan); out[good]=SCORE[sw[good]@POW]; return out

class Faidx:
    _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
    def __init__(self,fa):
        self.fh=open(fa,"rb"); self.idx={}
        for line in open(fa+".fai"):
            p=line.split(); self.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
    def fetch(self,chrom,start,end,strand="+"):
        c=chrom if chrom in self.idx else chrom.lstrip("chr")
        if c not in self.idx: return None
        L,off,lb,lw=self.idx[c]; start=max(0,start); end=min(end,L)
        if end<=start: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb)
        self.fh.seek(bp(start)); raw=self.fh.read(bp(end)-bp(start))
        s=raw.replace(b"\n",b"").replace(b"\r",b"")[:end-start].decode("ascii").upper()
        return s.translate(self._T)[::-1] if strand=="-" else s
G=Faidx(FASTA)

gene,isc,chrom,center,strand=[],[],[],[],[]
with open(META) as fh:
    for row in csv.DictReader(fh,delimiter="\t"):
        gene.append(row["gene"]); isc.append(int(row["is_canonical"]))
        chrom.append(row["chrom"]); center.append(int(row["center"])); strand.append(row["strand"])
gene=np.array(gene); isc=np.array(isc); NP=len(isc); kb=WIN/1000.0
log(f"[meta] {NP:,} peaks")

counts={t:np.zeros(NP) for t in LADDER}; npos=np.zeros(NP); Wlog=np.zeros((NP,20))
for i in range(NP):
    seq=G.fetch(chrom[i],center[i],center[i]+WIN,"-") if strand[i]=="-" else G.fetch(chrom[i],center[i]-WIN,center[i],"+")
    tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)] if seq else np.empty(0,np.int64)
    sc=score_window(tok); valid=np.isfinite(sc); npos[i]=valid.sum(); v=sc[valid]
    for t in LADDER: counts[t][i]=(v>=t).sum()
    tg=tok[tok>=0]
    if tg.size>=10:
        bc=np.bincount(tg,minlength=4).astype(float); st=np.bincount(tg[:-1]*4+tg[1:],minlength=16).astype(float).reshape(4,4)
        p0=(bc+0.5)/(bc.sum()+2.0); T=(st+0.25)/(st.sum(1,keepdims=True)+1.0)
        Wlog[i]=np.concatenate([np.log(p0),np.log(T).ravel()])
    else: Wlog[i]=np.nan
    if (i+1)%20000==0: log(f"  scored {i+1:,}/{NP:,}")

density={t:counts[t]/kb for t in LADDER}
Ht=np.where(SCORE>=PRIMARY)[0]; Fh=FEAT[Ht]
q=np.zeros(NP)
for s in range(0,NP,2000):
    Wl=Wlog[s:s+2000]; ok=np.isfinite(Wl).all(1)
    lp=Fh@Wl[ok].T; qc=np.exp(lp).sum(0); qq=np.full(Wl.shape[0],np.nan); qq[ok]=qc; q[s:s+2000]=qq
excess=(counts[PRIMARY]-npos*q)/kb

gu,gi=np.unique(gene,return_inverse=True)
np.savez("u1_window_cache.npz", isc=isc, gene_idx=gi, npos=npos,
         **{f"dens_{t}":density[t] for t in LADDER}, excess_525=excess,
         ladder=np.array(LADDER), primary=PRIMARY)
log("[cache] u1_window_cache.npz")

# pair-level tie analysis
gorder=np.argsort(gene,kind="stable"); gs=gene[gorder]
cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
pairs_c=[]; pairs_k=[]
for j in range(len(cut)-1):
    idx=gorder[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1:
        pairs_c.append(np.full(len(k),c[0])); pairs_k.append(k)
pc=np.concatenate(pairs_c); pk=np.concatenate(pairs_k)
log(f"[pairs] {len(pc):,} canonical-competitor pairs")

def tie_row(val,name):
    vc=val[pc]; vk=val[pk]
    fin=np.isfinite(vc)&np.isfinite(vk); vc=vc[fin]; vk=vk[fin]; n=len(vc)
    tie=(vc==vk); bothzero=(vc==0)&(vk==0)
    info=~tie; ninfo=int(info.sum())
    wr_excl=float((vk[info]>vc[info]).mean()) if ninfo else float("nan")
    wr_half=float(((vk>vc)*1.0+tie*0.5).mean())
    return dict(feature=name,pairs=n,frac_tie=tie.mean(),frac_bothzero=bothzero.mean(),
                wr_excl=wr_excl,n_info=ninfo,wr_half=wr_half)

rows=[tie_row(density[t],f"U1 density @ {t:g}") for t in LADDER]
rows.append(tie_row(excess,"u1_excess @ 5.25"))

print("\n================  TASK B2: tie decomposition of the win-rate  ================")
print(f"{len(pc):,} canonical-competitor pairs (pair-pooled)\n")
h=f"{'feature':<22} {'frac_tie':>9} {'both_zero':>10} {'wr_excl_ties':>13} {'n_info':>8} {'wr_ties=0.5':>12}"
print(h); print("-"*len(h))
for r in rows:
    print(f"{r['feature']:<22} {r['frac_tie']:>9.4f} {r['frac_bothzero']:>10.4f} "
          f"{r['wr_excl']:>13.4f} {r['n_info']:>8,} {r['wr_half']:>12.4f}")

with open("u1_tie_analysis.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t")
    w.writerow(["feature","pairs","frac_tie","frac_bothzero","winrate_ties_excluded","n_informative","winrate_ties_half"])
    for r in rows:
        w.writerow([r["feature"],r["pairs"],f"{r['frac_tie']:.6f}",f"{r['frac_bothzero']:.6f}",
                    f"{r['wr_excl']:.6f}",r["n_info"],f"{r['wr_half']:.6f}"])
print("\n[out] u1_tie_analysis.tsv ; u1_window_cache.npz")
