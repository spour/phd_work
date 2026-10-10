#!/usr/bin/env python3
# reversal scan: regenerate the u1 window and metagene panels at the primary cutoff 5.25
import sys, csv
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)

FASTA="../train/data/hg38.fa"; U1NPY="../train/splice_train/data/u1_lookup.npy"
META="fig3_ucube_meta.tsv"
THR=5.25
SPAN_UP=2000; SPAN_DN=500; PAD=12
WINS=np.arange(50,2001,50)
NYBIN=14
MSPAN=500; MBINW=25; MNBIN=2*MSPAN//MBINW; MEDGES=np.arange(-MSPAN,MSPAN+1,MBINW)
MCENTERS=(MEDGES[:-1]+MEDGES[1:])/2.0

# u1 lookup -> dense 4^9, negated (higher = stronger binding), as the pipeline
B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
a=np.load(U1NPY,allow_pickle=True); d=a.item() if a.dtype==object else dict(a)
kb=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
SCORE=np.full(4**9,np.nan); SCORE[(B[kb]@POW)]=-np.fromiter(d.values(),np.float64,len(d))
L=9

class Faidx:
    _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
    def __init__(s,fa):
        s.fh=open(fa,"rb"); s.idx={}
        for Ln in open(fa+".fai"):
            p=Ln.split(); s.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
    def fetch(s,c,a,b,st="+"):
        c=c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln,off,lb,lw=s.idx[c]
        if a<0 or b>Ln: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st=="-" else q
G=Faidx(FASTA)

# read peaks (aligned to fig3_ucube_meta / denswin is_canonical)
gene=[]; isc=[]; rows=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        gene.append(r["gene"]); isc.append(int(r["is_canonical"]))
        rows.append((r["chrom"],int(r["center"]),r["strand"]))
NP=len(rows); isc=np.array(isc); gene=np.array(gene)
log(f"[scan] {NP:,} peaks, thr={THR}, windows {WINS[0]}..{WINS[-1]}")

dwc=np.zeros((NP,len(WINS)),np.float64)
meta_d=np.full((NP,MNBIN),np.nan,np.float64)
dropped=0
for i,(c,ctr,st) in enumerate(rows):
    if st=="+": a,b=ctr-SPAN_UP-PAD, ctr+SPAN_DN+PAD
    else:       a,b=ctr-SPAN_DN-PAD, ctr+SPAN_UP+PAD
    seq=G.fetch(c,a,b,st)
    if not seq or len(seq)<(SPAN_UP+SPAN_DN):
        dropped+=1; continue
    tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
    sw=np.lib.stride_tricks.sliding_window_view(tok,L); good=(sw>=0).all(1)
    sc=np.full(sw.shape[0],np.nan); sc[good]=SCORE[sw[good]@POW]
    d0=-(SPAN_UP+PAD)
    dpos=d0+np.arange(sc.size)
    ok=np.isfinite(sc); dv=dpos[ok]; sv=sc[ok]
    hit=sv>=THR
    up=dv<0
    dup=dv[up]; hup=hit[up]
    if hup.any():
        dneg=-dup[hup]
        ds=np.sort(dneg)
        dwc[i]=np.searchsorted(ds, WINS, side="right")
    m=(dv>=-MSPAN)&(dv<MSPAN)
    if m.any():
        bi=np.digitize(dv[m],MEDGES)-1
        mm=(bi>=0)&(bi<MNBIN)
        cnt=np.bincount(bi[mm],weights=hit[m][mm].astype(float),minlength=MNBIN)
        tot=np.bincount(bi[mm],minlength=MNBIN).astype(float)
        meta_d[i]=np.where(tot>0,cnt/np.maximum(tot,1)*1000.0,np.nan)
    if (i+1)%10000==0: log(f"  {i+1:,}/{NP:,}")
log(f"[scan] done; dropped {dropped} (contig boundary)")

dens=dwc/(WINS[None,:]/1000.0)

# panel b: bcshow ratio + auroc (u1, thr 5.25)
def auroc(pos,neg):
    if len(pos)<20 or len(neg)<20: return np.nan
    a=np.concatenate([pos,neg]); order=np.argsort(a,kind="mergesort")
    ranks=np.empty(len(a)); ranks[order]=np.arange(1,len(a)+1)
    _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
    csum=np.cumsum(cnt); start=csum-cnt
    avg=(start+csum+1)/2.0
    ranks=avg[inv]
    R1=ranks[:len(pos)].sum()
    return (R1-len(pos)*(len(pos)+1)/2.0)/(len(pos)*len(neg))
y1=isc==1; y0=isc==0; n1=int(y1.sum()); n0=int(y0.sum()); eps=1.0/max(n1,n0)
au_rows=[]; rt_rows=[]
for wi,w in enumerate(WINS):
    sc=dens[:,wi]
    raw=auroc(sc[y1],sc[y0])
    if np.isfinite(raw):
        au_rows.append(dict(mech="u1",window=float(w),auroc_raw=raw,
                            auroc=1.0-raw if raw<0.5 else raw,
                            predicts="competitor" if raw<0.5 else "true end",
                            thr=THR,encoding=f"strong-site density/kb, threshold {THR:g}"))
hi=float(np.nanpercentile(dens,99.0)); edges=np.linspace(0.0,hi if hi>0 else 1.0,NYBIN+1)
ctr=(edges[:-1]+edges[1:])/2.0
for wi,w in enumerate(WINS):
    bi=np.clip(np.searchsorted(edges,dens[:,wi],side="right")-1,0,NYBIN-1)
    for j in range(NYBIN):
        m_=bi==j; n=int(m_.sum())
        if n==0: continue
        pc=float((m_&y1).sum())/n1; pk=float((m_&y0).sum())/n0
        rt_rows.append(dict(mech="u1",window=float(w),yval=float(ctr[j]),
                            log2ratio=float(np.log2((pc+eps)/(pk+eps))),n=n,strong_thr=THR))
import pandas as pd
pd.DataFrame(au_rows).to_csv("fig3rev_bcshow_auroc.tsv",sep="\t",index=False)
pd.DataFrame(rt_rows).to_csv("fig3rev_bcshow_ratio.tsv",sep="\t",index=False)
pkw=max(au_rows,key=lambda r:r["auroc"])
log(f"[panelB] u1@{THR}: peak oriented AUROC {pkw['auroc']:.4f} at {pkw['window']:.0f}nt "
    f"(raw {pkw['auroc_raw']:.4f}, predicts {pkw['predicts']})")

# panel f: within-gene paired metagene (thr 5.25)
by=defaultdict(lambda:([],[]))
for i in range(NP):
    by[gene[i]][0 if isc[i]==1 else 1].append(i)
gname=[]; CAN=[]; COM=[]
for g,(ci,ki) in by.items():
    if not ci or not ki: continue
    cc=np.nanmean(meta_d[ci],axis=0); kk=np.nanmean(meta_d[ki],axis=0)
    if not (np.isfinite(cc).all() and np.isfinite(kk).all()): continue
    gname.append(g); CAN.append(cc); COM.append(kk)
CAN=np.asarray(CAN); COM=np.asarray(COM); ng=len(gname)
mid=(CAN+COM)/2.0; CANc=CAN-mid; COMc=COM-mid; DIFF=COM-CAN
import math
def mse(A): return A.mean(0), A.std(0,ddof=1)/math.sqrt(A.shape[0])
can_m,can_s=mse(CANc); com_m,com_s=mse(COMc); dif_m,dif_s=mse(DIFF)
rng=np.random.default_rng(0); NBOOT=1000
idx=rng.integers(0,ng,size=(NBOOT,ng)); boot=np.empty((NBOOT,MNBIN))
for bi in range(NBOOT): boot[bi]=DIFF[idx[bi]].mean(0)
ci_lo=np.percentile(boot,2.5,0); ci_hi=np.percentile(boot,97.5,0)
with open("paired_u1_metagene_525.tsv","w") as fh:
    fh.write("pos_bin\tclass\tu1_centered_mean\tse\tn_genes\n")
    for j,p in enumerate(MCENTERS):
        fh.write("%g\tCompetitor\t%.6g\t%.6g\t%d\n"%(p,com_m[j],com_s[j],ng))
        fh.write("%g\tReal end\t%.6g\t%.6g\t%d\n"%(p,can_m[j],can_s[j],ng))
with open("paired_u1_excess_525.tsv","w") as fh:
    fh.write("pos_bin\texcess_mean\tci_lo\tci_hi\tse\tn_genes\tn_boot\n")
    for j,p in enumerate(MCENTERS):
        fh.write("%g\t%.6g\t%.6g\t%.6g\t%.6g\t%d\t%d\n"%(p,dif_m[j],ci_lo[j],ci_hi[j],dif_s[j],ng,NBOOT))
up=MCENTERS<0; dn=MCENTERS>0
n_above=int((ci_lo[up]>0).sum()); nbin_up=int(up.sum())
log(f"[panelF] paired genes {ng}; upstream excess {DIFF[:,up].mean():+.2f} sites/kb; "
    f"CI>0 in {n_above}/{nbin_up} upstream bins; at pA bin {dif_m[np.argmin(np.abs(MCENTERS))]:+.2f}")
np.savez_compressed("u1_metagene_cache_525.npz",M=meta_d.astype(np.float32),gene=gene,
                    centers=MCENTERS.astype(np.float32))
print(f"[out] fig3rev_bcshow_ratio.tsv fig3rev_bcshow_auroc.tsv "
      f"paired_u1_excess_525.tsv paired_u1_metagene_525.tsv u1_metagene_cache_525.npz")
print(f"[panelB] peak u1 AUROC {pkw['auroc']:.4f}@{pkw['window']:.0f}nt ; "
      f"[panelF] upstream excess {DIFF[:,up].mean():+.2f}/kb, CI>0 {n_above}/{nbin_up}")
