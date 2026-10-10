#!/usr/bin/env python3
# multi-window strong-3'ss / strong-5'ss counts up & down for the acceptor-count asymmetry
import sys, csv, numpy as np
from multiprocessing import Pool
sys.path.insert(0,"scripts")
def log(*a): print(*a,file=sys.stderr,flush=True)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
POS="fig3_new_positional_features.tsv"
DTHR=8.46; A3THR=6.5; PAD=25
WINS=[1000,2000,5000,10000,20000]
SPAN_REAL=20000; SPAN_DEC=5000

B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
_a=np.load(ME5NPY,allow_pickle=True); me5d=_a.item() if _a.dtype==object else dict(_a)
_ks=np.frombuffer("".join(me5d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
ME5=np.full(4**9,-99.0); ME5[B[_ks]@POW]=np.fromiter(me5d.values(),np.float64,len(me5d))

_G=None; _m3=None
def _init():
    global _G,_m3
    from maxentpy.maxent import load_matrix3
    _m3=load_matrix3()
    class Faidx:
        _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
        def __init__(s,fa):
            s.fh=open(fa,"rb"); s.idx={}
            for L in open(fa+".fai"):
                p=L.split(); s.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
        def fetch(s,c,a,b,st="+"):
            c=c if c in s.idx else c.lstrip("chr")
            if c not in s.idx: return None
            Ln,off,lb,lw=s.idx[c]
            if a<0 or b>Ln: return None
            bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
            q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
            return q.translate(s._T)[::-1] if st=="-" else q
    _G=Faidx(FASTA)

def scan_offsets(chrom,center,strand,span):
    from maxentpy import maxent
    a0=center-(span+PAD); b0=center+(span+PAD)
    seq=_G.fetch(chrom,a0,b0,strand)
    if not seq or len(seq)<2*(span+PAD): return None
    arr=np.frombuffer(seq.encode("ascii"),np.uint8)
    tok=B[arr]; sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    sc5=np.full(sw.shape[0],-99.0); sc5[good]=ME5[sw[good]@POW]
    o5=(-(span+PAD))+np.arange(sc5.size)+3
    o5s=o5[sc5>=DTHR]
    L3=len(seq)-22
    isA=(arr[18:18+L3]==65); isG=(arr[19:19+L3]==71)
    agidx=np.where(isA&isG)[0]
    o3all=(-(span+PAD))+agidx+20
    keep=[]
    for j,ix in enumerate(agidx):
        km=seq[ix:ix+23]
        if "N" in km: continue
        if maxent.score3(km,matrix=_m3)>=A3THR: keep.append(o3all[j])
    o3s=np.array(keep,float) if keep else np.empty(0)
    return o3s,o5s

def counts(o,wins):
    up=[int(((o>=-w)&(o<0)).sum()) for w in wins]
    dn=[int(((o>0)&(o<=w)).sum()) for w in wins]
    return up,dn

def work_real(job):
    i,chrom,center,strand=job
    r=scan_offsets(chrom,center,strand,SPAN_REAL)
    if r is None: return i,None
    o3,o5=r
    u3,d3=counts(o3,WINS); u5,d5=counts(o5,WINS)
    return i,(u3,d3,u5,d5)

def work_decoy(job):
    j,chrom,pos,strand=job
    r=scan_offsets(chrom,pos,strand,SPAN_DEC)
    if r is None: return j,(np.nan,np.nan)
    o3,_=r
    return j,(float(((o3>=-5000)&(o3<0)).sum()), float(((o3>0)&(o3<=5000)).sum()))

if __name__=="__main__":
    R=list(csv.DictReader(open(POS),delimiter="\t"))
    NP=len(R)
    chrom=[r["chrom"] for r in R]; center=[int(r["center"]) for r in R]; strand=[r["strand"] for r in R]
    NW=len(WINS)
    U3=np.full((NP,NW),np.nan); D3=np.full((NP,NW),np.nan)
    U5=np.full((NP,NW),np.nan); D5=np.full((NP,NW),np.nan)
    jobs=[(i,chrom[i],center[i],strand[i]) for i in range(NP)]
    log(f"[real] scanning {NP} peaks +/-{SPAN_REAL} ...")
    with Pool(32,initializer=_init) as p:
        for n,(i,res) in enumerate(p.imap_unordered(work_real,jobs,chunksize=64)):
            if res is not None:
                u3,d3,u5,d5=res; U3[i]=u3; D3[i]=d3; U5[i]=u5; D5[i]=d5
            if (n+1)%10000==0: log(f"  real {n+1}/{NP}")
    out={}
    for k,w in enumerate(WINS):
        wk=w//1000
        out[f"n3ss_up_{wk}"]=U3[:,k]; out[f"n3ss_dn_{wk}"]=D3[:,k]
        out[f"n5ss_up_{wk}"]=U5[:,k]; out[f"n5ss_dn_{wk}"]=D5[:,k]
    np.savez("asym_real_cache.npz",**out)
    log("saved asym_real_cache.npz")
    old=np.load("splice_feats_cache.npz")
    m=np.isfinite(out["n3ss_up_5"])&np.isfinite(old["C_n3ss_up5k"])
    log(f"  validate n3ss_up5k vs cache: corr {np.corrcoef(out['n3ss_up_5'][m],old['C_n3ss_up5k'][m])[0,1]:.4f} "
        f"identical {(out['n3ss_up_5'][m]==old['C_n3ss_up5k'][m]).mean():.4f}")

    D=np.load("decoy_set.npz"); dpos=D["pos"]; dcomp=D["comp"]
    ND=len(dpos)
    djobs=[(j,chrom[int(dcomp[j])],int(dpos[j]),strand[int(dcomp[j])]) for j in range(ND)]
    dU=np.full(ND,np.nan); dD=np.full(ND,np.nan)
    log(f"[decoy] scanning {ND} decoys +/-{SPAN_DEC} ...")
    with Pool(32,initializer=_init) as p:
        for n,(j,(u,d)) in enumerate(p.imap_unordered(work_decoy,djobs,chunksize=128)):
            dU[j]=u; dD[j]=d
            if (n+1)%20000==0: log(f"  decoy {n+1}/{ND}")
    np.savez("asym_decoy_cache.npz",d_n3ss_up5k=dU,d_n3ss_dn5k=dD)
    log("saved asym_decoy_cache.npz  DONE_BUILD_ASYM")
