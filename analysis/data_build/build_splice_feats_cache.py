#!/usr/bin/env python3
# one-time cache of new maxent-scan splice_arch candidate features (groups a/b/c)
import sys, csv, numpy as np
sys.path.insert(0, "scripts")
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
def log(*a): print(*a, file=sys.stderr, flush=True)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
POS="fig3_new_positional_features.tsv"
SPAN=5000; PAD=25; DTHR=8.46; A3THR=6.5
m3=load_matrix3()

B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
a=np.load(ME5NPY,allow_pickle=True); me5d=a.item() if a.dtype==object else dict(a)
ks=np.frombuffer("".join(me5d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
ME5=np.full(4**9,-99.0); ME5[B[ks]@POW]=np.fromiter(me5d.values(),np.float64,len(me5d))

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
G=Faidx(FASTA)

R=list(csv.DictReader(open(POS),delimiter="\t"))
chrom=[r["chrom"] for r in R]; center=[int(r["center"]) for r in R]; strand=[r["strand"] for r in R]
acc=np.array([float(r["acc"]) if r["acc"] not in("","nan") else np.nan for r in R])
dpa=np.array([float(r["d_anchor_to_pa"]) if r["d_anchor_to_pa"] not in("","nan") else np.nan for r in R])
NP=len(R)

names=["A_ndon_up","A_ndon_dn","A_sumdon_up","A_dstrong_up_pA","A_dstrong_up_anchor",
       "Dmax_up","Dmax_dn","B_acc_rank2k","B_acc_margin2k",
       "C_dist_3ss_up","C_dist_3ss_dn","C_dist_5ss_up","C_dist_5ss_dn",
       "C_n3ss_up5k","C_n3ss_dn5k","C_n5ss_up5k","C_n5ss_dn5k","C_in_intron","V_me3_anchor"]
OUT={n:np.full(NP,np.nan) for n in names}
W2=2000
for i in range(NP):
    a0=center[i]-(SPAN+PAD); b0=center[i]+(SPAN+PAD)
    seq=G.fetch(chrom[i],a0,b0,strand[i])
    if not seq or len(seq)<2*(SPAN+PAD): continue
    arr=np.frombuffer(seq.encode("ascii"),np.uint8)
    # donor (me5) dense scan
    tok=B[arr]; sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    sc5=np.full(sw.shape[0],-99.0); sc5[good]=ME5[sw[good]@POW]
    o5=(-(SPAN+PAD))+np.arange(sc5.size)+3
    # acceptor (me3) ag-anchored scan
    L3=len(seq)-22
    isA=(arr[18:18+L3]==65); isG=(arr[19:19+L3]==71)
    agidx=np.where(isA&isG)[0]
    o3=(-(SPAN+PAD))+agidx+20
    sc3=np.empty(agidx.size)
    for j,ix in enumerate(agidx):
        km=seq[ix:ix+23]
        sc3[j]=maxent.score3(km,matrix=m3) if "N" not in km else np.nan
    fin3=np.isfinite(sc3); o3=o3[fin3]; sc3=sc3[fin3]

    d=dpa[i]
    # a: donor dosage over (anchor,pa)=[-d,0) and (pa,pa+d)=(0,d]
    if np.isfinite(d) and d>=1:
        dd=min(d,SPAN)
        up=(o5>=-dd)&(o5<0); dn=(o5>0)&(o5<=dd)
        OUT["A_ndon_up"][i]=float(((sc5>=DTHR)&up).sum())
        OUT["A_ndon_dn"][i]=float(((sc5>=DTHR)&dn).sum())
        OUT["A_sumdon_up"][i]=float(np.clip(sc5[up],0,None).sum()) if up.any() else 0.0
        OUT["Dmax_up"][i]=float(sc5[up].max()) if up.any() else -99.0
        OUT["Dmax_dn"][i]=float(sc5[dn].max()) if dn.any() else -99.0
        if up.any():
            k=np.argmax(sc5[up]); ostar=o5[up][k]
            OUT["A_dstrong_up_pA"][i]=float(-ostar)
            OUT["A_dstrong_up_anchor"][i]=float(ostar+dd)
    # b: rank/margin of anchor 3'ss in 2 kb upstream (exclude anchor +/-5nt zone)
    if np.isfinite(acc[i]) and np.isfinite(d):
        w=(o3>=-W2)&(o3<0)
        if w.any():
            ow=o3[w]; sw3=sc3[w]
            far=np.abs(ow-(-d))>5
            others=sw3[far]
            OUT["B_acc_rank2k"][i]=float(1+int((others>acc[i]).sum()))
            OUT["B_acc_margin2k"][i]=float(acc[i]-others.max()) if others.size else np.nan
    s3=sc3>=A3THR; s5=(sc5>=DTHR)
    up3=s3&(o3<0); dn3=s3&(o3>0); up5=s5&(o5<0); dn5=s5&(o5>0)
    OUT["C_dist_3ss_up"][i]=float(np.min(-o3[up3])) if up3.any() else float(SPAN)
    OUT["C_dist_3ss_dn"][i]=float(np.min(o3[dn3])) if dn3.any() else float(SPAN)
    OUT["C_dist_5ss_up"][i]=float(np.min(-o5[up5])) if up5.any() else float(SPAN)
    OUT["C_dist_5ss_dn"][i]=float(np.min(o5[dn5])) if dn5.any() else float(SPAN)
    OUT["C_n3ss_up5k"][i]=float(up3.sum()); OUT["C_n3ss_dn5k"][i]=float(dn3.sum())
    OUT["C_n5ss_up5k"][i]=float(up5.sum()); OUT["C_n5ss_dn5k"][i]=float(dn5.sum())
    OUT["C_in_intron"][i]=1.0 if (up5.any() and dn3.any()) else 0.0
    if np.isfinite(d):
        near=np.abs(o3-(-d))<=5
        if near.any(): OUT["V_me3_anchor"][i]=float(sc3[near].max())
    if (i+1)%5000==0: log(f"  scan {i+1}/{NP}")
np.savez("splice_feats_cache.npz",**OUT)
log("saved splice_feats_cache.npz")
v=OUT["V_me3_anchor"]; ok=np.isfinite(v)&np.isfinite(acc)
log(f"validation corr(me3_anchor, master acc) = {np.corrcoef(v[ok],acc[ok])[0,1]:.4f}  (n={ok.sum()})")
