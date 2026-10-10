#!/usr/bin/env python3
# calibrated architecture rules (maxent only
import sys, csv, math
import numpy as np
from collections import defaultdict
sys.path.insert(0,"scripts")
from extract_splicing_features import init_maxentscan, load_me5_lookup
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
POS="fig3_new_positional_features.tsv"
BASE=0.09308
SENS5={0.50:8.46,0.75:6.64,0.90:4.13}; SENS3={0.50:8.19,0.75:5.94,0.90:2.51}
ABS5=4.13; PAD=12; CAP=30000; UP2K=2000

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
NP=len(R); log(f"peaks {NP} base {(isc==1).mean():.5f}")

def wilson(k,n,z=1.959963984540054):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

# scan: max 5'ss in (anchor,pa)
max5=np.full(NP,-99.0); max3=np.full(NP,-99.0); ncap=0
for i in range(NP):
    ce=center[i]; st=strand[i]; ch=chrom[i]
    if np.isfinite(apos[i]) and np.isfinite(dpa[i]) and dpa[i]>=9:
        d=int(min(dpa[i],CAP)); ncap+= (dpa[i]>CAP)
        if st=="+": a,b=ce-d-PAD, ce+PAD
        else:       a,b=ce-PAD, ce+d+PAD
        seq=G.fetch(ch,a,b,st)
        if seq and len(seq)>=9:
            tok=Bm[np.frombuffer(seq.encode("ascii"),np.uint8)]
            sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
            if good.any():
                sc=ME5[sw[good]@POW]
                idx=np.where(good)[0]+3-PAD
                m=(idx>=0)&(idx<d)
                if m.any(): max5[i]=float(sc[m].max())
                else: max5[i]=float(sc.max())
    if st=="+": a,b=ce-UP2K-3, ce+3
    else:       a,b=ce-3, ce+UP2K+3
    seq=G.fetch(ch,a,b,st)
    if seq and len(seq)>=23:
        arr=np.frombuffer(seq.encode("ascii"),np.uint8)
        ag=np.where((arr[:-1]==65)&(arr[1:]==71))[0]
        best=-99.0
        for p in ag:
            s0=p-18
            if s0<0 or s0+23>len(seq): continue
            v=sc3(seq[s0:s0+23])
            if v>best: best=v
        max3[i]=best
    if (i+1)%10000==0: log(f"  scan {i+1}/{NP}")
log(f"scan done; interval capped at {CAP} for {int(ncap)} peaks")

# classes (rule 1)
strong=acc>=8
def cls(absent5):
    term=strong&(max5<absent5); intern=strong&(max5>=8); amb=~(term|intern)
    return term,intern,amb
def report_classes(absent5,tag,w):
    term,intern,amb=cls(absent5)
    for nm,m in [("terminal-like",term),("internal-like",intern),("ambiguous",amb)]:
        n=int(m.sum()); k=int((m&(isc==1)).sum()); p=k/n if n else float("nan"); lo,hi=wilson(k,n)
        w.writerow([tag,nm,n,k,f"{p:.5f}",f"{p/BASE:.3f}",f"{lo:.5f}",f"{hi:.5f}"])
        log(f"  [{tag}] {nm:14s} n={n:6d} P(real)={p:.4f} [{lo:.4f},{hi:.4f}] enrich={p/BASE:.2f}x")
with open("arch_rules.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["absence_rule","class","n","n_canon","p_real","enrich","ci_lo","ci_hi"])
    log("CLASSES: absence at 5'ss>=4.13 (90% sensitivity)"); report_classes(ABS5,"absence>=4.13",w)
    log("CLASSES: absence at 5'ss>=8 (WRONG threshold, presence cutoff)"); report_classes(8.0,"absence>=8",w)

# 2d grid
xb=np.arange(-2,13.001,1.0); yb=np.arange(0,15.001,1.0)
xi=np.digitize(max5,xb)-1; yi=np.digitize(acc,yb)-1
with open("arch_grid2d.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["x_donor","y_acceptor","n","n_canon","p_real"])
    for a_ in range(len(xb)-1):
        for b_ in range(len(yb)-1):
            m=(xi==a_)&(yi==b_); n=int(m.sum())
            if n<200: continue
            k=int((m&(isc==1)).sum())
            w.writerow([f"{(xb[a_]+xb[a_+1])/2:.1f}",f"{(yb[b_]+yb[b_+1])/2:.1f}",n,k,f"{k/n:.5f}"])
with open("arch_grid2d_refs.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["axis","sensitivity","score"])
    for tg,v in SENS5.items(): w.writerow(["donor",f"{int(tg*100)}",v])
    for tg,v in SENS3.items(): w.writerow(["acceptor",f"{int(tg*100)}",v])

# ladder
gp=defaultdict(list)
for i in range(NP): gp[gene[i]].append(i)
GEN=[g for g,idx in gp.items() if len(idx)>=2 and any(isc[j]==1 for j in idx)]
GIDX={g:np.array(gp[g]) for g in GEN}
log(f"ladder genes (>=2 peaks, has canonical): {len(GEN)}")
def top1_hits(score,tiebreak):
    hits=[]
    for g in GEN:
        idx=GIDX[g]; order=np.lexsort((tiebreak[idx],score[idx])); top=idx[order[-1]]
        hits.append(1.0 if isc[top]==1 else 0.0)
    return np.array(hits)
def boot(h,n=2000):
    rs=np.random.default_rng(1); b=h[rs.integers(0,len(h),size=(n,len(h)))].mean(1)
    return h.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
def pooled(mask):
    n=int(mask.sum()); k=int((mask&(isc==1)).sum()); p=k/n if n else float("nan"); lo,hi=wilson(k,n)
    return n,p,lo,hi
chance=np.array([ (isc[GIDX[g]]==1).mean() for g in GEN]).mean()
zero=np.zeros(NP)
term413=strong&(max5<ABS5)
top3=acc>=(max3-1e-6)
rows=[]
h=np.array([ (isc[GIDX[g]]==1).mean() for g in GEN]); rows.append(("a. base rate",BASE,0,0,0,chance,chance,chance,"random"))
dec=np.clip(np.digitize(acc,np.nanquantile(acc,np.linspace(0,1,11)[1:-1])),0,9)
ptop=pooled(dec==9); pbot=pooled(dec==0)
hb=top1_hits(acc,zero); mb,lob,hib=boot(hb)
rows.append(("b. acceptor decile",ptop[1],ptop[0],pbot[1],pbot[0],mb,lob,hib,f"top/bot ratio {ptop[1]/max(pbot[1],1e-9):.2f}"))
pc=pooled(strong); hc=top1_hits(strong.astype(float),acc); mc,loc,hic=boot(hc)
rows.append(("c. acceptor>=8",pc[1],pc[0],pc[2],pc[3],mc,loc,hic,"pooled among acc>=8"))
pd_=pooled(term413); hd=top1_hits(term413.astype(float),acc); md,lod,hid=boot(hd)
rows.append(("d. terminal-like (rule1)",pd_[1],pd_[0],pd_[2],pd_[3],md,lod,hid,"acc>=8 & no 5'ss>=4.13"))
te=term413&top3; pe=pooled(te); he=top1_hits(te.astype(float),acc); me_,loe,hie=boot(he)
rows.append(("e. + anchor is top 3'ss/2kb",pe[1],pe[0],pe[2],pe[3],me_,loe,hie,f"top3ss frac {top3.mean():.3f}"))
with open("arch_ladder.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t")
    w.writerow(["rule","p_real_or_topdecile","n_or_topn","p_bot_or_x","n_bot_or_x","top1_acc","top1_lo","top1_hi","note"])
    for r in rows: w.writerow([r[0],f"{r[1]:.5f}",r[2],f"{r[3]:.5f}",r[4],f"{r[5]:.5f}",f"{r[6]:.5f}",f"{r[7]:.5f}",r[8]])
log(f"LADDER (chance top-1 = {chance:.4f}):")
for r in rows: log(f"  {r[0]:32s} pooled/top P(real)={r[1]:.4f}  top-1 acc={r[5]:.4f} [{r[6]:.4f},{r[7]:.4f}]  {r[8]}")

with open("arch_absence_sweep.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["donor_absence_cutoff","n_terminal","p_real","ci_lo","ci_hi","top1_acc","top1_lo","top1_hi"])
    log("ABSENCE SWEEP (terminal-like = acc>=8 & max5'ss < cutoff):")
    best=None
    for cutf in (2,3,4.13,5,6,8):
        tt=strong&(max5<cutf); n,p,lo,hi=pooled(tt)
        h=top1_hits(tt.astype(float),acc); mt,lot,hit=boot(h)
        w.writerow([f"{cutf}",n,f"{p:.5f}",f"{lo:.5f}",f"{hi:.5f}",f"{mt:.5f}",f"{lot:.5f}",f"{hit:.5f}"])
        log(f"  cutoff {cutf:<5} n={n:6d} P(real)={p:.4f} [{lo:.4f},{hi:.4f}]  top-1 acc={mt:.4f}")
        if best is None or p>best[1]: best=(cutf,p,mt)
    log(f"  peak P(real) at donor-absence cutoff {best[0]} (P={best[1]:.4f}, top-1={best[2]:.4f})")
print("[out] arch_rules.tsv arch_grid2d.tsv arch_grid2d_refs.tsv arch_ladder.tsv arch_absence_sweep.tsv")
