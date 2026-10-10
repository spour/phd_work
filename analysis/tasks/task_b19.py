#!/usr/bin/env python3
# b19 - characterise the surviving positive finding
import sys, csv
import numpy as np
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; U1NPY="../train/splice_train/data/u1_lookup.npy"
META="fig3_ucube_meta.tsv"; SPAN=1500; BIN=100; PRIMARY=5.25; SUB=2500
PAS12=["AATAAA","ATTAAA","TATAAA","AGTAAA","AAGAAA","AATACA","CATAAA","GATAAA","AATATA","AATGAA","TTTAAA","ACTAAA"]
PAS_STRONG=("AATAAA","ATTAAA"); PAS_W=50; DN_W=50

B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
a=np.load(U1NPY,allow_pickle=True); d=a.item() if a.dtype==object else dict(a)
kb=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
SCORE=np.empty(4**9); SCORE[(B[kb]@POW)]=-np.fromiter(d.values(),np.float64,len(d))
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

NB=2*SPAN//BIN; offs=np.arange(-SPAN,SPAN,BIN)+BIN/2
def metagene(positions):
    accU=np.zeros(NB); accG=np.zeros(NB); cnt=np.zeros(NB)
    for (c,ctr,st) in positions:
        seq=G.fetch(c,ctr-SPAN,ctr+SPAN,st)
        if not seq or len(seq)<2*SPAN: continue
        tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
        sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
        sc=np.full(sw.shape[0],np.nan); sc[good]=SCORE[sw[good]@POW]
        hit=(sc>=PRIMARY).astype(float); gcv=((tok==1)|(tok==2)).astype(float); gcv[tok<0]=np.nan
        for bi in range(NB):
            s0=bi*BIN; s1=s0+BIN
            h=hit[s0:min(s1,len(hit))]; g=gcv[s0:s1]
            if h.size: accU[bi]+=np.nansum(h)/(BIN/1000.0);
            if np.isfinite(g).any(): accG[bi]+=np.nanmean(g)
            cnt[bi]+=1
    return accU/np.maximum(cnt,1), accG/np.maximum(cnt,1)

Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; u1d525=Z["dens_5.25"]; excess525=Z["excess_525"]
gc_real=np.load("gc_up500.npy")
rows=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["chrom"],int(r["center"]),r["strand"]))
rows=np.array(rows,dtype=object)
canon_idx=np.where(isc==1)[0]; comp_idx=np.where(isc==0)[0]
D=np.load("decoy_set.npz"); dpos=D["pos"]; dgene=D["gene"]; du1=D["u1"]; dgc=D["gc"]; dexc=D["exc"]
o=np.argsort(Z["gene_idx"],kind="stable"); gs=Z["gene_idx"][o]; cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gcanon=[]
for j in range(len(cut)-1):
    idx=o[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1: gcanon.append(int(c[0]))
gcanon=np.array(gcanon)

def samp(idx,n): return idx if len(idx)<=n else rng.choice(idx,n,replace=False)
cs=samp(canon_idx,SUB); ks=samp(comp_idx,SUB); ds=samp(np.arange(len(dpos)),SUB)
pos_c=[(rows[i][0],rows[i][1],rows[i][2]) for i in cs]
pos_k=[(rows[i][0],rows[i][1],rows[i][2]) for i in ks]
pos_d=[(rows[gcanon[dgene[i]]][0],int(dpos[i]),rows[gcanon[dgene[i]]][2]) for i in ds]
log("metagene canonical..."); Uc,Gc=metagene(pos_c)
log("metagene competitor..."); Uk,Gk=metagene(pos_k)
log("metagene decoy..."); Ud,Gd=metagene(pos_d)

print("\n============  B19  metagene: U1 hit-density (/kb) around pA (offset 0)  ============")
print(f"  {'offset':>7} {'canon':>8} {'comp':>8} {'decoy':>8}   |  GC canon/comp/decoy")
for bi in range(NB):
    if int(offs[bi])%300==50-SPAN%300 or abs(offs[bi])<=250 or True:
        if int(offs[bi])%300 in (50,-250) or abs(offs[bi])<300:
            print(f"  {int(offs[bi]):>7} {Uc[bi]:>8.1f} {Uk[bi]:>8.1f} {Ud[bi]:>8.1f}   |  {Gc[bi]:.3f} {Gk[bi]:.3f} {Gd[bi]:.3f}")
def band(A,lo,hi):
    m=(offs>=lo)&(offs<hi); return A[m].mean()
print("\n  band means (U1 /kb):")
for nm,A in [("canonical",Uc),("competitor",Uk),("decoy",Ud)]:
    print(f"    {nm:<11} pA-proximal[-500,0] {band(A,-500,0):.1f} ; far-up[-1500,-1000] {band(A,-1500,-1000):.1f} ; downstream[0,500] {band(A,0,500):.1f}")
print("  -> compare shape: sharp local pA dip (proximal<<far) vs broad compartment offset")

# parsimonious: local u1/gc vs poly-a architecture
print("\n============  B19  parsimonious test: is the pA U1/GC depletion poly-A architecture?  ============")
def arch(pos,ch,st):
    up=G.fetch(ch,pos,pos+500,"-") if st=="-" else G.fetch(ch,pos-500,pos,"+")
    dn=G.fetch(ch,pos-DN_W,pos,"-") if st=="-" else G.fetch(ch,pos,pos+DN_W,"+")
    ps=at=gu=np.nan
    if up: pw=up[-PAS_W:]; ps=1.0 if any(h in pw for h in PAS_STRONG) else 0.0
    if dn:
        dt=B[np.frombuffer(dn.encode("ascii"),np.uint8)]; dg=dt[dt>=0]
        if dg.size: at=((dg==0)|(dg==3)).mean(); di=dg[:-1]*4+dg[1:]; gu=np.isin(di,(11,14)).mean() if di.size else np.nan
    return ps,at,gu
NP=len(rows); ps=np.full(NP,np.nan); at=np.full(NP,np.nan); gu=np.full(NP,np.nan)
for i in range(NP):
    ps[i],at[i],gu[i]=arch(rows[i][1],rows[i][0],rows[i][2])
    if (i+1)%30000==0: log(f"  arch {i+1:,}")
ok=np.isfinite(u1d525)&np.isfinite(gc_real)&np.isfinite(ps)&np.isfinite(at)
print(f"  corr(local U1@5.25, strong-PAS) = {np.corrcoef(u1d525[ok],ps[ok])[0,1]:+.3f}")
print(f"  corr(local GC,      strong-PAS) = {np.corrcoef(gc_real[ok],ps[ok])[0,1]:+.3f}")
print(f"  corr(local U1@5.25, AT-downstream) = {np.corrcoef(u1d525[ok],at[ok])[0,1]:+.3f}")
print(f"  corr(local GC,      AT-downstream) = {np.corrcoef(gc_real[ok],at[ok])[0,1]:+.3f}")
print(f"  mean U1@5.25: strong-PAS present {u1d525[ok&(ps==1)].mean():.1f} vs absent {u1d525[ok&(ps==0)].mean():.1f}")
print(f"  mean GC:      strong-PAS present {gc_real[ok&(ps==1)].mean():.3f} vs absent {gc_real[ok&(ps==0)].mean():.3f}")
cflag=isc==1
for lab,m in [("all",ok),("strong-PAS present",ok&(ps==1)),("strong-PAS absent",ok&(ps==0))]:
    mc=m&cflag; mk=m&(~cflag)
    print(f"  [{lab:<20}] mean U1@5.25 canonical {u1d525[mc].mean():.1f} vs competitor {u1d525[mk].mean():.1f} ; GC {gc_real[mc].mean():.3f} vs {gc_real[mk].mean():.3f}")
print("\n[done]")
