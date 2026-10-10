#!/usr/bin/env python3
# b16 (firm the negative) + b17 (pas/dse decoy battery), one scoring pass
import sys, csv, re
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED=42; rng=np.random.default_rng(SEED)
FASTA="../train/data/hg38.fa"; GTF="../train/data/annotation/gencode.v49.annotation.gtf"
ATLAS="../train/data/atlas.chr.bed"; META="fig3_ucube_meta.tsv"
U1NPY="../train/splice_train/data/u1_lookup.npy"
WIN=500; EXCL=1000; STEP=25; KDEC=5; PRIMARY=5.25; MAXCAND=600
DPA_CAP=500; PAS_W=50; DN_W=50
PAS12=["AATAAA","ATTAAA","TATAAA","AGTAAA","AAGAAA","AATACA","CATAAA","GATAAA","AATATA","AATGAA","TTTAAA","ACTAAA"]
PAS_STRONG=("AATAAA","ATTAAA")

# u1 score + excess machinery
B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
a=np.load(U1NPY,allow_pickle=True); d=a.item() if a.dtype==object else dict(a)
kb=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9); ktok=B[kb]; kidx=ktok@POW
SCORE=np.empty(4**9); SCORE[kidx]=-np.fromiter(d.values(),np.float64,len(d))
o=np.argsort(kidx); tbc=ktok[o]; init=np.zeros((4**9,4)); init[np.arange(4**9),tbc[:,0]]=1.0
steps=tbc[:,:-1]*4+tbc[:,1:]; dcn=np.zeros((4**9,16))
for i in range(16): dcn[:,i]=(steps==i).sum(1)
FEAT=np.hstack([init,dcn]); Ht=np.where(SCORE>=PRIMARY)[0]; Fh=FEAT[Ht]
def score_window(tok):
    if tok.shape[0]<9: return np.empty(0)
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    out=np.full(sw.shape[0],np.nan); out[good]=SCORE[sw[good]@POW]; return out

class Faidx:
    _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
    def __init__(s,fa):
        s.fh=open(fa,"rb"); s.idx={}
        for L in open(fa+".fai"):
            p=L.split(); s.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
    def raw(s,c,a,b,st):
        c=c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln,off,lb,lw=s.idx[c]; a=max(0,a); b=min(b,Ln)
        if b<=a: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a]
        return q[::-1] if st=="-" else q
    def fetch(s,c,a,b,st="+"):
        c=c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln,off,lb,lw=s.idx[c]; a=max(0,a); b=min(b,Ln)
        if b<=a: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st=="-" else q
G=Faidx(FASTA)

def u1_feats(pos,ch,st):
    seq=G.fetch(ch,pos,pos+WIN,"-") if st=="-" else G.fetch(ch,pos-WIN,pos,"+")
    if not seq: return (np.nan,)*4
    tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
    sc=score_window(tok); valid=np.isfinite(sc); npos=int(valid.sum()); v=sc[valid]
    if npos==0: return (np.nan,)*4
    cnt=float((v>=PRIMARY).sum()); dens=cnt/(WIN/1000.0)
    tg=tok[tok>=0]; gc=((tg==1)|(tg==2)).mean() if tg.size else np.nan
    exc=np.nan
    if tg.size>=10:
        bc=np.bincount(tg,minlength=4).astype(float); stc=np.bincount(tg[:-1]*4+tg[1:],minlength=16).astype(float).reshape(4,4)
        p0=(bc+0.5)/(bc.sum()+2.0); T=(stc+0.25)/(stc.sum(1,keepdims=True)+1.0)
        Wl=np.concatenate([np.log(p0),np.log(T).ravel()]); q=float(np.exp(Fh@Wl).sum()); exc=(cnt-npos*q)/(WIN/1000.0)
    rb=G.raw(ch,pos,pos+WIN,"-") if st=="-" else G.raw(ch,pos-WIN,pos,"+")
    low=np.nan
    if rb:
        arr=np.frombuffer(rb,np.uint8); low=float(((arr>=97)&(arr<=122)).mean())
    return dens,gc,exc,low

def arch_feats(pos,ch,st):
    up=G.fetch(ch,pos,pos+WIN,"-") if st=="-" else G.fetch(ch,pos-WIN,pos,"+")
    dn=G.fetch(ch,pos-DN_W,pos,"-") if st=="-" else G.fetch(ch,pos,pos+DN_W,"+")
    ps=pa=pc=at=td=gu=np.nan
    if up:
        pw=up[-PAS_W:]; cnt=sum(pw.count(h) for h in PAS12)
        pc=cnt; pa=1.0 if cnt>0 else 0.0; ps=1.0 if any(h in pw for h in PAS_STRONG) else 0.0
    if dn:
        dt=B[np.frombuffer(dn.encode("ascii"),np.uint8)]; dg=dt[dt>=0]
        if dg.size:
            td=(dg==3).mean(); at=((dg==0)|(dg==3)).mean()
            di=dg[:-1]*4+dg[1:]; gu=np.isin(di,(11,14)).mean() if di.size else np.nan
    return ps,pa,pc,at,td,gu

# gencode per-gene introns / acceptors / donors / terminal exon
TID=re.compile(r'transcript_id "([^"]+)"'); GNM=re.compile(r'gene_name "([^"]+)"')
tx=defaultdict(lambda:{"g":None,"chrom":None,"strand":None,"exon":[]}); n=0
with open(GTF) as fh:
    for line in fh:
        if line[0]=="#": continue
        f=line.split("\t")
        if f[2]!="exon": continue
        m=TID.search(f[8]); gm=GNM.search(f[8])
        if not m or not gm: continue
        t=tx[m.group(1)]; t["g"]=gm.group(1); t["chrom"]=f[0]; t["strand"]=f[6]; t["exon"].append((int(f[3])-1,int(f[4])))
        n+=1
        if n%4_000_000==0: log(f"  gtf {n:,}")
ge=defaultdict(list); gmeta={}
for t in tx.values():
    if t["exon"]: ge[t["g"]].extend(t["exon"]); gmeta[t["g"]]=(t["chrom"],t["strand"])
def merge(ivs):
    ivs=sorted(ivs); cs=[];ce=[]
    for a,b in ivs:
        if cs and a<=ce[-1]: ce[-1]=max(ce[-1],b)
        else: cs.append(a);ce.append(b)
    return list(zip(cs,ce))
g_intron={}; g_acc={}; g_don={}; g_term={}
for g,ex in ge.items():
    mex=merge(ex); st=gmeta[g][1]
    intr=[(mex[i][1],mex[i+1][0]) for i in range(len(mex)-1) if mex[i+1][0]>mex[i][1]]
    g_intron[g]=intr
    g_acc[g]=np.array([b for _,b in intr] if st=="+" else [a for a,_ in intr],np.int64)
    g_don[g]=np.array([a for a,_ in intr] if st=="+" else [b for _,b in intr],np.int64)
    g_term[g]=[max(mex,key=lambda iv:iv[1])] if st=="+" else [min(mex,key=lambda iv:iv[0])]
log(f"[gtf] {len(ge):,} genes")

atl=defaultdict(list)
with open(ATLAS) as fh:
    for line in fh:
        p=line.split("\t")
        if len(p)<6: continue
        atl[(p[0],p[5])].append((int(p[1])+int(p[2]))//2)
ATL={k:np.array(sorted(v),np.int64) for k,v in atl.items()}
def near_atlas(pos,ch,st,rad):
    arr=ATL.get((ch,st))
    if arr is None or arr.size==0: return False
    j=np.searchsorted(arr,pos)
    for jj in (j-1,j):
        if 0<=jj<arr.size and abs(int(arr[jj])-pos)<=rad: return True
    return False

# peaks / within-gene
Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; gi=Z["gene_idx"]
u1d525=Z["dens_5.25"]; excess525=Z["excess_525"]; gc_real=np.load("gc_up500.npy")
rows=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["gene"],r["chrom"],int(r["center"]),r["strand"]))
gname=np.array([x[0] for x in rows]); chrom=np.array([x[1] for x in rows])
center=np.array([x[2] for x in rows]); strand=np.array([x[3] for x in rows])
o2=np.argsort(gi,kind="stable"); gs=gi[o2]; cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=o2[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1: gmemb.append(idx);gcanon.append(int(c[0]))
NG=len(gmemb)

def nearest(pos,arr): return float(np.abs(arr-pos).min()) if arr.size else 1e9

# draw + tightened match
dec_pos=[[] for _ in range(NG)]; dec_comp=[[] for _ in range(NG)]; dec_acc=[[] for _ in range(NG)]
term_pos=[[] for _ in range(NG)]
bal=[[],[],[]]; ncomp_ok=0; ncomp_tot=0
for g in range(NG):
    idx=gmemb[g]; c=gcanon[g]; comps=idx[idx!=c]; ncomp_tot+=len(comps)
    gnm=gname[c]; ch=chrom[c]; st=strand[c]; canonpA=center[c]
    if gnm not in g_intron: continue
    acc=g_acc[gnm]; don=g_don[gnm]
    cand=[]
    for a,b in g_intron[gnm]:
        for p in range(a+1,b,STEP):
            if not near_atlas(p,ch,st,EXCL): cand.append(p)
    if not cand: continue
    cand=np.array(cand,np.int64)
    if cand.size>MAXCAND: cand=cand[rng.choice(cand.size,MAXCAND,replace=False)]
    cdpa=np.abs(cand-canonpA).astype(float)
    cdac=np.array([nearest(p,acc) for p in cand]); cddo=np.array([nearest(p,don) for p in cand])
    spa=cdpa.std()+1e-9; sac=cdac.std()+1e-9; sdo=cddo.std()+1e-9
    used=np.zeros(cand.size,bool)
    for comp in comps:
        cp=center[comp]; qpa=abs(cp-canonpA); qac=nearest(cp,acc); qdo=nearest(cp,don)
        dist=((cdpa-qpa)/spa)**2 + 2.0*((cdac-qac)/sac)**2 + 2.0*((cddo-qdo)/sdo)**2
        dist[np.abs(cdpa-qpa)>DPA_CAP]=np.inf; dist[used]=np.inf
        take=np.argsort(dist)[:KDEC]; take=take[np.isfinite(dist[take])]
        if take.size==0: continue
        used[take]=True; ncomp_ok+=1
        for ti in take:
            dec_pos[g].append(int(cand[ti])); dec_comp[g].append(int(comp)); dec_acc[g].append(float(cdac[ti]))
            bal[0].append(cdpa[ti]-qpa); bal[1].append(cdac[ti]-qac); bal[2].append(cddo[ti]-qdo)
    tc=[]
    for a,b in g_term[gnm]:
        for p in range(a+1,b,STEP):
            if not near_atlas(p,ch,st,EXCL): tc.append(p)
    if tc:
        tc=np.array(tc,np.int64); kk=min(KDEC*max(len(comps),1),tc.size)
        term_pos[g]=list(map(int,rng.choice(tc,kk,replace=False)))
    if (g+1)%1500==0: log(f"  match gene {g+1}/{NG} ; matched {ncomp_ok:,}")
print("\n=== B16 tightened decoy setup ===")
print(f"  seed {SEED}; excl {EXCL}nt; |dPA| cap {DPA_CAP}nt; match on pA+3'ss+5'ss (donor/acc weight 2)")
print(f"  competitors matched: {ncomp_ok:,}/{ncomp_tot:,}")
print(f"  balance median|delta|: pA {np.median(np.abs(bal[0])):.0f}nt  3'ss {np.median(np.abs(bal[1])):.0f}nt  5'ss {np.median(np.abs(bal[2])):.0f}nt")

allpos=[]; idx_i=[]; idx_t=[]
for g in range(NG):
    for p in dec_pos[g]: idx_i.append(len(allpos)); allpos.append((p,chrom[gcanon[g]],strand[gcanon[g]]))
    for p in term_pos[g]: idx_t.append(len(allpos)); allpos.append((p,chrom[gcanon[g]],strand[gcanon[g]]))
log(f"[score] {len(allpos):,} decoy windows")
N=len(allpos)
sU=np.empty(N);sG=np.empty(N);sE=np.empty(N);sL=np.empty(N)
sPS=np.empty(N);sPA=np.empty(N);sGU=np.empty(N);sAT=np.empty(N)
for i,(p,ch,st) in enumerate(allpos):
    sU[i],sG[i],sE[i],sL[i]=u1_feats(p,ch,st)
    sPS[i],_,_,sAT[i],_,sGU[i]=arch_feats(p,ch,st)
    if (i+1)%40000==0: log(f"  scored {i+1:,}/{N}")

psR=np.empty(len(rows));atR=np.empty(len(rows));guR=np.empty(len(rows));lowR=np.empty(len(rows))
for i,(gn,ch,ct,st) in enumerate(rows):
    psR[i],_,_,atR[i],_,guR[i]=arch_feats(ct,ch,st)
    rb=G.raw(ch,ct,ct+WIN,"-") if st=="-" else G.raw(ch,ct-WIN,ct,"+")
    lowR[i]=float(((np.frombuffer(rb,np.uint8)>=97)&(np.frombuffer(rb,np.uint8)<=122)).mean()) if rb else np.nan
    if (i+1)%30000==0: log(f"  real feats {i+1:,}")

DI={"U1@5.25":[[] for _ in range(NG)],"GC":[[] for _ in range(NG)],"u1_excess":[[] for _ in range(NG)],
    "strongPAS":[[] for _ in range(NG)],"GU/UG-dn":[[] for _ in range(NG)],"AT-dn":[[] for _ in range(NG)],"low":[[] for _ in range(NG)]}
DCMP=[[] for _ in range(NG)]
DT={k:[[] for _ in range(NG)] for k in DI}
k=0
for g in range(NG):
    for _ in dec_pos[g]:
        DI["U1@5.25"][g].append(sU[k]);DI["GC"][g].append(sG[k]);DI["u1_excess"][g].append(sE[k])
        DI["strongPAS"][g].append(sPS[k]);DI["GU/UG-dn"][g].append(sGU[k]);DI["AT-dn"][g].append(sAT[k]);DI["low"][g].append(sL[k])
        DCMP[g].append(dec_comp[g][len(DI["U1@5.25"][g])-1]); k+=1
    for _ in term_pos[g]:
        DT["U1@5.25"][g].append(sU[k]);DT["GC"][g].append(sG[k]);DT["u1_excess"][g].append(sE[k])
        DT["strongPAS"][g].append(sPS[k]);DT["GU/UG-dn"][g].append(sGU[k]);DT["AT-dn"][g].append(sAT[k]);DT["low"][g].append(sL[k]); k+=1

REAL={"U1@5.25":u1d525,"GC":gc_real,"u1_excess":excess525,"strongPAS":psR,"GU/UG-dn":guR,"AT-dn":atR,"low":lowR}

def ga(pg):
    ok=np.isfinite(pg); v=pg[ok]
    if len(v)<5: return (np.nan,np.nan,np.nan,int(ok.sum()))
    b=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)
    return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5),int(ok.sum())
def ref_vs_dec(real,dec,refidx):
    half=np.full(NG,np.nan);sign=np.full(NG,np.nan)
    for g in range(NG):
        dv=np.array(dec[g],float); dv=dv[np.isfinite(dv)]
        if dv.size==0: continue
        rv=real[refidx(g)]
        if not np.isfinite(rv): continue
        dd=dv-rv; nz=dd!=0; w=int((dd>0).sum()); t=int((dd==0).sum()); half[g]=(w+0.5*t)/dv.size
        if nz.any(): sign[g]=(dd[nz]>0).mean()
    return half,sign
def comp_vs_dec(real,dec,cmp_):
    half=np.full(NG,np.nan);sign=np.full(NG,np.nan)
    for g in range(NG):
        dv=np.array(dec[g],float); cm=np.array(cmp_[g]); ok=np.isfinite(dv)
        if ok.sum()==0: continue
        cv=real[cm[ok]]; dvv=dv[ok]; okk=np.isfinite(cv)
        if okk.sum()==0: continue
        dd=cv[okk]-dvv[okk]; nz=dd!=0; w=int((dd>0).sum()); t=int((dd==0).sum()); half[g]=(w+0.5*t)/int(okk.sum())
        if nz.any(): sign[g]=(dd[nz]>0).mean()
    return half,sign
def emit(nm,half,sign):
    hm,hlo,hhi,ng=ga(half); sm,slo,shi,_=ga(sign)
    print(f"  {nm:<11} ties.5 {hm:.4f} [{hlo:.4f},{hhi:.4f}] exc {hm-0.5:+.4f} | sign {sm:.4f} [{slo:.4f},{shi:.4f}] | {ng}g")

FEATS_U1=["GC","U1@5.25","u1_excess"]; FEATS_ARCH=["strongPAS","GU/UG-dn","AT-dn"]
for title,arm in [("B13.1 canonical vs intronic decoy (placebo)",("ref",lambda g:gcanon[g],DI)),
                  ("B13.2 COMPETITOR vs matched intronic decoy (decisive)",("comp",None,DI)),
                  ("B13.3 canonical vs TERM decoy (mirror)",("ref",lambda g:gcanon[g],DT))]:
    print(f"\n============  {title}  ============")
    kind,refidx,DEC=arm
    print("  -- U1/composition (B16) --")
    for nm in FEATS_U1:
        half,sign=(ref_vs_dec(REAL[nm],DEC[nm],refidx) if kind=="ref" else comp_vs_dec(REAL[nm],DEC[nm],DCMP)); emit(nm,half,sign)
    print("  -- PAS/DSE architecture (B17) --")
    for nm in FEATS_ARCH:
        half,sign=(ref_vs_dec(REAL[nm],DEC[nm],refidx) if kind=="ref" else comp_vs_dec(REAL[nm],DEC[nm],DCMP)); emit(nm,half,sign)

print("\n=== permutation null check (B13.2 U1@5.25 competitor-vs-decoy) ===")
perm=[]
for _ in range(200):
    half=np.full(NG,np.nan)
    for g in range(NG):
        dv=np.array(DI["U1@5.25"][g],float); cm=np.array(DCMP[g]); ok=np.isfinite(dv)
        if ok.sum()==0: continue
        cv=REAL["U1@5.25"][cm[ok]]; pool=np.concatenate([cv[np.isfinite(cv)],dv[ok]])
        if pool.size<2: continue
        foc=pool[rng.integers(pool.size)]; dd=foc-pool; nz=dd!=0
        half[g]=((dd>0).sum()+0.5*(dd==0).sum())/pool.size
    perm.append(np.nanmean(half))
print(f"  permutation null mean {np.mean(perm):.4f} [{np.percentile(perm,2.5):.4f},{np.percentile(perm,97.5):.4f}]  (expect ~0.500)")

print("\n=== repeat/low-complexity integrity (soft-mask lowercase fraction) ===")
comp_low=np.array([REAL["low"][c] for g in range(NG) for c in gmemb[g][gmemb[g]!=gcanon[g]]])
dec_low=np.array([x for g in range(NG) for x in DI["low"][g]])
canon_low=np.array([REAL["low"][gcanon[g]] for g in range(NG)])
tdec_low=np.array([x for g in range(NG) for x in DT["low"][g]])
print(f"  competitors mean low-frac {np.nanmean(comp_low):.3f} vs intronic decoys {np.nanmean(dec_low):.3f}")
print(f"  canonical  mean low-frac {np.nanmean(canon_low):.3f} vs TERM decoys    {np.nanmean(tdec_low):.3f}")

flatpos=np.array([p for g in range(NG) for p in dec_pos[g]],np.int64)
flatgene=np.array([g for g in range(NG) for _ in dec_pos[g]],np.int64)
flatcmp=np.array([c for g in range(NG) for c in dec_comp[g]],np.int64)
flatacc=np.array([a for g in range(NG) for a in dec_acc[g]],np.float64)
np.savez("decoy_set.npz",pos=flatpos,gene=flatgene,comp=flatcmp,acc_dist=flatacc,
         u1=np.array([x for g in range(NG) for x in DI["U1@5.25"][g]]),
         gc=np.array([x for g in range(NG) for x in DI["GC"][g]]),
         exc=np.array([x for g in range(NG) for x in DI["u1_excess"][g]]))
print("\n[out] decoy_set.npz")
