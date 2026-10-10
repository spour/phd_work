#!/usr/bin/env python3
# b13 - matched intronic decoy control
import sys, csv, re
import numpy as np
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED=42; rng=np.random.default_rng(SEED)
FASTA="../train/data/hg38.fa"; GTF="../train/data/annotation/gencode.v49.annotation.gtf"
ATLAS="../train/data/atlas.chr.bed"; META="fig3_ucube_meta.tsv"
U1NPY="../train/splice_train/data/u1_lookup.npy"
WIN=500; EXCL=1000; STEP=25; KDEC=5; PRIMARY=5.25; MAXCAND=600

# u1 score + excess machinery (identical scale to builder)
B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
a=np.load(U1NPY,allow_pickle=True); d=a.item() if a.dtype==object else dict(a)
kb=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
ktok=B[kb]; kidx=ktok@POW
SCORE=np.empty(4**9); SCORE[kidx]=-np.fromiter(d.values(),np.float64,len(d))
order=np.argsort(kidx); tbc=ktok[order]
init=np.zeros((4**9,4)); init[np.arange(4**9),tbc[:,0]]=1.0
steps=tbc[:,:-1]*4+tbc[:,1:]; dc=np.zeros((4**9,16))
for i in range(16): dc[:,i]=(steps==i).sum(1)
FEAT=np.hstack([init,dc]); Ht=np.where(SCORE>=PRIMARY)[0]; Fh=FEAT[Ht]
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
    def fetch(s,c,a,b,st="+"):
        c=c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln,off,lb,lw=s.idx[c]; a=max(0,a); b=min(b,Ln)
        if b<=a: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st=="-" else q
G=Faidx(FASTA)

def score_pos(pos, chrom, st):
    seq=G.fetch(chrom,pos,pos+WIN,"-") if st=="-" else G.fetch(chrom,pos-WIN,pos,"+")
    if not seq: return (np.nan,np.nan,np.nan)
    tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
    sc=score_window(tok); valid=np.isfinite(sc); npos=int(valid.sum()); v=sc[valid]
    if npos==0: return (np.nan,np.nan,np.nan)
    cnt=float((v>=PRIMARY).sum()); dens=cnt/(WIN/1000.0)
    tg=tok[tok>=0]; gc=((tg==1)|(tg==2)).mean() if tg.size else np.nan
    exc=np.nan
    if tg.size>=10:
        bc=np.bincount(tg,minlength=4).astype(float); stc=np.bincount(tg[:-1]*4+tg[1:],minlength=16).astype(float).reshape(4,4)
        p0=(bc+0.5)/(bc.sum()+2.0); T=(stc+0.25)/(stc.sum(1,keepdims=True)+1.0)
        Wl=np.concatenate([np.log(p0),np.log(T).ravel()])
        q=float(np.exp(Fh@Wl).sum()); exc=(cnt-npos*q)/(WIN/1000.0)
    return (dens,gc,exc)

# gencode per-gene structure
TID=re.compile(r'transcript_id "([^"]+)"'); GNM=re.compile(r'gene_name "([^"]+)"')
tx=defaultdict(lambda:{"g":None,"chrom":None,"strand":None,"exon":[]})
n=0
with open(GTF) as fh:
    for line in fh:
        if line[0]=="#": continue
        f=line.split("\t")
        if f[2]!="exon": continue
        m=TID.search(f[8]); gm=GNM.search(f[8])
        if not m or not gm: continue
        t=tx[m.group(1)]; t["g"]=gm.group(1); t["chrom"]=f[0]; t["strand"]=f[6]
        t["exon"].append((int(f[3])-1,int(f[4])))
        n+=1
        if n%3_000_000==0: log(f"  gtf {n:,}")
gene_exons=defaultdict(list); gene_meta={}
for t in tx.values():
    if not t["exon"]: continue
    gene_exons[t["g"]].extend(t["exon"]); gene_meta[t["g"]]=(t["chrom"],t["strand"])
def merge(ivs):
    ivs=sorted(ivs); cs=[];ce=[]
    for a,b in ivs:
        if cs and a<=ce[-1]: ce[-1]=max(ce[-1],b)
        else: cs.append(a);ce.append(b)
    return list(zip(cs,ce))
gene_introns={}; gene_acc={}; gene_term={}
for g,ex in gene_exons.items():
    mex=merge(ex); st=gene_meta[g][1]
    introns=[(mex[i][1],mex[i+1][0]) for i in range(len(mex)-1) if mex[i+1][0]>mex[i][1]]
    gene_introns[g]=introns
    gene_acc[g]=np.array([b for _,b in introns] if st=="+" else [a for a,_ in introns],np.int64)
    gene_term[g]=[max(mex,key=lambda iv:iv[1])] if st=="+" else [min(mex,key=lambda iv:iv[0])]
log(f"[gtf] {len(gene_exons):,} genes")

# atlas exclusion centers per (chrom,strand)
atl=defaultdict(list)
with open(ATLAS) as fh:
    for line in fh:
        p=line.split("\t")
        if len(p)<6: continue
        atl[(p[0],p[5])].append((int(p[1])+int(p[2]))//2)
ATL={k:np.array(sorted(v),np.int64) for k,v in atl.items()}
def near_atlas(pos,chrom,st):
    arr=ATL.get((chrom,st))
    if arr is None or arr.size==0: return False
    j=np.searchsorted(arr,pos)
    for jj in (j-1,j):
        if 0<=jj<arr.size and abs(int(arr[jj])-pos)<=EXCL: return True
    return False
log(f"[atlas] {sum(len(v) for v in ATL.values()):,} clusters")

# peaks / within-gene
Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; gi=Z["gene_idx"]
u1d525=Z["dens_5.25"]; excess525=Z["excess_525"]; gc_real=np.load("gc_up500.npy")
rows=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["gene"],r["chrom"],int(r["center"]),r["strand"]))
gname=np.array([x[0] for x in rows]); chrom=np.array([x[1] for x in rows])
center=np.array([x[2] for x in rows]); strand=np.array([x[3] for x in rows])
order2=np.argsort(gi,kind="stable"); gs=gi[order2]
cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=order2[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1: gmemb.append(idx);gcanon.append(int(c[0]))
NG=len(gmemb)

def candidates(introns, chrom, st, canonpA, acc):
    cand=[]
    for a,b in introns:
        for p in range(a+1,b,STEP):
            if not near_atlas(p,chrom,st): cand.append(p)
    if not cand: return None
    cand=np.array(cand,np.int64)
    if cand.size>MAXCAND: cand=cand[rng.choice(cand.size,MAXCAND,replace=False)]
    dpa=np.abs(cand-canonpA).astype(float)
    dac=np.abs(cand[:,None]-acc[None,:]).min(1).astype(float) if acc.size else np.full(cand.size,1e9)
    return cand,dpa,dac

dec_pos=[[] for _ in range(NG)]; dec_comp=[[] for _ in range(NG)]
termdec_pos=[[] for _ in range(NG)]
bal_dpa=[]; bal_dac=[]; ncomp_ok=0; ncomp_tot=0; genes_ok=0
for g in range(NG):
    idx=gmemb[g]; c=gcanon[g]; comps=idx[idx!=c]
    gnm=gname[c]; ch=chrom[c]; st=strand[c]; canonpA=center[c]
    if gnm not in gene_introns: continue
    acc=gene_acc[gnm]
    cc=candidates(gene_introns[gnm],ch,st,canonpA,acc)
    ncomp_tot+=len(comps)
    if cc is None: continue
    cand,dpa,dac=cc
    cd_pa=np.abs(center[comps]-canonpA).astype(float)
    cd_ac=(np.abs(center[comps][:,None]-acc[None,:]).min(1).astype(float) if acc.size else np.full(len(comps),1e9))
    spa=dpa.std()+1e-9; sac=dac.std()+1e-9
    used=np.zeros(cand.size,bool)
    for ci,comp in enumerate(comps):
        dist=((dpa-cd_pa[ci])/spa)**2 + 2.0*((dac-cd_ac[ci])/sac)**2
        dist[used]=np.inf
        take=np.argsort(dist)[:KDEC]
        take=take[np.isfinite(dist[take])]
        if take.size==0: continue
        used[take]=True
        for ti in take:
            dec_pos[g].append(int(cand[ti])); dec_comp[g].append(int(comp))
            bal_dpa.append(dpa[ti]-cd_pa[ci]); bal_dac.append(dac[ti]-cd_ac[ci])
        ncomp_ok+=1
    if dec_pos[g]: genes_ok+=1
    tcand=[]
    for a,b in gene_term[gnm]:
        for p in range(a+1,b,STEP):
            if not near_atlas(p,ch,st): tcand.append(p)
    if tcand:
        tcand=np.array(tcand,np.int64)
        kk=min(KDEC*max(len(comps),1),tcand.size)
        termdec_pos[g]=list(map(int,rng.choice(tcand,kk,replace=False)))
    if (g+1)%1000==0: log(f"  decoys gene {g+1}/{NG}  matched competitors {ncomp_ok:,}")

print(f"\n=== B13 setup ===")
print(f"  seed {SEED} ; exclusion {EXCL} nt from any of {sum(len(v) for v in ATL.values()):,} atlas clusters ; {KDEC} decoys/competitor")
print(f"  competitors with decoys: {ncomp_ok:,}/{ncomp_tot:,} ; genes with decoys: {genes_ok:,}/{NG:,}")
print(f"  match balance (decoy-competitor): |d_pA| median {np.median(np.abs(bal_dpa)):.0f} nt ; "
      f"|d_3'ss| median {np.median(np.abs(bal_dac)):.0f} nt  (3'ss prioritised)")

allpos=[]; ptr_intron=[]; ptr_term=[]
for g in range(NG):
    for p in dec_pos[g]: ptr_intron.append((g,len(allpos))); allpos.append((p,chrom[gcanon[g]],strand[gcanon[g]]))
    for p in termdec_pos[g]: ptr_term.append((g,len(allpos))); allpos.append((p,chrom[gcanon[g]],strand[gcanon[g]]))
log(f"[score] {len(allpos):,} decoy windows")
sc_u1=np.empty(len(allpos)); sc_gc=np.empty(len(allpos)); sc_ex=np.empty(len(allpos))
for i,(p,ch,st) in enumerate(allpos):
    sc_u1[i],sc_gc[i],sc_ex[i]=score_pos(p,ch,st)
    if (i+1)%50000==0: log(f"  scored {i+1:,}/{len(allpos):,}")

gi_u1=[[] for _ in range(NG)]; gi_gc=[[] for _ in range(NG)]; gi_ex=[[] for _ in range(NG)]; gi_cmp=[[] for _ in range(NG)]
k=0
for g in range(NG):
    for p in dec_pos[g]:
        gi_u1[g].append(sc_u1[k]); gi_gc[g].append(sc_gc[k]); gi_ex[g].append(sc_ex[k]); gi_cmp[g].append(dec_comp[g][len(gi_u1[g])-1]); k+=1
    k+=len(termdec_pos[g])
gt_u1=[[] for _ in range(NG)]; gt_gc=[[] for _ in range(NG)]; gt_ex=[[] for _ in range(NG)]
k=0
for g in range(NG):
    k+=len(dec_pos[g])
    for p in termdec_pos[g]:
        gt_u1[g].append(sc_u1[k]); gt_gc[g].append(sc_gc[k]); gt_ex[g].append(sc_ex[k]); k+=1

REALV={"GC":gc_real,"U1@5.25":u1d525,"u1_excess":excess525}
DECV ={"GC":gi_gc,"U1@5.25":gi_u1,"u1_excess":gi_ex}
TERMV={"GC":gt_gc,"U1@5.25":gt_u1,"u1_excess":gt_ex}

def ga_boot(pg):
    ok=np.isfinite(pg); v=pg[ok]
    if len(v)<5: return (np.nan,np.nan,np.nan,int(ok.sum()))
    b=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)
    return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5),int(ok.sum())

def arm_ref_vs_decoy(realval, decval, ref_idx_fn, higher_is):
    half=np.full(NG,np.nan); sign=np.full(NG,np.nan)
    for g in range(NG):
        dv=np.array(decval[g],float); dv=dv[np.isfinite(dv)]
        if dv.size==0: continue
        rv=realval[ref_idx_fn(g)]
        if not np.isfinite(rv): continue
        d=dv-rv
        nz=d!=0; w=int((d>0).sum()); t=int((d==0).sum()); n=dv.size
        half[g]=(w+0.5*t)/n
        if nz.any(): sign[g]=(d[nz]>0).mean()
    return half,sign

def arm_comp_vs_decoy(realval, decval, deccmp):
    half=np.full(NG,np.nan); sign=np.full(NG,np.nan)
    for g in range(NG):
        dv=np.array(decval[g],float); cm=np.array(deccmp[g])
        ok=np.isfinite(dv)
        if ok.sum()==0: continue
        cv=realval[cm[ok]]; dvv=dv[ok]; okk=np.isfinite(cv)
        if okk.sum()==0: continue
        d=cv[okk]-dvv[okk]
        nz=d!=0; w=int((d>0).sum()); t=int((d==0).sum()); n=int(okk.sum())
        half[g]=(w+0.5*t)/n
        if nz.any(): sign[g]=(d[nz]>0).mean()
    return half,sign

print("\n============  B13.1  canonical vs intronic DECOY (placebo)  ============")
print("  win-rate that DECOY > canonical (compare to canonical-vs-competitor marginal)\n")
for nm in ["GC","U1@5.25","u1_excess"]:
    half,sign=arm_ref_vs_decoy(REALV[nm],DECV[nm],lambda g:gcanon[g],"decoy")
    hm,hlo,hhi,ng=ga_boot(half); sm,slo,shi,_=ga_boot(sign)
    print(f"  {nm:<10} ties.5 {hm:.4f} [{hlo:.4f},{hhi:.4f}] exc {hm-0.5:+.4f} | sign {sm:.4f} [{slo:.4f},{shi:.4f}] | {ng} genes")

print("\n============  B13.2  COMPETITOR vs matched intronic DECOY (decisive)  ============")
print("  both intronic, compartment fixed; win-rate that real CPA competitor > matched non-CPA decoy\n")
for nm in ["GC","U1@5.25","u1_excess"]:
    half,sign=arm_comp_vs_decoy(REALV[nm],DECV[nm],dec_comp)
    hm,hlo,hhi,ng=ga_boot(half); sm,slo,shi,_=ga_boot(sign)
    print(f"  {nm:<10} ties.5 {hm:.4f} [{hlo:.4f},{hhi:.4f}] exc {hm-0.5:+.4f} | sign {sm:.4f} [{slo:.4f},{shi:.4f}] | {ng} genes")

print("\n============  B13.3  canonical vs TERM-exon DECOY (mirror placebo)  ============")
print("  win-rate that TERM decoy > canonical (same compartment as canonical)\n")
for nm in ["GC","U1@5.25","u1_excess"]:
    half,sign=arm_ref_vs_decoy(REALV[nm],TERMV[nm],lambda g:gcanon[g],"decoy")
    hm,hlo,hhi,ng=ga_boot(half); sm,slo,shi,_=ga_boot(sign)
    print(f"  {nm:<10} ties.5 {hm:.4f} [{hlo:.4f},{hhi:.4f}] exc {hm-0.5:+.4f} | sign {sm:.4f} [{slo:.4f},{shi:.4f}] | {ng} genes")
print("\n[done]")
