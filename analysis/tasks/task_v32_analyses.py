#!/usr/bin/env python3
# figure3_v3_2 - three new within-gene analyses for the 'terminal exon definition' reframe
import sys, csv, math
import numpy as np
from collections import defaultdict
sys.path.insert(0,"scripts")
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
POS="fig3_new_positional_features.tsv"; SA="sa_peak_scalars_final.tsv"
ND=8.0; SPAN=2000; PAD=12; MBIN=100; DEC_SUB=30000

# dense me5 (== maxentpy score5)
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

def donor_profile(chrom,center,strand):
    if strand=="+": a,b=center-SPAN-PAD, center+SPAN+PAD
    else:           a,b=center-SPAN-PAD, center+SPAN+PAD
    seq=G.fetch(chrom,a,b,strand)
    if not seq or len(seq)<2*SPAN: return None
    tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    sc=np.full(sw.shape[0],-99.0); sc[good]=ME5[sw[good]@POW]
    d=-(SPAN+PAD)+np.arange(sc.size)+3
    m=(d>=-SPAN)&(d<=SPAN)
    return d[m], (sc[m]>=ND)

def strong_in(prof,lo,hi):
    if prof is None: return np.nan
    d,h=prof; sel=(d>=lo)&(d<hi)
    return float(h[sel].any()) if sel.any() else 0.0

# load peaks
R=list(csv.DictReader(open(POS),delimiter="\t"))
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
chrom=[r["chrom"] for r in R]; center=[int(r["center"]) for r in R]; strand=[r["strand"] for r in R]
acc=np.array([float(r["acc"]) if r["acc"] not in ("","nan") else np.nan for r in R])
def gf(col): return np.array([float(r[col]) if r[col] not in ("","nan") else np.nan for r in R])
f5dn_present=gf("first5_dn_present"); f5dn_dist=gf("first5_dn_dist")
NP=len(R); log(f"peaks {NP}  canon {int((isc==1).sum())}  comp {int((isc==0).sum())}")

# +/-2kb donor scan for all peaks
int2=np.full(NP,np.nan)
up2k=np.full(NP,np.nan); dn2k=np.full(NP,np.nan)
prof_bins=np.zeros((NP,2*SPAN//MBIN)); prof_ok=np.zeros(NP,bool)
BEDGES=np.arange(-SPAN,SPAN+1,MBIN); BCEN=(BEDGES[:-1]+BEDGES[1:])/2.0
for i in range(NP):
    pr=donor_profile(chrom[i],center[i],strand[i])
    int2[i]=strong_in(pr,1,SPAN+1); up2k[i]=strong_in(pr,-SPAN,0); dn2k[i]=strong_in(pr,1,SPAN+1)
    if pr is not None:
        d,h=pr; bi=np.digitize(d,BEDGES)-1; ok=(bi>=0)&(bi<len(BCEN))
        cnt=np.bincount(bi[ok],weights=h[ok].astype(float),minlength=len(BCEN))
        tot=np.bincount(bi[ok],minlength=len(BCEN)).astype(float)
        prof_bins[i]=np.where(tot>0,cnt/np.maximum(tot,1)*1000.0,0.0); prof_ok[i]=True
    if (i+1)%15000==0: log(f"  peak scan {i+1}/{NP}")
log("peak donor scan done")

# within-gene machinery
def pairs_by_gene(y=isc,g=gene):
    by=defaultdict(lambda:([],[]))
    for i in range(len(g)): by[g[i]][0 if y[i]==1 else 1].append(i)
    return {k:(np.array(c),np.array(k2)) for k,(c,k2) in by.items() if c and k2}
PAIRS=pairs_by_gene()
GK=list(PAIRS.keys()); NG=len(GK)

def wg_winrate(val,perm=False,rs=None):
    rates=[]
    for g in GK:
        ci,ki=PAIRS[g]; idx=np.concatenate([ci,ki]); v=val[idx]
        m=np.isfinite(v)
        if m.sum()<2: continue
        idx=idx[m]; v=v[m]; nc=int(np.isin(idx,ci).sum())
        if perm:
            lab=rs.permutation(np.r_[np.ones(nc),np.zeros(len(idx)-nc)])
        else:
            lab=np.isin(idx,ci).astype(float)
        cv=v[lab==1][:,None]; kv=v[lab==0][None,:]
        if not cv.size or not kv.size: continue
        w=(kv>cv).sum()+0.5*(kv==cv).sum(); rates.append(w/(cv.size*kv.size))
    return float(np.mean(rates)) if rates else np.nan, len(rates)

def wg_full(val,nboot=1000,nperm=500,lbl=""):
    obs,ng=wg_winrate(val)
    rs=np.random.default_rng(42)
    per_gene=[]
    for g in GK:
        ci,ki=PAIRS[g]; v=val[np.concatenate([ci,ki])]; lab=np.r_[np.ones(len(ci)),np.zeros(len(ki))]
        m=np.isfinite(v)
        if m.sum()<2: per_gene.append(np.nan); continue
        v=v[m]; lab=lab[m]; cv=v[lab==1][:,None]; kv=v[lab==0][None,:]
        per_gene.append(((kv>cv).sum()+0.5*(kv==cv).sum())/(cv.size*kv.size) if cv.size and kv.size else np.nan)
    pg=np.array(per_gene); pg=pg[np.isfinite(pg)]
    bs=pg[rs.integers(0,len(pg),size=(nboot,len(pg)))].mean(1)
    lo,hi=np.percentile(bs,[2.5,97.5])
    nul=np.array([wg_winrate(val,perm=True,rs=rs)[0] for _ in range(nperm)])
    p=(np.sum(np.abs(nul-0.5)>=abs(obs-0.5))+1)/(nperm+1)
    log(f"[{lbl}] win-rate {obs:.4f} excess {obs-0.5:+.4f} CI [{lo:.4f},{hi:.4f}] permP {p:.4f} ng {ng}")
    return dict(winrate=obs,excess=obs-0.5,lo=lo,hi=hi,p=p,n_genes=ng)

log("=== A1 ordered geometry ===")
a1_int1=wg_full(f5dn_present,lbl="A1 interval1 (anchor,pA)")
a1_int2=wg_full(int2,lbl="A1 interval2 (pA,pA+2000)")
frac1=(np.nanmean(f5dn_present[isc==1]),np.nanmean(f5dn_present[isc==0]))
frac2=(np.nanmean(int2[isc==1]),np.nanmean(int2[isc==0]))
with open("v32_A1_geometry.tsv","w") as fh:
    fh.write("interval\twinrate\texcess\tci_lo\tci_hi\tperm_p\tn_genes\tfrac_canon\tfrac_comp\n")
    fh.write(f"anchor->pA\t{a1_int1['winrate']:.5f}\t{a1_int1['excess']:.5f}\t{a1_int1['lo']:.5f}\t{a1_int1['hi']:.5f}\t{a1_int1['p']:.5f}\t{a1_int1['n_genes']}\t{frac1[0]:.4f}\t{frac1[1]:.4f}\n")
    fh.write(f"pA->pA+2000\t{a1_int2['winrate']:.5f}\t{a1_int2['excess']:.5f}\t{a1_int2['lo']:.5f}\t{a1_int2['hi']:.5f}\t{a1_int2['p']:.5f}\t{a1_int2['n_genes']}\t{frac2[0]:.4f}\t{frac2[1]:.4f}\n")

log("=== A2 exon-size dependence ===")
BANDS=[("<50",0,50),("50-150",50,150),("150-300",150,300),("300-1000",300,1000),(">1000",1000,1e12)]
a2=[]
for nm,lo,hi in BANDS:
    ind=((f5dn_present==1)&(f5dn_dist>=lo)&(f5dn_dist<hi)).astype(float)
    r=wg_full(ind,lbl=f"A2 band {nm}")
    fc=np.nanmean(ind[isc==1]); fk=np.nanmean(ind[isc==0])
    a2.append((nm,r,fc,fk))
with open("v32_A2_exonsize.tsv","w") as fh:
    fh.write("band\twinrate\texcess\tci_lo\tci_hi\tperm_p\tn_genes\tfrac_canon\tfrac_comp\n")
    for nm,r,fc,fk in a2:
        fh.write(f"{nm}\t{r['winrate']:.5f}\t{r['excess']:.5f}\t{r['lo']:.5f}\t{r['hi']:.5f}\t{r['p']:.5f}\t{r['n_genes']}\t{fc:.4f}\t{fk:.4f}\n")

log("=== A3 decoy architecture ===")
D=np.load("decoy_set.npz"); dpos=D["pos"]; dgene=D["gene"]; dac=D["acc_dist"]
gi=np.load("u1_window_cache.npz")["gene_idx"]
o=np.argsort(gi,kind="stable"); gs=gi[o]; cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gcanon=[]
for j in range(len(cut)-1):
    idx=o[cut[j]:cut[j+1]]; cc=idx[isc[idx]==1]; kk=idx[isc[idx]==0]
    if len(cc)==1 and len(kk)>=1: gcanon.append(int(cc[0]))
gcanon=np.array(gcanon)
assert dgene.max()<len(gcanon), f"dgene {dgene.max()} vs gcanon {len(gcanon)}"
dsub=rng.choice(len(dpos),min(DEC_SUB,len(dpos)),replace=False)
dec_up=np.full(len(dsub),np.nan); dec_dn=np.full(len(dsub),np.nan); dec_gene=[]
dec_bins=[];
for j,di in enumerate(dsub):
    ci=int(gcanon[int(dgene[di])])
    ch=chrom[ci]; st=strand[ci]
    pr=donor_profile(ch,int(dpos[di]),st)
    dec_up[j]=strong_in(pr,-SPAN,0); dec_dn[j]=strong_in(pr,1,SPAN+1)
    dec_gene.append(gene[ci])
    if pr is not None:
        d,h=pr; bi=np.digitize(d,BEDGES)-1; ok=(bi>=0)&(bi<len(BCEN))
        cnt=np.bincount(bi[ok],weights=h[ok].astype(float),minlength=len(BCEN))
        tot=np.bincount(bi[ok],minlength=len(BCEN)).astype(float)
        dec_bins.append(np.where(tot>0,cnt/np.maximum(tot,1)*1000.0,0.0))
    else: dec_bins.append(np.full(len(BCEN),np.nan))
    if (j+1)%10000==0: log(f"  decoy scan {j+1}/{len(dsub)}")
dec_gene=np.array(dec_gene); dec_bins=np.array(dec_bins)
def gene_avg_frac(val,genes):
    by=defaultdict(list)
    for v,g in zip(val,genes):
        if np.isfinite(v): by[g].append(v)
    r=[np.mean(x) for x in by.values() if x]
    r=np.array(r); rs=np.random.default_rng(1)
    bs=r[rs.integers(0,len(r),size=(1000,len(r)))].mean(1)
    return r.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),len(r)
grp={"canonical":(up2k[isc==1],gene[isc==1],dn2k[isc==1]),
     "competitor":(up2k[isc==0],gene[isc==0],dn2k[isc==0]),
     "decoy":(dec_up,dec_gene,dec_dn)}
with open("v32_A3_decoy_arch.tsv","w") as fh:
    fh.write("group\tside\tfrac\tci_lo\tci_hi\tn_genes\n")
    for nm,(uv,gg,dv) in grp.items():
        for side,vv in [("upstream_2kb",uv),("downstream_2kb",dv)]:
            m,lo,hi,ng=gene_avg_frac(vv,gg)
            fh.write(f"{nm}\t{side}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\t{ng}\n")
            log(f"[A3] {nm:11s} {side:15s} frac {m:.4f} [{lo:.4f},{hi:.4f}] ng {ng}")

def metag(bins_arr,sel):
    A=bins_arr[sel]; A=A[np.isfinite(A).all(1)]
    return A.mean(0), A.std(0,ddof=1)/math.sqrt(len(A)), len(A)
cm,cs,cn=metag(prof_bins,(isc==1)&prof_ok)
km,ks_,kn=metag(prof_bins,(isc==0)&prof_ok)
dm=np.nanmean(dec_bins,0); dstd=np.nanstd(dec_bins,0)/math.sqrt(np.isfinite(dec_bins[:,0]).sum())
with open("v32_F_metagene.tsv","w") as fh:
    fh.write("offset\tgroup\tdonor_per_kb\tse\tn\n")
    for j,o in enumerate(BCEN):
        fh.write(f"{o:.0f}\tcanonical\t{cm[j]:.4f}\t{cs[j]:.4f}\t{cn}\n")
        fh.write(f"{o:.0f}\tcompetitor\t{km[j]:.4f}\t{ks_[j]:.4f}\t{kn}\n")
        fh.write(f"{o:.0f}\tdecoy\t{dm[j]:.4f}\t{dstd[j]:.4f}\t{len(dsub)}\n")

S=list(csv.DictReader(open(SA),delimiter="\t"))
assert [r["gene"] for r in S]==list(gene)
sa_don=np.array([float(r["sa_don_sum"]) for r in S]); sa_acc=np.array([float(r["sa_acc_max"]) for r in S])
def auroc(pos,neg):
    if len(pos)<20 or len(neg)<20: return np.nan
    a=np.concatenate([pos,neg]); _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
    csum=np.cumsum(cnt); rank=((csum-cnt)+csum+1)/2.0; rk=rank[inv]
    R1=rk[:len(pos)].sum(); return (R1-len(pos)*(len(pos)+1)/2.0)/(len(pos)*len(neg))
qt=np.nanquantile(acc,[1/3,2/3]); tert=np.digitize(acc,qt)
with open("v32_C_sa_donor_tertile.tsv","w") as fh:
    fh.write("tertile\tsa_donor_auroc\tn\n")
    for k,nm in enumerate(("weak","medium","strong")):
        m=(tert==k)&np.isfinite(sa_don)
        au=auroc(sa_don[m&(isc==1)],sa_don[m&(isc==0)])
        fh.write(f"{nm}\t{au:.4f}\t{int(m.sum())}\n")
        log(f"[C] SpliceAI donor AUROC {nm} acceptor tertile = {au:.4f} (n {int(m.sum())})")

print("[out] v32_A1_geometry.tsv v32_A2_exonsize.tsv v32_A3_decoy_arch.tsv v32_F_metagene.tsv v32_C_sa_donor_tertile.tsv")
print(f"A1: interval1 (anchor,pA) excess {a1_int1['excess']:+.4f} (p {a1_int1['p']:.3f}) | interval2 (pA,+2000) excess {a1_int2['excess']:+.4f} (p {a1_int2['p']:.3f})")
print("A2 bands: "+" | ".join(f"{nm} {r['excess']:+.4f}(p{r['p']:.3f})" for nm,r,_,_ in a2))
