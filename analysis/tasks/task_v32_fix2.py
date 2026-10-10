#!/usr/bin/env python3
# fix 2: length-match the d intervals and add a distance-to-nearest-strong-donor metric
import sys, csv, math
import numpy as np
from collections import defaultdict
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
POS="fig3_new_positional_features.tsv"; ND=8.0; PAD=12; CAP=8000; GWIN=2000; DEC_SUB=30000

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

def hit_offsets(chrom,center,strand,span):
    a,b=center-span-PAD, center+span+PAD
    seq=G.fetch(chrom,a,b,strand)
    if not seq or len(seq)<2*span: return None
    tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    sc=np.full(sw.shape[0],-99.0); sc[good]=ME5[sw[good]@POW]
    d=-(span+PAD)+np.arange(sc.size)+3
    m=(d>=-span)&(d<=span)&(sc>=ND)
    return d[m]

# load peaks
R=list(csv.DictReader(open(POS),delimiter="\t"))
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
chrom=[r["chrom"] for r in R]; center=[int(r["center"]) for r in R]; strand=[r["strand"] for r in R]
def gf(col): return np.array([float(r[col]) if r[col] not in("","nan") else np.nan for r in R])
dpa=gf("d_anchor_to_pa")
NP=len(R)

PAIRS=None
def build_pairs(y=isc,g=gene):
    by=defaultdict(lambda:([],[]))
    for i in range(len(g)): by[g[i]][0 if y[i]==1 else 1].append(i)
    return {k:(np.array(c),np.array(k2)) for k,(c,k2) in by.items() if c and k2}
PAIRS=build_pairs(); GK=list(PAIRS.keys()); NG=len(GK)

def wg_winrate(val,perm=False,rs=None):
    rates=[]
    for g in GK:
        ci,ki=PAIRS[g]; idx=np.concatenate([ci,ki]); v=val[idx]; m=np.isfinite(v)
        if m.sum()<2: continue
        idx=idx[m]; v=v[m]; nc=int(np.isin(idx,ci).sum())
        lab=rs.permutation(np.r_[np.ones(nc),np.zeros(len(idx)-nc)]) if perm else np.isin(idx,ci).astype(float)
        cv=v[lab==1][:,None]; kv=v[lab==0][None,:]
        if not cv.size or not kv.size: continue
        rates.append(((kv>cv).sum()+0.5*(kv==cv).sum())/(cv.size*kv.size))
    return (float(np.mean(rates)) if rates else np.nan), len(rates)
def wg_full(val,lbl="",nboot=1000,nperm=500):
    obs,ng=wg_winrate(val); rs=np.random.default_rng(42)
    pg=[]
    for g in GK:
        ci,ki=PAIRS[g]; v=val[np.concatenate([ci,ki])]; lab=np.r_[np.ones(len(ci)),np.zeros(len(ki))]
        m=np.isfinite(v)
        if m.sum()<2: pg.append(np.nan); continue
        v=v[m]; lab=lab[m]; cv=v[lab==1][:,None]; kv=v[lab==0][None,:]
        pg.append(((kv>cv).sum()+0.5*(kv==cv).sum())/(cv.size*kv.size) if cv.size and kv.size else np.nan)
    pg=np.array(pg); pg=pg[np.isfinite(pg)]
    bs=pg[rs.integers(0,len(pg),size=(nboot,len(pg)))].mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
    nul=np.array([wg_winrate(val,True,rs)[0] for _ in range(nperm)])
    p=(np.sum(np.abs(nul-0.5)>=abs(obs-0.5))+1)/(nperm+1)
    log(f"[{lbl}] wr {obs:.4f} exc {obs-0.5:+.4f} CI[{lo:.4f},{hi:.4f}] p {p:.4f} ng {ng}")
    return dict(winrate=obs,excess=obs-0.5,lo=lo,hi=hi,p=p,n_genes=ng)

# scan d (length-matched)
pres_up=np.full(NP,np.nan); pres_dn=np.full(NP,np.nan)
dist_up=np.full(NP,np.nan); dist_dn=np.full(NP,np.nan)
ncap=0
for i in range(NP):
    d=dpa[i]
    if not np.isfinite(d) or d<1: continue
    if d>CAP: d=CAP; ncap+=1
    d=int(d)
    off=hit_offsets(chrom[i],center[i],strand[i],d)
    if off is None: continue
    up=off[(off>=-d)&(off<0)]; dn=off[(off>0)&(off<=d)]
    pres_up[i]=1.0 if up.size else 0.0; pres_dn[i]=1.0 if dn.size else 0.0
    dist_up[i]=(-np.min(np.abs(up))) if up.size else -d
    dist_dn[i]=(-np.min(dn)) if dn.size else -d
    if (i+1)%15000==0: log(f"  D scan {i+1}/{NP}")
log(f"D scan done; d capped at {CAP} for {ncap} peaks")

log("=== D length-matched, PRESENCE ===")
d_pres_up=wg_full(pres_up,"D int1 (anchor,pA) presence")
d_pres_dn=wg_full(pres_dn,"D int2 (pA,pA+d) presence, LENGTH-MATCHED")
log("=== D length-matched, DISTANCE-to-nearest-donor (competitor closer) ===")
d_dist_up=wg_full(dist_up,"D int1 (anchor,pA) distance")
d_dist_dn=wg_full(dist_dn,"D int2 (pA,pA+d) distance, LENGTH-MATCHED")
with open("v32_D_lengthmatched.tsv","w") as fh:
    fh.write("metric\tinterval\twinrate\texcess\tci_lo\tci_hi\tperm_p\tn_genes\n")
    for nm,iv,r in [("presence","anchor->pA",d_pres_up),("presence","pA->pA+d (matched)",d_pres_dn),
                    ("distance","anchor->pA",d_dist_up),("distance","pA->pA+d (matched)",d_dist_dn)]:
        fh.write(f"{nm}\t{iv}\t{r['winrate']:.5f}\t{r['excess']:.5f}\t{r['lo']:.5f}\t{r['hi']:.5f}\t{r['p']:.5f}\t{r['n_genes']}\n")

# scan g (distance, fixed 2kb upstream)
def upstream_dist(chrom,center,strand):
    off=hit_offsets(chrom,center,strand,GWIN)
    if off is None: return np.nan
    up=off[(off>=-GWIN)&(off<0)]
    return float(np.min(np.abs(up))) if up.size else float(GWIN)
gd=np.full(NP,np.nan)
for i in range(NP):
    gd[i]=upstream_dist(chrom[i],center[i],strand[i])
    if (i+1)%20000==0: log(f"  G scan {i+1}/{NP}")
D=np.load("decoy_set.npz"); dpos=D["pos"]; dgene=D["gene"]
gi=np.load("u1_window_cache.npz")["gene_idx"]
o=np.argsort(gi,kind="stable"); gs=gi[o]; cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gcanon=[]
for j in range(len(cut)-1):
    idx=o[cut[j]:cut[j+1]]; cc=idx[isc[idx]==1]; kk=idx[isc[idx]==0]
    if len(cc)==1 and len(kk)>=1: gcanon.append(int(cc[0]))
gcanon=np.array(gcanon)
dsub=rng.choice(len(dpos),min(DEC_SUB,len(dpos)),replace=False)
dec_d=np.full(len(dsub),np.nan); dec_g=[]
for j,di in enumerate(dsub):
    ci=int(gcanon[int(dgene[di])]); dec_d[j]=upstream_dist(chrom[ci],int(dpos[di]),strand[ci]); dec_g.append(gene[ci])
    if (j+1)%10000==0: log(f"  G decoy {j+1}/{len(dsub)}")
dec_g=np.array(dec_g)
def gene_med(val,genes):
    by=defaultdict(list)
    for v,g in zip(val,genes):
        if np.isfinite(v): by[g].append(v)
    per=np.array([np.median(x) for x in by.values() if x]); rs=np.random.default_rng(1)
    bs=per[rs.integers(0,len(per),size=(1000,len(per)))].mean(1)
    return np.median(per),per.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),len(per)
with open("v32_G_distance.tsv","w") as fh:
    fh.write("group\tmedian_dist\tmean_of_gene_medians\tci_lo\tci_hi\tn_genes\n")
    for nm,val,gg in [("canonical",gd[isc==1],gene[isc==1]),("competitor",gd[isc==0],gene[isc==0]),("decoy",dec_d,dec_g)]:
        md,mn,lo,hi,ng=gene_med(val,gg)
        fh.write(f"{nm}\t{md:.1f}\t{mn:.1f}\t{lo:.1f}\t{hi:.1f}\t{ng}\n")
        log(f"[G-dist] {nm:11s} median gene-median dist {md:.0f} nt  mean {mn:.0f} [{lo:.0f},{hi:.0f}] ng {ng}")

print("[out] v32_D_lengthmatched.tsv v32_G_distance.tsv")
print(f"D presence: anchor->pA {d_pres_up['excess']:+.4f}(p{d_pres_up['p']:.3f}) | pA->pA+d MATCHED {d_pres_dn['excess']:+.4f}(p{d_pres_dn['p']:.3f})")
print(f"D distance: anchor->pA {d_dist_up['excess']:+.4f}(p{d_dist_up['p']:.3f}) | pA->pA+d MATCHED {d_dist_dn['excess']:+.4f}(p{d_dist_dn['p']:.3f})")
