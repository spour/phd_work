#!/usr/bin/env python3
# b4.2: is u1 the same feature as maxent 5'ss, scored twice?
import sys
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; U1NPY="../train/splice_train/data/u1_lookup.npy"
ME5NPY="../train/splice_train/data/me5_lookup.npy"; META="fig3_ucube_meta.tsv"; WIN=500

B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
def load(npy):
    a=np.load(npy,allow_pickle=True); d=a.item() if a.dtype==object else dict(a)
    kb=np.frombuffer("".join(d.keys()).encode("ascii"),np.uint8).reshape(-1,9); ki=B[kb]@POW
    arr=np.empty(4**9); arr[ki]=np.fromiter(d.values(),np.float64,len(d)); return arr
SCORE=-load(U1NPY); ME5=load(ME5NPY)

# cube-derived counts (no rescan)
z=np.load("fig3_ucube.npz"); uc=z["cube"]; mc=z["me5cube"]; edges=z["edges"]
lo,hi=int(np.searchsorted(edges,-500)),int(np.searchsorted(edges,0))
uthr=list(z["thrs"]); mthr=list(z["me5_thrs"])
u1c8_cube=uc[:,lo:hi,3+uthr.index(8.0)].sum(1)
me5c8=mc[:,lo:hi,3+mthr.index(8.0)].sum(1)
Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; gi=Z["gene_idx"]
u1d525=Z["dens_5.25"]; u1d8=Z["dens_8.0"]; u1c8_cache=u1d8*0.5
NP=len(isc)

# faidx-swap validation: cache u1 count(>=8) vs pipeline ucube
match=np.mean(np.abs(u1c8_cache-u1c8_cube)<1e-9)
log(f"[validate] faidx U1 count(>=8) vs pysam ucube: exact-match fraction {match:.6f} "
    f"(max abs diff {np.abs(u1c8_cache-u1c8_cube).max():.3g})")

def spearman(x,y):
    import numpy as np
    ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    return np.corrcoef(rx,ry)[0,1]
def pearson(x,y):
    ok=np.isfinite(x)&np.isfinite(y); return np.corrcoef(x[ok],y[ok])[0,1]

print("\n=== B4.2a  correlation U1 vs MaxEnt5 (pA-500..pA), per peak ===")
for nm,u in [("U1 density @5.25",u1d525),("U1 count @8.0",u1c8_cache)]:
    print(f"  {nm:20s} vs MaxEnt5 count>=8 : Spearman {spearman(u,me5c8):+.3f}  Pearson {pearson(u,me5c8):+.3f}")

# positional overlap (needs positions -> rescan)
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
        bp=lambda p: off+(p//lb)*lw+(p%lb)
        s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st=="-" else q
G=Faidx(FASTA)
rows=[]
import csv
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["chrom"],int(r["center"]),r["strand"]))
u1hits=exact=near3=me5hits=0
for i,(c,ctr,st) in enumerate(rows):
    seq=G.fetch(c,ctr,ctr+WIN,"-") if st=="-" else G.fetch(c,ctr-WIN,ctr,"+")
    if not seq: continue
    tok=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
    if tok.shape[0]<9: continue
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    code=np.where(good, sw@POW, 0)
    u1h=good&(SCORE[code]>=8.0); m5h=good&(ME5[code]>=8.0)
    nu=int(u1h.sum());
    if nu==0: continue
    m5d=m5h.copy()
    for s3 in range(1,4):
        m5d[s3:]|=m5h[:-s3]; m5d[:-s3]|=m5h[s3:]
    u1hits+=nu; exact+=int((u1h&m5h).sum()); near3+=int((u1h&m5d).sum()); me5hits+=int(m5h.sum())
    if (i+1)%20000==0: log(f"  overlap {i+1:,}")
print(f"\n=== B4.2a  positional overlap of U1>=8 hits with MaxEnt5>=8 ===")
print(f"  U1>=8 hits {u1hits:,} ; MaxEnt5>=8 hits {me5hits:,}")
print(f"  same 9-mer position (U1>=8 AND me5>=8): {exact/u1hits:.3f}")
print(f"  MaxEnt5>=8 within +-3 nt of a U1>=8 hit: {near3/u1hits:.3f}")

# within-gene machinery
order=np.argsort(gi,kind="stable"); gs=gi[order]
cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb=[];gcanon=[]
for j in range(len(cut)-1):
    idx=order[cut[j]:cut[j+1]]; cc=idx[isc[idx]==1]; kk=idx[isc[idx]==0]
    if len(cc)==1 and len(kk)>=1: gmemb.append(idx);gcanon.append(int(cc[0]))
NG=len(gmemb)
def per_gene_excl(val,canon):
    out=np.full(NG,np.nan)
    for i,(m,c) in enumerate(zip(gmemb,canon)):
        vc=val[c]; vk=val[m[m!=c]]; d=vk-vc; nz=d!=0
        if nz.any(): out[i]=(d[nz]>0).mean()
    return out
def ga_ci(pg):
    pg=pg[np.isfinite(pg)]; b=pg[rng.integers(0,len(pg),size=(2000,len(pg)))].mean(1)
    return pg.mean(),np.percentile(b,2.5),np.percentile(b,97.5),len(pg)
def perm_null(val):
    return np.mean([np.nanmean(per_gene_excl(val,[int(m[rng.integers(len(m))]) for m in gmemb])) for _ in range(100)])

print("\n=== B4.2b  MaxEnt5 density(>=8) within-gene, gene-averaged tie-excluded ===")
m,lo_,hi_,n=ga_ci(per_gene_excl(me5c8,gcanon))
print(f"  MaxEnt5 count>=8 : GA tie-excl {m:.4f}  95% [{lo_:.4f},{hi_:.4f}]  ({n:,} genes)  null~{perm_null(me5c8):.3f}")
m,lo_,hi_,n=ga_ci(per_gene_excl(u1d525,gcanon))
print(f"  U1 density @5.25 : GA tie-excl {m:.4f}  95% [{lo_:.4f},{hi_:.4f}]  ({n:,} genes)  null~{perm_null(u1d525):.3f}  [for reference]")

print("\n=== B4.2c  does U1 survive conditioning on MaxEnt5 ? ===")
win=info=0; wz=iz=0
for m,c in zip(gmemb,gcanon):
    ks=m[m!=c]; me5eq=me5c8[ks]==me5c8[c]
    ks=ks[me5eq]
    if ks.size==0: continue
    d=u1d525[ks]-u1d525[c]; nz=d!=0
    win+=int((d[nz]>0).sum()); info+=int(nz.sum())
    z0=(me5c8[ks]==0)&(me5c8[c]==0); d0=u1d525[ks[z0]]-u1d525[c]; nz0=d0!=0
    wz+=int((d0[nz0]>0).sum()); iz+=int(nz0.sum())
print(f"  matched on MaxEnt5 count (equal): U1@5.25 pair-pooled win-rate {win/max(info,1):.4f}  ({info:,} informative pairs)")
print(f"    of which both MaxEnt5=0 (no strong donor either side): {wz/max(iz,1):.4f}  ({iz:,} pairs)")
ok=np.isfinite(u1d525)&np.isfinite(me5c8)
bta=np.polyfit(me5c8[ok],u1d525[ok],1); resid=u1d525-(bta[0]*me5c8+bta[1])
m,lo_,hi_,n=ga_ci(per_gene_excl(resid,gcanon))
print(f"  U1@5.25 residualized on MaxEnt5 count: GA tie-excl {m:.4f}  95% [{lo_:.4f},{hi_:.4f}]  null~{perm_null(resid):.3f}")
