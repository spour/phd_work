#!/usr/bin/env python3
# fig s3 scan: 200 gene-paired peaks, dense per-position me3/me5/u1 over pa-2000..pa+500
import sys, csv, os
import numpy as np
sys.path.insert(0,"scripts")
from extract_splicing_features import init_maxentscan, load_me5_lookup, load_u1_lookup
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(42)
FASTA="../train/data/hg38.fa"
ME5NPY="../train/splice_train/data/me5_lookup.npy"; U1NPY="../train/splice_train/data/u1_lookup.npy"
FEAT="fig3_new_positional_features.tsv"
LO,HI=-2000,500; PAD=25; NPOS=HI-LO+1
ME3_HIT,ME5_HIT,U1_HIT=8.0,8.0,5.25

score5,score3,m5,m3=init_maxentscan()
me5d=load_me5_lookup(ME5NPY); u1d=load_u1_lookup(U1NPY)
Bn=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): Bn[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
def dict_to_arr(dd,neg):
    ks=np.frombuffer("".join(dd.keys()).encode("ascii"),np.uint8).reshape(-1,9)
    idx=Bn[ks]@POW; arr=np.full(4**9,np.nan); vals=np.fromiter(dd.values(),np.float64,len(dd))
    arr[idx]=(-vals if neg else vals); return arr
ME5=dict_to_arr(me5d,False); U1=dict_to_arr(u1d,True)
log("lookups -> arrays done")

class Faidx:
    _T=str.maketrans("ACGTNacgtn","TGCANTGCAN")
    def __init__(s,fa):
        s.fh=open(fa,"rb"); s.idx={}
        for L in open(fa+".fai"):
            p=L.split(); s.idx[p[0]]=(int(p[1]),int(p[2]),int(p[3]),int(p[4]))
    def fetch(s,c,a,b,st):
        c=c if c in s.idx else c.lstrip("chr")
        if c not in s.idx: return None
        Ln,off,lb,lw=s.idx[c]
        if a<0 or b>Ln: return None
        bp=lambda p: off+(p//lb)*lw+(p%lb); s.fh.seek(bp(a)); raw=s.fh.read(bp(b)-bp(a))
        q=raw.replace(b"\n",b"").replace(b"\r",b"")[:b-a].decode("ascii").upper()
        return q.translate(s._T)[::-1] if st=="-" else q
G=Faidx(FASTA)

# selection
bygene={}
with open(FEAT) as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        try: dpa=float(r["d_anchor_to_pa"])
        except: dpa=np.nan
        bygene.setdefault(r["gene"],[]).append((int(r["is_canonical"]),r["chrom"],int(r["center"]),r["strand"],dpa))
elig=sorted([g for g,v in bygene.items() if any(x[0]==1 for x in v) and any(x[0]==0 for x in v)])
pick=sorted(rng.choice(np.array(elig,dtype=object),100,replace=False).tolist())
sel=[]
for g in pick:
    v=bygene[g]; canon=[x for x in v if x[0]==1][0]; comps=[x for x in v if x[0]==0]
    comp=comps[int(rng.integers(len(comps)))]
    sel.append((g,canon,comp,comp[4]))
sel.sort(key=lambda t:(np.isnan(t[3]),t[3]))
log(f"selected {len(sel)} genes")

# scan one peak: dense me3/me5/u1 at every position, transcript-oriented
def scan(chrom,center,strand):
    if strand=="+": a,b=center+LO-PAD, center+HI+PAD+1
    else:           a,b=center-HI-PAD, center-LO+PAD+1
    seq=G.fetch(chrom,a,b,strand)
    if seq is None or len(seq)!=(NPOS+2*PAD): return None
    tok=Bn[np.frombuffer(seq.encode("ascii"),np.uint8)]
    base=PAD
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    code=np.where(good,sw@POW,0)
    me5=np.full(NPOS,np.nan); u1=np.full(NPOS,np.nan); me3=np.full(NPOS,np.nan)
    for j in range(NPOS):
        idx=base+j
        s9=idx-3
        if 0<=s9<len(good) and good[s9]:
            me5[j]=ME5[code[s9]]; u1[j]=U1[code[s9]]
        s23=idx-20
        if s23>=0 and idx+3<=len(seq):
            k=seq[s23:idx+3]
            if len(k)==23 and "N" not in k: me3[j]=score3(k)
    return me5,u1,me3,seq

pos_rel=np.arange(LO,HI+1)
rows_long=[]; dropped=0; row_order=[]
scan_cache=[]
for gi,(g,canon,comp,dpa) in enumerate(sel):
    keep=True; recs={}
    for cls,pk in (("real",canon),("competitor",comp)):
        _,chrom,center,strand,pdpa=pk
        r=scan(chrom,center,strand)
        if r is None: keep=False; break
        recs[cls]=(pk,r)
    if not keep: dropped+=1; continue
    scan_cache.append((g,recs,dpa))
NG=len(scan_cache)
for k,(g,recs,dpa) in enumerate(scan_cache):
    for block,cls in ((0,"real"),(1,"competitor")):
        pk,(me5,u1,me3,seq)=recs[cls]
        _,chrom,center,strand,pdpa=pk
        ridx=block*NG + (k+1)
        pid=f"{g}:{center}:{strand}:{cls}"
        row_order.append((ridx,g,cls,pid,center,strand,dpa,pdpa))
        for feat,arr,thr in (("me3",me3,ME3_HIT),("me5",me5,ME5_HIT),("u1",u1,U1_HIT)):
            for j in range(NPOS):
                s=arr[j]
                if np.isnan(s): continue
                rows_long.append((pid,g,cls,ridx,int(pos_rel[j]),feat,float(s),bool(s>=thr)))
log(f"kept {NG} genes ({2*NG} peaks), dropped {dropped} genes for boundary")

# sanity checks (register validation)
def topfrac(feat,check):
    vals=[(r[6],r[0],r[4]) for r in rows_long if r[5]==feat]
    vals.sort(reverse=True); top=vals[:max(1,len(vals)//100)]
    ok=0
    seqmap={}
    for g,recs,dpa in scan_cache:
        for cls in ("real","competitor"):
            pk,(me5,u1,me3,seq)=recs[cls]; seqmap[f"{g}:{pk[2]}:{pk[3]}:{cls}"]=seq
    for sc,pid,pos in top:
        seq=seqmap[pid]; idx=PAD+(pos-LO)
        ok+=check(seq,idx)
    return ok/len(top)
gt5=topfrac("me5",lambda seq,idx: seq[idx]=="G" and seq[idx+1]=="T")
ag3=topfrac("me3",lambda seq,idx: seq[idx-2]=="A" and seq[idx-1]=="G")
m5v=np.array([r[6] for r in rows_long if r[5]=="me5"]); u1v=np.array([r[6] for r in rows_long if r[5]=="u1"])
corr=np.corrcoef(m5v,u1v)[0,1]
print(f"SANITY me5 top-1% GT@3:4 frac : {gt5:.3f}")
print(f"SANITY me3 top-1% AG@18:19 frac: {ag3:.3f}")
print(f"SANITY corr(me5,u1) all pos    : {corr:.3f}")
if gt5<0.5 or ag3<0.8 or not (0.05<corr<0.95):
    print("REGISTER CHECK FAILED — stopping, not writing output"); sys.exit(1)

# write row order
with open("s3_row_order.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["row_index","gene","class","peak_id","center","strand","gene_sort_d_anchor_to_pa","peak_d_anchor_to_pa"])
    for r in sorted(row_order): w.writerow(r)
with open("s3_raster_long.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["peak_id","gene","class","row_index","pos_rel_pa","feature","score","hit"])
    for r in rows_long: w.writerow([r[0],r[1],r[2],r[3],r[4],r[5],f"{r[6]:.4f}","TRUE" if r[7] else "FALSE"])
NBIN=(HI-LO)//10
def binof(p): return min(max((p-LO)//10,0),NBIN-1)
binmax={}; binhit={}
for r in rows_long:
    key=(r[3],r[5],binof(r[4]))
    if key not in binmax or r[6]>binmax[key]: binmax[key]=r[6]
    binhit[key]=binhit.get(key,False) or r[7]
rowmeta={x[0]:(x[3],x[1],x[2]) for x in row_order}
with open("s3_raster_binned.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["peak_id","gene","class","row_index","bin_start","feature","score_max","hit_any"])
    for (ridx,feat,b),sc in sorted(binmax.items()):
        pid,g,cls=rowmeta[ridx]; w.writerow([pid,g,cls,ridx,LO+b*10,feat,f"{sc:.4f}","TRUE" if binhit[(ridx,feat,b)] else "FALSE"])
print(f"[out] s3_row_order.tsv s3_raster_long.tsv s3_raster_binned.tsv ; kept {NG} genes, dropped {dropped}")
