#!/usr/bin/env python3
# fig s3 rebuild data: paired-difference / within-pair z / anchor-referenced
import sys, csv
import numpy as np, pandas as pd
def log(*a): print(*a,file=sys.stderr,flush=True)

ro=pd.read_csv("s3_row_order.tsv",sep="\t")
long=pd.read_csv("s3_raster_long.tsv",sep="\t")
binned=pd.read_csv("s3_raster_binned.tsv",sep="\t")
feat=pd.read_csv("fig3_new_positional_features.tsv",sep="\t")[
     ["gene","is_canonical","chrom","center","strand","acc","anchor_3ss_pos","d_anchor_to_pa"]]
feat["class"]=np.where(feat.is_canonical==1,"real","competitor")
ro=ro.merge(feat[["gene","center","strand","class","acc","d_anchor_to_pa","anchor_3ss_pos"]],
            on=["gene","center","strand","class"],how="left",suffixes=("","_f"))
r2g={int(r.row_index):r.gene for r in ro.itertuples()}
r2c={int(r.row_index):r["class"] for _,r in ro.iterrows()}
NG=ro["gene"].nunique()
pair={}
for _,r in ro.iterrows():
    pair.setdefault(r.gene,{})[r["class"]]=int(r.row_index)

# panel 1: paired difference
up=long[long.pos_rel_pa<=0]
me_max=up[up.feature.isin(["me3","me5"])].groupby(["row_index","feature"]).score.max().unstack()
u1_dens=(up[(up.feature=="u1")&(up.score>=5.25)].groupby("row_index").size()
         .reindex(range(1,201),fill_value=0)/2.0)
rowval={}
for ri in range(1,201):
    rowval[ri]={"me3":me_max.loc[ri,"me3"] if ri in me_max.index else np.nan,
                "me5":me_max.loc[ri,"me5"] if ri in me_max.index else np.nan,
                "u1":float(u1_dens.loc[ri])}
FEATS=[("me3","MaxEnt 3'ss max (pA-2000..pA)"),("me5","MaxEnt 5'ss max (pA-2000..pA)"),
       ("u1","U1 density ≥5.25 /kb (pA-2000..pA)")]
frac_above={}
with open("c_s3_panel1.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["gene","feature","real","competitor","diff","rank"])
    for fk,_ in FEATS:
        rows=[]
        for g,pr in pair.items():
            if "real" not in pr or "competitor" not in pr: continue
            rv=rowval[pr["real"]][fk]; cv=rowval[pr["competitor"]][fk]
            if np.isnan(rv) or np.isnan(cv): continue
            rows.append((g,rv,cv,rv-cv))
        rows.sort(key=lambda t:t[3])
        diffs=np.array([x[3] for x in rows]); frac_above[fk]=float((diffs>0).mean())
        for rank,(g,rv,cv,df) in enumerate(rows):
            w.writerow([g,fk,f"{rv:.4f}",f"{cv:.4f}",f"{df:.4f}",rank])
        log(f"[P1] {fk}: fraction real>comp = {frac_above[fk]:.3f}  (n={len(rows)})")

# panel 2: within-gene-pair z
b2=binned.copy()
b2["gene"]=b2.row_index.map(r2g)
zrows=[]
for (g,fk),grp in b2.groupby(["gene","feature"]):
    v=grp.score_max.values; mu=v.mean(); sd=v.std()
    if sd<1e-9: sd=1.0
    for _,r in grp.iterrows():
        zrows.append((int(r.row_index),r["class"],int(r.bin_start),fk,(r.score_max-mu)/sd))
pd.DataFrame(zrows,columns=["row_index","class","bin_start","feature","z"]).to_csv(
    "c_s3_panel2.tsv",sep="\t",index=False)
log("[P2] within-pair z written")

# panel 3a: acc vs d_anchor_to_pa
ro[["row_index","gene","class","acc","d_anchor_to_pa"]].to_csv("c_s3_panel3a.tsv",sep="\t",index=False)
log("[P3a] anchor scatter written")

# panel 3b: me3 re-aligned on anchor
d_by_row={int(r.row_index):(r.d_anchor_to_pa if pd.notna(r.d_anchor_to_pa) else np.nan) for _,r in ro.iterrows()}
me3=long[long.feature=="me3"][["row_index","pos_rel_pa","score"]].copy()
me3["d"]=me3.row_index.map(d_by_row)
me3=me3[pd.notna(me3.d)]
me3["x_anchor"]=me3.pos_rel_pa+me3.d
me3=me3[(me3.x_anchor>=-500)&(me3.x_anchor<500)]
me3["xbin"]=((me3.x_anchor+500)//10).astype(int)*10-500
p3b=me3.groupby(["row_index","xbin"]).score.max().reset_index()
p3b["class"]=p3b.row_index.map(r2c)
p3b.rename(columns={"score":"me3_max"})[["row_index","class","xbin","me3_max"]].to_csv(
    "c_s3_panel3b.tsv",sep="\t",index=False)
log(f"[P3b] anchor-aligned me3 written ({p3b.row_index.nunique()} peaks with anchor in-window)")

FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
GTF="../train/data/annotation/gencode.v49.annotation.gtf"
Bm=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): Bm[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
me5d=np.load(ME5NPY,allow_pickle=True); me5d=me5d.item() if me5d.dtype==object else dict(me5d)
ks=np.frombuffer("".join(me5d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
ME5=np.full(4**9,np.nan); ME5[Bm[ks]@POW]=np.fromiter(me5d.values(),np.float64,len(me5d))
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
LO,HI,PAD=-2000,500,25; NPOS=HI-LO+1
gt_all=[]; sc_all=[]
chrom_map={(row.gene,int(row.center),row.strand):row.chrom for row in feat.itertuples()}
for _,r in ro.iterrows():
    ce=int(r.center); st=r.strand; ch=chrom_map.get((r.gene,ce,st))
    if ch is None: continue
    if st=="+": a,b=ce+LO-PAD, ce+HI+PAD+1
    else:       a,b=ce-HI-PAD, ce-LO+PAD+1
    seq=G.fetch(ch,a,b,st)
    if seq is None or len(seq)!=NPOS+2*PAD: continue
    tok=Bm[np.frombuffer(seq.encode("ascii"),np.uint8)]
    sw=np.lib.stride_tricks.sliding_window_view(tok,9); good=(sw>=0).all(1)
    code=np.where(good,sw@POW,0)
    for j in range(NPOS):
        idx=PAD+j; s9=idx-3
        if 0<=s9<len(good) and good[s9]:
            sc_all.append(ME5[code[s9]]); gt_all.append(seq[idx]=="G" and seq[idx+1]=="T")
sc_all=np.array(sc_all); gt_all=np.array(gt_all)
def topgt(frac):
    n=max(1,int(len(sc_all)*frac)); thr=np.sort(sc_all)[-n]; m=sc_all>=thr
    return gt_all[m].mean(), int(m.sum())
gt1,n1=topgt(0.01); gt01,n01=topgt(0.001)

import re
TID=re.compile(r'transcript_id "([^"]+)"')
tx={}
with open(GTF) as fh:
    for line in fh:
        if line[0]=="#": continue
        f=line.split("\t")
        if f[2]!="exon": continue
        m=TID.search(f[8])
        if not m: continue
        t=tx.setdefault(m.group(1),{"chrom":f[0],"strand":f[6],"ex":[]}); t["ex"].append((int(f[3])-1,int(f[4])))
donors=[]
for t in tx.values():
    ex=sorted(t["ex"]);
    if len(ex)<2: continue
    st=t["strand"]; ch=t["chrom"]
    for i in range(len(ex)-1):
        donors.append((ch, ex[i][1] if st=="+" else ex[i+1][0], st))
rng=np.random.default_rng(42); donors=[donors[i] for i in rng.choice(len(donors),min(8000,len(donors)),replace=False)]
gtD=[]; scD=[]
for ch,dp,st in donors:
    if st=="+": seq=G.fetch(ch,dp-3,dp+6,"+")
    else:       seq=G.fetch(ch,dp-6,dp+3,"-")
    if not seq or len(seq)!=9 or "N" in seq: continue
    tok=Bm[np.frombuffer(seq.encode("ascii"),np.uint8)]
    if (tok<0).any(): continue
    scD.append(ME5[tok@POW]); gtD.append(seq[3]=="G" and seq[4]=="T")
gtD=np.array(gtD); scD=np.array(scD)
print(f"REGISTER me5 top-1%  GT@3:4 = {gt1:.3f}  (n={n1})")
print(f"REGISTER me5 top-0.1% GT@3:4 = {gt01:.3f}  (n={n01})")
print(f"REGISTER GENCODE donors GT@3:4 = {gtD.mean():.3f} ; mean me5 = {np.nanmean(scD):.2f}  (n={len(gtD):,})")
print("PANEL1 fraction real>comp:", {k:round(v,3) for k,v in frac_above.items()})
print("[out] c_s3_panel1/2/3a/3b.tsv")
