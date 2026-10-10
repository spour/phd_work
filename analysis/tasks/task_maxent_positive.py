#!/usr/bin/env python3
# task a (maxent): positive-control calibration of maxent 3'ss and 5'ss against annotated
import sys, csv, re
import numpy as np
from collections import defaultdict
sys.path.insert(0,"scripts")
from extract_splicing_features import init_maxentscan, load_me5_lookup
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; ME5NPY="../train/splice_train/data/me5_lookup.npy"
GTF="../train/data/annotation/gencode.v49.annotation.gtf"
N_MAX=60000; CHROMS={f"chr{c}" for c in list(range(1,23))+["X","Y"]}

s5,s3,m5,m3=init_maxentscan()
B=np.full(256,-1,np.int64)
for ch,v in zip(b"ACGT",(0,1,2,3)): B[ch]=v
POW=(4**np.arange(8,-1,-1)).astype(np.int64)
me5d=load_me5_lookup(ME5NPY)
ks=np.frombuffer("".join(me5d.keys()).encode("ascii"),np.uint8).reshape(-1,9)
ME5=np.full(4**9,np.nan); ME5[B[ks]@POW]=np.fromiter(me5d.values(),np.float64,len(me5d))
def sc5(seq):
    if not seq or len(seq)!=9: return np.nan
    t=B[np.frombuffer(seq.encode("ascii"),np.uint8)]
    if (t<0).any(): return np.nan
    return float(ME5[int(t@POW)])
def sc3(seq):
    if not seq or len(seq)!=23 or "N" in seq: return np.nan
    v=s3(seq)
    return float(np.nanmax(np.asarray(v,float))) if hasattr(v,"__len__") else float(v)

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

# parse gencode protein_coding exons
log("[gtf] parsing protein_coding exons ...")
TID=re.compile(r'transcript_id "([^"]+)"'); GNM=re.compile(r'gene_name "([^"]+)"')
tx=defaultdict(lambda:{"chrom":None,"strand":None,"gene":None,"ex":[]})
n=0
with open(GTF) as fh:
    for line in fh:
        if line[0]=="#": continue
        f=line.split("\t")
        if f[2]!="exon": continue
        if 'transcript_type "protein_coding"' not in f[8]: continue
        if f[0] not in CHROMS: continue
        m=TID.search(f[8]);
        if not m: continue
        t=tx[m.group(1)]; t["chrom"]=f[0]; t["strand"]=f[6]
        g=GNM.search(f[8]); t["gene"]=g.group(1) if g else m.group(1)
        t["ex"].append((int(f[3]),int(f[4])))
        n+=1
        if n%3_000_000==0: log(f"  {n:,} exon lines")
log(f"[gtf] {len(tx):,} protein_coding transcripts")

donors={}; acceptors={}; introns=[]
for t in tx.values():
    ex=sorted(t["ex"]); st=t["strand"]; ch=t["chrom"]; gn=t["gene"]
    if len(ex)<2: continue
    order=ex if st=="+" else ex[::-1]
    for i in range(len(order)-1):
        s0,e0=order[i]; s1,e1=order[i+1]
        if st=="+":
            donors[(ch,e0,"+")]=gn; acceptors[(ch,s1,"+")]=gn
            lo,hi=e0+1,s1-1
        else:
            donors[(ch,s0,"-")]=gn; acceptors[(ch,e1,"-")]=gn
            lo,hi=e1+1,s0-1
        if hi-lo+1>=60: introns.append((ch,lo,hi,st,gn))
log(f"[gtf] {len(donors):,} donors ; {len(acceptors):,} acceptors ; {len(introns):,} introns >=60nt")

# extract + score annotated sites
def don_seq(ch,pos,st):  return G.fetch(ch,pos-3,pos+6,"+") if st=="+" else G.fetch(ch,pos-7,pos+2,"-")
def acc_seq(ch,pos,st):  return G.fetch(ch,pos-21,pos+2,"+") if st=="+" else G.fetch(ch,pos-3,pos+20,"-")

def annotated(sites,seqfn,klen,anch_idx,anch,scorer):
    keys=list(sites.keys())
    if len(keys)>N_MAX: keys=[keys[i] for i in rng.choice(len(keys),N_MAX,replace=False)]
    seqs=[]; anch_ok=0
    for ch,pos,st in keys:
        sq=seqfn(ch,pos,st)
        if sq and len(sq)==klen and "N" not in sq:
            seqs.append(sq)
            if sq[anch_idx:anch_idx+2]==anch: anch_ok+=1
    reg=anch_ok/max(len(seqs),1)
    seqs=[s for s in seqs if s[anch_idx:anch_idx+2]==anch]
    sc=np.array([scorer(s) for s in seqs]); sc=sc[np.isfinite(sc)]
    return sc,reg,len(sc)

def intronic_bg(n,klen,seqfn_random,scorer):
    ilen=np.array([hi-lo+1-40 for (_,lo,hi,_,_) in introns],float); ilen=np.clip(ilen,1,None)
    p=ilen/ilen.sum(); pick=rng.choice(len(introns),size=int(n*1.3),replace=True,p=p)
    out=[]
    for j in pick:
        ch,lo,hi,st,gn=introns[j]; a=lo+20; b=hi-20-klen
        if b<=a: continue
        pos=int(rng.integers(a,b+1)); sq=seqfn_random(ch,pos,st)
        if sq and len(sq)==klen and "N" not in sq:
            v=scorer(sq)
            if np.isfinite(v): out.append(v)
        if len(out)>=n: break
    return np.array(out)

def process(name,sites,seqfn,klen,anch_idx,anch,scorer,rand_klen_seqfn,allk_scores,xlo,xhi):
    sc,reg,N=annotated(sites,seqfn,klen,anch_idx,anch,scorer)
    log(f"[{name}] annotated {anch}-anchored register {reg*100:.1f}% ; positive n={N} ; "
        f"median {np.median(sc):.2f} mean {sc.mean():.2f}")
    bg=intronic_bg(N,klen,rand_klen_seqfn,scorer)
    log(f"[{name}] random intronic n={len(bg)} ; median {np.median(bg):.2f}")
    SWEEP=np.round(np.arange(xlo,xhi+1e-9,0.25),3)
    def tail(x,t): return float(np.mean(x>=t))
    sens=np.array([tail(sc,t) for t in SWEEP]); itl=np.array([tail(bg,t) for t in SWEEP])
    utl=np.array([tail(allk_scores,t) for t in SWEEP])
    enr=np.where(itl>0,sens/itl,np.nan); enru=np.where(utl>0,sens/utl,np.nan)
    def sc_at(tg): return float(np.quantile(sc,1-tg))
    S={tg:sc_at(tg) for tg in (0.50,0.75,0.90)}
    def at(t):
        j=int(np.argmin(np.abs(SWEEP-t))); return sens[j],enr[j],enru[j],itl[j]
    with open(f"{name}_positive_calibration.tsv","w",newline="") as fh:
        w=csv.writer(fh,delimiter="\t")
        w.writerow(["threshold","sensitivity","intron_tail","allkmer_tail","enrich_vs_intron","enrich_vs_allkmer"])
        for j,t in enumerate(SWEEP):
            w.writerow([f"{t:.3f}",f"{sens[j]:.6f}",f"{itl[j]:.6f}",f"{utl[j]:.6f}",f"{enr[j]:.4f}",f"{enru[j]:.4f}"])
        w.writerow([]); w.writerow([f"# register_{anch}_frac={reg:.4f}",f"n_pos={N}",f"n_intron={len(bg)}"])
        for tg in (0.50,0.75,0.90):
            t=S[tg]; s_,e_,eu_,it_=at(t); w.writerow([f"# sens{int(tg*100)}",f"thr={t:.3f}",f"enrich_intron={e_:.3f}",f"sens={s_:.4f}"])
        s8,e8,eu8,it8=at(8.0); w.writerow([f"# at_8.0",f"sens={s8:.4f}",f"enrich_intron={e8:.3f}",f"enrich_allkmer={eu8:.3f}"])
    edges=np.arange(-35.0,xhi+2.001,0.5); ctr=(edges[:-1]+edges[1:])/2
    def hist(x):
        x=x[np.isfinite(x)]; h,_=np.histogram(x,bins=edges); return h/max(h.sum(),1)
    with open(f"{name}_positive_dist.tsv","w",newline="") as fh:
        w=csv.writer(fh,delimiter="\t"); w.writerow(["score","annotated","intronic","allkmer"])
        for c,a1,a2,a3 in zip(ctr,hist(sc),hist(bg),hist(allk_scores)):
            w.writerow([f"{c:.3f}",f"{a1:.6f}",f"{a2:.6f}",f"{a3:.6f}"])
    print(f"\n================  MaxEnt {name.upper()} POSITIVE-CONTROL CALIBRATION  ================")
    print(f"register: {anch} at k-mer idx {anch_idx}-{anch_idx+1} in {reg*100:.1f}% of annotated sites")
    print(f"positive: {N:,} canonical {anch} sites ; median {np.median(sc):.2f}  mean {sc.mean():.2f}")
    print(f"intronic bg: {len(bg):,} ; median {np.median(bg):.2f}")
    print(f"{'sensitivity':>11} {'score>=':>8} {'enrich_vs_intron':>17} {'enrich_vs_allkmer':>18}")
    for tg in (0.50,0.75,0.90):
        t=S[tg]; s_,e_,eu_,it_=at(t); print(f"{int(tg*100):>10}% {t:8.2f} {e_:17.1f} {eu_:18.1f}")
    print(f"  >=8.0 : sensitivity {s8*100:5.1f}%  enrich_vs_intron {e8:.1f}  enrich_vs_allkmer {eu8:.1f}")
    return dict(name=name,reg=reg,N=N,S=S,at8=(s8,e8,eu8),median=float(np.median(sc)))

def rand9(ch,pos,st):  return G.fetch(ch,pos-1,pos+8,st)
def rand23(ch,pos,st): return G.fetch(ch,pos-1,pos+22,st)

ALL5=ME5[np.isfinite(ME5)]
log("[all-23mer] sampling uniform random 23-mers ...")
rb=np.random.default_rng(7); bases=np.array(list("ACGT"))
r23=["".join(bases[rb.integers(0,4,23)]) for _ in range(200000)]
ALL3=np.array([sc3(s) for s in r23]); ALL3=ALL3[np.isfinite(ALL3)]

res5=process("maxent5",donors,don_seq,9,3,"GT",sc5,rand9,ALL5,-2.0,12.0)
res3=process("maxent3",acceptors,acc_seq,23,18,"AG",sc3,rand23,ALL3,-2.0,14.0)
print("\n[out] maxent5_positive_calibration.tsv/_dist.tsv ; maxent3_positive_calibration.tsv/_dist.tsv")
