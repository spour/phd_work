#!/usr/bin/env python3
# task c data tables for the one-layer figure 3
import sys, csv
import numpy as np
def log(*a): print(*a,file=sys.stderr,flush=True)
rng=np.random.default_rng(42)
FASTA="../train/data/hg38.fa"; U1NPY="../train/splice_train/data/u1_lookup.npy"; META="fig3_ucube_meta.tsv"
WIN=500; PRIMARY=5.25; PAS_W=50; DN_W=50; SPAN=2000; BIN=100; SUB=2500
PAS12=["AATAAA","ATTAAA","TATAAA","AGTAAA","AAGAAA","AATACA","CATAAA","GATAAA","AATATA","AATGAA","TTTAAA","ACTAAA"]
PAS_STRONG=("AATAAA","ATTAAA")

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
def arch(pos,ch,st):
    up=G.fetch(ch,pos,pos+WIN,"-") if st=="-" else G.fetch(ch,pos-WIN,pos,"+")
    dn=G.fetch(ch,pos-DN_W,pos,"-") if st=="-" else G.fetch(ch,pos,pos+DN_W,"+")
    ps=at=gu=np.nan
    if up: pw=up[-PAS_W:]; ps=1.0 if any(h in pw for h in PAS_STRONG) else 0.0
    if dn:
        dt=B[np.frombuffer(dn.encode("ascii"),np.uint8)]; dg=dt[dt>=0]
        if dg.size: at=((dg==0)|(dg==3)).mean(); di=dg[:-1]*4+dg[1:]; gu=np.isin(di,(11,14)).mean() if di.size else np.nan
    return ps,at,gu

# peaks / within-gene
Z=np.load("u1_window_cache.npz"); isc=Z["isc"]; gi=Z["gene_idx"]; u1d525=Z["dens_5.25"]; excess525=Z["excess_525"]
gc_real=np.load("gc_up500.npy"); comp_lab=np.load("peak_compartment.npy")
rows=[]; acc=[]
with open(META) as fh:
    for r in csv.DictReader(fh,delimiter="\t"): rows.append((r["chrom"],int(r["center"]),r["strand"])); acc.append(float(r["acc"]))
acc=np.array(acc); NP=len(rows)
chrom=np.array([x[0] for x in rows]); center=np.array([x[1] for x in rows]); strand=np.array([x[2] for x in rows])
o=np.argsort(gi,kind="stable"); gs=gi[o]; cut=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
gmemb,gcanon=[],[]
for j in range(len(cut)-1):
    idx=o[cut[j]:cut[j+1]]; c=idx[isc[idx]==1]; k=idx[isc[idx]==0]
    if len(c)==1 and len(k)>=1: gmemb.append(idx);gcanon.append(int(c[0]))
NG=len(gmemb); canon_arr=np.array(gcanon)
comp_flat=np.concatenate([m[m!=c] for m,c in zip(gmemb,gcanon)])
comp_gene=np.concatenate([np.full((m!=c).sum(),i) for i,(m,c) in enumerate(zip(gmemb,gcanon))])
compK=comp_lab[comp_flat]; compC=comp_lab[canon_arr][comp_gene]

def ga(h):
    ok=np.isfinite(h); v=h[ok]
    if len(v)<5: return (np.nan,np.nan,np.nan,int(ok.sum()))
    b=v[rng.integers(0,len(v),size=(4000,len(v)))].mean(1)
    return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5),int(ok.sum())
def wg_half(Aval,sel):
    vc=Aval[canon_arr][comp_gene]; vk=Aval[comp_flat]
    win=(vk>vc)&sel; tie=(vk==vc)&sel; nn=np.bincount(comp_gene,sel.astype(float),NG)
    w=np.bincount(comp_gene,win,NG); t=np.bincount(comp_gene,tie,NG)
    return np.where(nn>0,(w+0.5*t)/nn,np.nan)

# panel d: acceptor within-gene
same=compK==compC
strata=[("marginal",np.ones(len(comp_flat),bool)),("same-compartment",same),("TERM-TERM",same&(compK==0))]
with open("c_panelD_acceptor_wg.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["stratum","obs","null","excess","ci_lo","ci_hi","n_genes"])
    for nm,sel in strata:
        m,lo,hi,ng=ga(wg_half(acc,sel)); w.writerow([nm,f"{m:.4f}","0.5000",f"{m-0.5:+.4f}",f"{lo:.4f}",f"{hi:.4f}",ng])
        log(f"[D] {nm}: {m:.4f} [{lo:.4f},{hi:.4f}]")

# panels e & g: decoy arms
D=np.load("decoy_set.npz"); dpos=D["pos"]; dgene=D["gene"]; dcomp=D["comp"]; du1=D["u1"]; dgc=D["gc"]; dexc=D["exc"]
log("arch at decoys...")
dps=np.empty(len(dpos)); dat=np.empty(len(dpos)); dgu=np.empty(len(dpos))
for i in range(len(dpos)):
    g=dgene[i]; ch=chrom[gcanon[g]]; st=strand[gcanon[g]]
    dps[i],dat[i],dgu[i]=arch(int(dpos[i]),ch,st)
log("arch at real peaks...")
rps=np.empty(NP); rat=np.empty(NP); rgu=np.empty(NP)
for i in range(NP):
    rps[i],rat[i],rgu[i]=arch(center[i],chrom[i],strand[i])
    if (i+1)%30000==0: log(f"  {i+1}")

by_gene_dec={}; by_gene_cmp={}
for feat in ["u1","gc","exc","ps","at","gu"]: by_gene_dec[feat]=[[] for _ in range(NG)]
for k in range(len(dpos)):
    g=dgene[k]
    by_gene_dec["u1"][g].append(du1[k]);by_gene_dec["gc"][g].append(dgc[k]);by_gene_dec["exc"][g].append(dexc[k])
    by_gene_dec["ps"][g].append(dps[k]);by_gene_dec["at"][g].append(dat[k]);by_gene_dec["gu"][g].append(dgu[k])
    by_gene_cmp.setdefault(g,[]).append(dcomp[k])
REAL={"u1":u1d525,"gc":gc_real,"exc":excess525,"ps":rps,"at":rat,"gu":rgu}
def comp_vs_dec(feat):
    real=REAL[feat]; dec=by_gene_dec[feat]; half=np.full(NG,np.nan)
    for g in range(NG):
        dv=np.array(dec[g],float); cm=np.array(by_gene_cmp.get(g,[])); ok=np.isfinite(dv)
        if ok.sum()==0: continue
        cv=real[cm[ok]]; dvv=dv[ok]; okk=np.isfinite(cv)
        if okk.sum()==0: continue
        dd=cv[okk]-dvv[okk]; w=(dd>0).sum(); t=(dd==0).sum(); half[g]=(w+0.5*t)/okk.sum()
    return half
def canon_vs_dec(feat):
    real=REAL[feat]; dec=by_gene_dec[feat]; half=np.full(NG,np.nan)
    for g in range(NG):
        dv=np.array(dec[g],float); dv=dv[np.isfinite(dv)]
        if dv.size==0: continue
        rv=real[gcanon[g]]
        if not np.isfinite(rv): continue
        dd=dv-rv; w=(dd>0).sum(); t=(dd==0).sum(); half[g]=(w+0.5*t)/dv.size
    return half

ELAB={"ps":"strong PAS hexamer","gu":"GU/UG downstream (DSE)","at":"A/T downstream"}
with open("c_panelE_competence.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["feature","obs","null","excess","ci_lo","ci_hi","n_genes"])
    for feat in ["ps","gu","at"]:
        m,lo,hi,ng=ga(comp_vs_dec(feat)); w.writerow([ELAB[feat],f"{m:.4f}","0.5000",f"{m-0.5:+.4f}",f"{lo:.4f}",f"{hi:.4f}",ng])
        log(f"[E] {ELAB[feat]}: {m:.4f} [{lo:.4f},{hi:.4f}]")

GLAB={"u1":"U1 density @5.25","exc":"u1_excess","gc":"GC content"}
with open("c_panelG_decoy.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["feature","arm","obs","null","excess","ci_lo","ci_hi"])
    for feat in ["u1","exc","gc"]:
        m,lo,hi,_=ga(wg_half(REAL[feat],np.ones(len(comp_flat),bool)))
        w.writerow([GLAB[feat],"canonical vs competitor",f"{m:.4f}","0.5000",f"{m-0.5:+.4f}",f"{lo:.4f}",f"{hi:.4f}"])
        m,lo,hi,_=ga(canon_vs_dec(feat))
        w.writerow([GLAB[feat],"canonical vs decoy (placebo)",f"{m:.4f}","0.5000",f"{m-0.5:+.4f}",f"{lo:.4f}",f"{hi:.4f}"])
        m,lo,hi,_=ga(comp_vs_dec(feat))
        w.writerow([GLAB[feat],"competitor vs decoy (decisive)",f"{m:.4f}","0.5000",f"{m-0.5:+.4f}",f"{lo:.4f}",f"{hi:.4f}"])
        log(f"[G] {GLAB[feat]} done")

# panel f: metagene +/-2kb
NB=2*SPAN//BIN; offs=(np.arange(NB)*BIN)-SPAN+BIN/2
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
            if h.size: accU[bi]+=np.nansum(h)/(BIN/1000.0)
            if np.isfinite(g).any(): accG[bi]+=np.nanmean(g)
            cnt[bi]+=1
    return accU/np.maximum(cnt,1), accG/np.maximum(cnt,1)
canon_idx=np.where(isc==1)[0]; comp_idx=np.where(isc==0)[0]
def samp(idx,n): return idx if len(idx)<=n else rng.choice(idx,n,replace=False)
cs=samp(canon_idx,SUB); ks=samp(comp_idx,SUB); ds=samp(np.arange(len(dpos)),SUB)
pos_c=[(chrom[i],center[i],strand[i]) for i in cs]
pos_k=[(chrom[i],center[i],strand[i]) for i in ks]
pos_d=[(chrom[gcanon[dgene[i]]],int(dpos[i]),strand[gcanon[dgene[i]]]) for i in ds]
log("metagene canonical..."); Uc,Gc=metagene(pos_c)
log("metagene competitor..."); Uk,Gk=metagene(pos_k)
log("metagene decoy..."); Ud,Gd=metagene(pos_d)
with open("c_panelF_metagene.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["offset","class","u1_per_kb","gc"])
    for bi in range(NB):
        w.writerow([int(offs[bi]),"canonical",f"{Uc[bi]:.3f}",f"{Gc[bi]:.4f}"])
        w.writerow([int(offs[bi]),"competitor",f"{Uk[bi]:.3f}",f"{Gk[bi]:.4f}"])
        w.writerow([int(offs[bi]),"decoy",f"{Ud[bi]:.3f}",f"{Gd[bi]:.4f}"])
log("[F] metagene written")

zg=np.load("fig3_denswin.npz"); dwd=zg["dw_donor"]; td=list(zg["thr_donor"]); wins=zg["wins"].astype(float); y=zg["is_canonical"].astype(int)
NYBIN=14; n1=max((y==1).sum(),1); n0=max((y==0).sum(),1); eps=1.0/max(n1,n0)
def bcshow(dens_full,tv):
    dens=dens_full/(wins[None,:]/1000.0); hi=float(np.nanpercentile(dens,99.0))
    edges=np.linspace(0.0,hi if hi>0 else 1.0,NYBIN+1); ctr=(edges[:-1]+edges[1:])/2.0; out=[]
    for wi,wv in enumerate(wins):
        bi=np.clip(np.searchsorted(edges,dens[:,wi],side="right")-1,0,NYBIN-1)
        for j in range(NYBIN):
            m=bi==j; n=int(m.sum())
            if n==0: continue
            pc=float((m&(y==1)).sum())/n1; pk=float((m&(y==0)).sum())/n0
            out.append((float(wv),float(ctr[j]),float(np.log2((pc+eps)/(pk+eps))),n,tv))
    return out
with open("c_panelC_donor.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["window","yval","log2ratio","n","strong_thr"])
    for tv in (4.0,8.0):
        for r in bcshow(dwd[:,td.index(tv),:],tv): w.writerow([f"{r[0]:.1f}",f"{r[1]:.4f}",f"{r[2]:.4f}",r[3],f"{tv:.1f}"])
log("[C] donor bcshow @4 and @8 written")

with open("c_class_counts.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["cls","n"])
    w.writerow(["Canonical",int((isc==1).sum())]); w.writerow(["Competitor",int((isc==0).sum())])
print("[out] c_panelD/E/F/G/C tables + c_class_counts.tsv")
