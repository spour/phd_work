#!/usr/bin/env python3
# four contained checks on the competitor class (see task brief)
import csv, numpy as np, bisect
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK="#1A1A1A"; CAN="#0072B2"; COMP="#E69F00"; NUL="#8A8F98"; GRN="#009E73"; PUR="#9467bd"
rng=np.random.default_rng(42)

# load
R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t"))
NP=len(R)
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
chrom=np.array([r["chrom"] for r in R]); cen=np.array([int(r["center"]) for r in R]); strd=np.array([r["strand"] for r in R])
acc=np.array([float(r["acc"]) if r["acc"] not in("","nan") else np.nan for r in R])
comp=np.load("peak_compartment.npy")
LAB=["TERM","INTERNAL","INTRON","OTHER"]
Zg=np.load("geom_cache.npz"); pres_up=Zg["pres_up"]; pres_dn=Zg["pres_dn"]; max5=Zg["max5"]
u1exc=np.load("u1_window_cache.npz")["excess_525"]
At=np.load("atlas_peak_cache.npz"); a_dist=At["dist"]; a_tpm=At["tpm"]; a_nsamp=At["nsamp"]

tc=np.where(strd=="+",cen,-cen).astype(float)
gmap=defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)

dn=np.zeros(NP,bool); up=np.zeros(NP,bool); canon_not3=set()
for g,idxs in gmap.items():
    idx=np.array(idxs); c=idx[isc[idx]==1]
    if len(c)!=1 or len(idx)<2: continue
    c=int(c[0])
    if tc[c]!=tc[idx].max(): canon_not3.add(g)
    for i in idx:
        if isc[i]==0:
            if tc[i]>tc[c]: dn[i]=True
            else: up[i]=True
DN=np.where(dn)[0]

OUT=open("competitor_class_split.tsv","w"); W=lambda *a: OUT.write("\t".join(str(x) for x in a)+"\n")
def hdr(s): print("\n"+s); W("#",s)

# item 1: compartment breakdown
hdr("ITEM 1: compartment breakdown (0=TERM 1=INTERNAL 2=INTRON 3=OTHER)")
W("set","n","TERM","INTERNAL","INTRON","OTHER")
sets=[("downstream_competitors",dn),("all_competitors",isc==0),("canonicals",isc==1)]
for nm,m in sets:
    ct=[int((comp[m]==k).sum()) for k in range(4)]; n=int(m.sum())
    frac=" ".join(f"{LAB[k]} {ct[k]} ({ct[k]/n*100:.1f}%)" for k in range(4))
    print(f"  {nm:24s} n={n:6d} | {frac}")
    W(nm,n,*[f"{ct[k]}({ct[k]/n*100:.1f}%)" for k in range(4)])
term_comp=(isc==0)&(comp==0)
dn_term=dn&(comp==0)
print(f"  downstream competitors that are TERM: {int(dn_term.sum())}/{int(dn.sum())} = {dn_term.sum()/dn.sum()*100:.1f}%")
print(f"  TERM competitors that are downstream: {int(dn_term.sum())}/{int(term_comp.sum())} = {dn_term.sum()/term_comp.sum()*100:.1f}%")
W("overlap_downstream_and_TERM", int(dn_term.sum()), "of_downstream", int(dn.sum()), "of_TERMcomp", int(term_comp.sum()))

# item 2: atlas usage
hdr("ITEM 2: PolyASite 2.0 per-site usage (nearest cluster same strand, <=25 nt)")
TOL=25; matched=a_dist<=TOL
gene_tot=np.zeros(NP)
for g,idxs in gmap.items():
    idx=np.array(idxs); s=a_tpm[idx].sum();
    for i in idx: gene_tot[i]=s
ufrac=np.where(gene_tot>0, a_tpm/np.where(gene_tot>0,gene_tot,1), np.nan)
W("class","n_total","n_matched","match_frac","ufrac_Q1","ufrac_med","ufrac_Q3","med_nsamp","frac_ge10","frac_ge25","frac_ge40")
for nm,m in [("downstream_competitors",dn),("upstream_competitors",up),("canonicals",isc==1)]:
    mm=m&matched; nt=int(m.sum()); nmatch=int(mm.sum())
    uf=ufrac[mm]; uf=uf[np.isfinite(uf)]; ns=a_nsamp[mm]
    q1,md,q3=np.percentile(uf,[25,50,75]) if len(uf) else (np.nan,)*3
    g10,g25,g40=[float(np.mean(uf>=t)) for t in (.10,.25,.40)] if len(uf) else (np.nan,)*3
    print(f"  {nm:24s} matched {nmatch}/{nt} ({nmatch/nt*100:.1f}%) | ufrac Q1/med/Q3 {q1:.3f}/{md:.3f}/{q3:.3f} | med nsamp {np.median(ns):.0f} | >=10/25/40% {g10:.2f}/{g25:.2f}/{g40:.2f}")
    W(nm,nt,nmatch,f"{nmatch/nt:.3f}",f"{q1:.3f}",f"{md:.3f}",f"{q3:.3f}",f"{np.median(ns):.0f}",f"{g10:.3f}",f"{g25:.3f}",f"{g40:.3f}")
print("  NOTE: usage available only for matched peaks; unmatched (majority of upstream comps) have no PolyASite cluster -> effectively unused/uncalled.")

# item 3: split + re-run
hdr("ITEM 3: split competitor class -> TANDEM (TERM comp) vs INTRONIC (INTRON comp); re-run core results")
print("  [EXPLORATORY] split is post-hoc on the same data; NOT a held-out test.")
tandem=(isc==0)&(comp==0); intronic=(isc==0)&(comp==2)
def sub_report(nm,m):
    genes={gene[i] for i in np.where(m)[0]}
    print(f"  {nm:9s}: {int(m.sum())} peaks, {len(genes)} genes")
    return genes
gT=sub_report("TANDEM",tandem); gI=sub_report("INTRONIC",intronic)
print(f"  genes with BOTH subclasses: {len(gT & gI)}")
W("subclass","peaks","genes"); W("TANDEM",int(tandem.sum()),len(gT)); W("INTRONIC",int(intronic.sum()),len(gI))
W("genes_with_both", len(gT & gI))

def build_pairs(comp_mask):
    P={}
    for g,idxs in gmap.items():
        idx=np.array(idxs); c=idx[isc[idx]==1]; k=idx[(isc[idx]==0)&comp_mask[idx]]
        if len(c)>=1 and len(k)>=1: P[g]=(c,k)
    return P
def wg(val,P,higher):
    GK=list(P);
    def rate(perm=False,rs=None):
        rr=[]
        for g in GK:
            ci,ki=P[g]; idx=np.concatenate([ci,ki]); v=val[idx]; mm=np.isfinite(v)
            if mm.sum()<2: continue
            v=v[mm]; nc=int(mm[:len(ci)].sum())
            lab=rs.permutation(np.r_[np.ones(nc),np.zeros(len(v)-nc)]) if perm else np.r_[np.ones(len(ci)),np.zeros(len(ki))][mm]
            cv=v[lab==1]; kv=v[lab==0]
            if not len(cv) or not len(kv): continue
            cv=cv[:,None]; kv=kv[None,:]
            if higher=='comp': w=(kv>cv).sum()+0.5*(kv==cv).sum()
            else:              w=(cv>kv).sum()+0.5*(cv==kv).sum()
            rr.append(w/(cv.shape[0]*kv.shape[1]))
        return rr
    obs_rates=rate(); obs=float(np.mean(obs_rates)); ng=len(obs_rates)
    pg=np.array(obs_rates); bs=pg[rng.integers(0,len(pg),size=(1000,len(pg)))].mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
    rs=np.random.default_rng(7); nperm=500
    nul=np.array([np.mean(rate(True,rs)) for _ in range(nperm)])
    p=(np.sum(np.abs(nul-0.5)>=abs(obs-0.5))+1)/(nperm+1)
    return dict(wr=obs,exc=obs-0.5,lo=lo,hi=hi,p=p,ng=ng)

P_all=build_pairs(isc==0); P_T=build_pairs(tandem); P_I=build_pairs(intronic)
metrics=[("a_acc_acceptor",acc,"canon"),
         ("b_geom_presUP(anchor->pA)",pres_up,"comp"),
         ("b_geom_presDN(pA->pA+d)",pres_dn,"comp"),
         ("c_u1_excess",u1exc,"comp")]
W("metric","subclass","winrate","excess","ci_lo","ci_hi","perm_p","n_genes","orient(>0.5=)")
res={}
for mnm,val,hi in metrics:
    for snm,P in [("ALL_comp",P_all),("TANDEM",P_T),("INTRONIC",P_I)]:
        r=wg(val,P,hi); res[(mnm,snm)]=r
        ostr="canonical higher" if hi=="canon" else "competitor higher"
        print(f"  {mnm:26s} {snm:9s} wr {r['wr']:.4f} exc {r['exc']:+.4f} CI[{r['lo']:.4f},{r['hi']:.4f}] p {r['p']:.4f} ng {r['ng']}")
        W(mnm,snm,f"{r['wr']:.4f}",f"{r['exc']:+.4f}",f"{r['lo']:.4f}",f"{r['hi']:.4f}",f"{r['p']:.4f}",r['ng'],ostr)

strong=(acc>=8).astype(float); term8=((acc>=8)&(max5<8)).astype(float)
def p_at1_sub(score,comp_mask):
    hits=[]; chances=[]
    for g,idxs in gmap.items():
        idx=np.array(idxs); c=idx[isc[idx]==1]
        if len(c)!=1: continue
        k=idx[(isc[idx]==0)&comp_mask[idx]]
        if len(k)<1: continue
        sub=np.r_[c,k]
        pick=sub[np.lexsort((acc[sub],score[sub]))[-1]]
        hits.append(1.0 if isc[pick]==1 else 0.0); chances.append(1.0/len(sub))
    h=np.array(hits); b=h[rng.integers(0,len(h),size=(2000,len(h)))].mean(1)
    return h.mean(),np.percentile(b,2.5),np.percentile(b,97.5),float(np.mean(chances)),len(h)
hdr("ITEM 3d: P@1 rules per subclass (restricted to canonical + that subclass's competitors)")
W("rule","subclass","p_at1","ci_lo","ci_hi","chance","n_genes")
for rnm,score in [("acceptor_only(acc>=8)",strong),("terminal_like@8",term8)]:
    for snm,cm in [("ALL_comp",isc==0),("TANDEM",tandem),("INTRONIC",intronic)]:
        p1,lo,hi,ch,ng=p_at1_sub(score,cm)
        print(f"  {rnm:22s} {snm:9s} P@1 {p1:.4f} [{lo:.4f},{hi:.4f}] chance {ch:.4f} ng {ng}")
        W(rnm,snm,f"{p1:.4f}",f"{lo:.4f}",f"{hi:.4f}",f"{ch:.4f}",ng)

rT=res[("b_geom_presUP(anchor->pA)","TANDEM")]; rI=res[("b_geom_presUP(anchor->pA)","INTRONIC")]
print("\n  MECHANISTIC PREDICTION (exon definition = strong 5'ss in (anchor,pA)):")
print(f"    INTRONIC presUP excess {rI['exc']:+.4f} CI[{rI['lo']-0.5:+.4f},{rI['hi']-0.5:+.4f}] p{rI['p']:.4f}")
print(f"    TANDEM   presUP excess {rT['exc']:+.4f} CI[{rT['lo']-0.5:+.4f},{rT['hi']-0.5:+.4f}] p{rT['p']:.4f}")
OUT.close()

print("\nITEM 4: neighbouring-gene / readthrough check")
O4=open("competitor_readthrough.tsv","w"); W4=lambda *a: O4.write("\t".join(str(x) for x in a)+"\n")
bodies=defaultdict(list)
for L in open("../train/data/pc_genes.bed"):
    p=L.rstrip("\n").split("\t")
    if len(p)<4: continue
    bodies[p[0]].append((int(p[1]),int(p[2]),p[3],p[5] if len(p)>5 else "."))
bstart={c:np.array([x[0] for x in sorted(v)]) for c,v in bodies.items()}
bsorted={c:sorted(v) for c,v in bodies.items()}
tes=defaultdict(list)
for c,v in bodies.items():
    for s,e,nm,st in v: tes[c].append(( e if st=="+" else s, nm))
tesarr={c:(np.array([x[0] for x in sorted(v)]),[x[1] for x in sorted(v)]) for c,v in tes.items()}

def in_other_body(i):
    v=bsorted.get(chrom[i]);
    if not v: return False
    c=cen[i]; g=gene[i]
    for s,e,nm,st in v:
        if s<=c<=e and nm!=g: return True
    return False
def near_other_tes(i,d=500):
    arr=tesarr.get(chrom[i]);
    if not arr: return False
    pos,names=arr; c=cen[i]; g=gene[i]
    j=bisect.bisect_left(pos,c-d)
    while j<len(pos) and pos[j]<=c+d:
        if names[j]!=g: return True
        j+=1
    return False

W4("set","n","frac_in_other_gene_body","frac_within500_of_other_TES","frac_OTHER_compartment")
for nm,m in [("downstream_competitors",dn),("canonicals",isc==1)]:
    ii=np.where(m)[0]
    fb=np.mean([in_other_body(i) for i in ii]); ft=np.mean([near_other_tes(i) for i in ii])
    foth=float(np.mean(comp[ii]==3))
    print(f"  {nm:24s} n={len(ii):5d} | in other gene body {fb*100:.1f}% | within 500nt of other TES {ft*100:.1f}% | OTHER compartment {foth*100:.1f}%")
    W4(nm,len(ii),f"{fb:.4f}",f"{ft:.4f}",f"{foth:.4f}")
O4.close()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.4))
groups=[("downstream\ncompetitors",dn),("all\ncompetitors",isc==0),("canonicals",isc==1)]
cols=[CAN,NUL,GRN,COMP]
x=np.arange(len(groups)); bottom=np.zeros(len(groups))
for k in range(4):
    vals=np.array([ (comp[m]==k).mean() for _,m in groups])*100
    ax1.bar(x,vals,bottom=bottom,color=cols[k],width=0.62,edgecolor="white",linewidth=0.6,label=LAB[k])
    bottom+=vals
ax1.set_xticks(x); ax1.set_xticklabels([g[0] for g in groups],fontsize=9)
ax1.set_ylabel("% of peaks"); ax1.set_ylim(0,100); ax1.set_title("Compartment breakdown",color=INK,fontsize=11)
ax1.legend(frameon=False,fontsize=8,ncol=2,loc="lower center",bbox_to_anchor=(0.5,-0.32))
for s in ("top","right"): ax1.spines[s].set_visible(False)
labels=["ALL_comp","INTRONIC","TANDEM"]; colB=[NUL,COMP,CAN]
xs=np.arange(len(labels))
for j,lab in enumerate(labels):
    r=res[("b_geom_presUP(anchor->pA)",lab)]
    ax2.errorbar(j,r['exc'],yerr=[[r['exc']-(r['lo']-0.5)],[(r['hi']-0.5)-r['exc']]],fmt="o",color=colB[j],
                 capsize=4,ms=8,elinewidth=1.5,mec=INK,mew=0.6)
    ax2.annotate(f"{r['exc']:+.3f}\np={r['p']:.3f}",(j,r['exc']),textcoords="offset points",xytext=(10,0),fontsize=8,va="center")
ax2.axhline(0,color=INK,lw=0.8,ls="--")
ax2.set_xticks(xs); ax2.set_xticklabels(labels,fontsize=9); ax2.set_xlim(-0.5,2.6)
ax2.set_ylabel("win-rate excess (competitor higher)")
ax2.set_title("Exon definition: strong 5'ss in (anchor, pA)",color=INK,fontsize=11)
for s in ("top","right"): ax2.spines[s].set_visible(False)
fig.suptitle("Competitor class split: distal-tandem (TERM) vs intronic (pCPA)",color=INK,fontsize=12,y=1.02)
fig.tight_layout()
fig.savefig("competitor_split.pdf",bbox_inches="tight"); fig.savefig("competitor_split.png",dpi=150,bbox_inches="tight")
print("\n[out] competitor_class_split.tsv  competitor_readthrough.tsv  competitor_split.pdf/png")
