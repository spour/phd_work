#!/usr/bin/env python3
# split the model-false-positive competitor class and find what sequence separates a real
import csv, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK="#1A1A1A"; CAN="#0072B2"; COMP_C="#E69F00"; NUL="#8A8F98"; GRN="#009E73"; PUR="#9467bd"
NFOLD=5; RIDGE=1.0; THR=8.46
rng=np.random.default_rng(42)

R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t")); NP=len(R)
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
cen=np.array([int(r["center"]) for r in R]); strd=np.array([r["strand"] for r in R])
acc=np.array([float(r["acc"]) if r["acc"] not in("","nan") else np.nan for r in R])
comp=np.load("peak_compartment.npy")
F1=np.load("hypothesis_test_features.npy")
Zg=np.load("geom_cache.npz"); max_up=Zg["max_up"]; max_dn=Zg["max_dn"]; max5=Zg["max5"]
At=np.load("atlas_peak_cache.npz"); a_dist=At["dist"]; a_tpm=At["tpm"]

tc=np.where(strd=="+",cen,-cen).astype(float)
gmap=defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)
dn=np.zeros(NP,bool)
for g,idxs in gmap.items():
    idx=np.array(idxs); c=idx[isc[idx]==1]
    if len(c)!=1: continue
    c=int(c[0])
    for i in idx:
        if isc[i]==0 and tc[i]>tc[c]: dn[i]=True
gene_tot=np.zeros(NP)
for g,idxs in gmap.items():
    idx=np.array(idxs); s=a_tpm[idx].sum()
    for i in idx: gene_tot[i]=s
ufrac=np.where(gene_tot>0,a_tpm/np.where(gene_tot>0,gene_tot,1),0.0)
matched=a_dist<=25

presUP=np.where(np.isfinite(max_up),(max_up>=THR).astype(float),np.nan)
presDN=np.where(np.isfinite(max_dn),(max_dn>=THR).astype(float),np.nan)

# subclass masks
CLEAN_FP=(isc==0)&(~matched)&(comp==2)
REAL_MINOR=(isc==0)&matched&(ufrac>=0.10)
CLEAN_FP_ND=CLEAN_FP&(~dn)
leftover=(isc==0)&(~CLEAN_FP)&(~REAL_MINOR)
SUBS=[("ALL_comp",isc==0),("CLEAN_FP",CLEAN_FP),("CLEAN_FP_nodown",CLEAN_FP_ND),("REAL_MINOR",REAL_MINOR)]

OUT=open("competitor_subclass_analysis.tsv","w"); W=lambda *a: OUT.write("\t".join(str(x) for x in a)+"\n")
def hdr(s): print("\n"+s); W("#",s)
hdr("SUBCLASS DEFINITIONS (competitors only; EXPLORATORY post-hoc split, not held out)")
W("subclass","n_peaks","n_genes")
for nm,m in SUBS+[("leftover",leftover)]:
    ng=len({gene[i] for i in np.where(m)[0]})
    print(f"  {nm:16s} {int(m.sum()):6d} peaks  {ng:5d} genes")
    W(nm,int(m.sum()),ng)

# within-gene win-rate scaffold
def build_pairs(cm):
    P={}
    for g,idxs in gmap.items():
        idx=np.array(idxs); c=idx[isc[idx]==1]; k=idx[(isc[idx]==0)&cm[idx]]
        if len(c)>=1 and len(k)>=1: P[g]=(c,k)
    return P
def wg(val,P,higher):
    GK=list(P)
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
            w=((kv>cv).sum()+0.5*(kv==cv).sum()) if higher=="comp" else ((cv>kv).sum()+0.5*(cv==kv).sum())
            rr.append(w/(cv.shape[0]*kv.shape[1]))
        return rr
    obs=rate(); wr=float(np.mean(obs)); ng=len(obs)
    pg=np.array(obs); bs=pg[rng.integers(0,len(pg),size=(1000,len(pg)))].mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
    rs=np.random.default_rng(7); nul=np.array([np.mean(rate(True,rs)) for _ in range(500)])
    p=(np.sum(np.abs(nul-0.5)>=abs(wr-0.5))+1)/501
    return dict(wr=wr,exc=wr-0.5,lo=lo,hi=hi,p=p,ng=ng)

hdr("WITHIN-GENE WIN-RATE: acceptor + ordered geometry (strong 5'ss>=8.46), canonical vs subclass")
W("metric","subclass","winrate","excess","ci_lo","ci_hi","perm_p","n_genes","orient")
WM={}
for mnm,val,hi in [("acc_acceptor",acc,"canon"),("geom_presUP(anchor,pA)",presUP,"comp"),("geom_presDN(pA,pA+d)_matched",presDN,"comp")]:
    for snm,m in SUBS:
        r=wg(val,build_pairs(m),hi); WM[(mnm,snm)]=r
        print(f"  {mnm:28s} {snm:16s} wr {r['wr']:.4f} exc {r['exc']:+.4f} CI[{r['lo']:.4f},{r['hi']:.4f}] p {r['p']:.4f} ng {r['ng']}")
        W(mnm,snm,f"{r['wr']:.4f}",f"{r['exc']:+.4f}",f"{r['lo']:.4f}",f"{r['hi']:.4f}",f"{r['p']:.4f}",r['ng'],"canon_higher" if hi=="canon" else "comp_higher")

# p@1 rules per subclass
strong=(acc>=8).astype(float); term8=((acc>=8)&(max5<8)).astype(float)
def p1_rule(score,cm):
    hits=[]; ch=[]
    for g,idxs in gmap.items():
        idx=np.array(idxs); c=idx[isc[idx]==1]
        if len(c)!=1: continue
        k=idx[(isc[idx]==0)&cm[idx]]
        if len(k)<1: continue
        sub=np.r_[c,k]; pick=sub[np.lexsort((acc[sub],score[sub]))[-1]]
        hits.append(1.0 if isc[pick]==1 else 0.0); ch.append(1.0/len(sub))
    h=np.array(hits); b=h[rng.integers(0,len(h),size=(2000,len(h)))].mean(1)
    return h.mean(),np.percentile(b,2.5),np.percentile(b,97.5),float(np.mean(ch)),len(h)
hdr("P@1 RULES per subclass (restricted to canonical + subclass; vs own chance)")
W("rule","subclass","p_at1","ci_lo","ci_hi","chance","n_genes")
for rnm,sc in [("acceptor_only(acc>=8)",strong),("terminal_like@8",term8)]:
    for snm,m in SUBS:
        p1,lo,hi,ch,ng=p1_rule(sc,m)
        print(f"  {rnm:22s} {snm:16s} P@1 {p1:.4f} [{lo:.4f},{hi:.4f}] chance {ch:.4f} ng {ng}")
        W(rnm,snm,f"{p1:.4f}",f"{lo:.4f}",f"{hi:.4f}",f"{ch:.4f}",ng)

# fitted blocks, 5-fold gene-split cv, per subclass
have=np.isfinite(F1[:,0:60]).all(1)
allg=sorted(set(gene)); rsf=np.random.default_rng(7); permf=rsf.permutation(len(allg))
fold_of={g:permf[i]%NFOLD for i,g in enumerate(allg)}
BLOCKS={"COMP":F1[:,0:60],"SPLICE_arch":F1[:,66:70],"COMP+SPLICE_arch":np.column_stack([F1[:,0:60],F1[:,66:70]])}
def fit_ridge(X,yv,lam):
    P=np.eye(X.shape[1])*lam; P[0,0]=0.0; beta=np.zeros(X.shape[1])
    for _ in range(100):
        eta=np.clip(X@beta,-30,30); mu=1/(1+np.exp(-eta)); w=np.maximum(mu*(1-mu),1e-9)
        z=eta+(yv-mu)/w; XtW=X.T*w
        try: new=np.linalg.solve(XtW@X+P,XtW@z+P@beta)
        except np.linalg.LinAlgError: new=np.linalg.lstsq(XtW@X+P,XtW@z+P@beta,rcond=None)[0]
        if np.max(np.abs(new-beta))<1e-9: beta=new; break
        beta=new
    return beta
def auroc(yv,s):
    order=np.argsort(s,kind="mergesort"); sv=s[order]; n=len(s); rk=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and sv[j+1]==sv[i]: j+=1
        rk[i:j+1]=(i+j)/2.0+1.0; i=j+1
    rr=np.empty(n); rr[order]=rk; npos=yv.sum(); nneg=n-npos
    return np.nan if npos==0 or nneg==0 else (rr[yv==1].sum()-npos*(npos+1)/2.0)/(npos*nneg)
def cv_pred(Xfull,rows):
    yv=isc[rows].astype(float); fold=np.array([fold_of[gene[i]] for i in rows]); pred=np.full(len(rows),np.nan)
    Xr=Xfull[rows]
    for f in range(NFOLD):
        tr=fold!=f; te=fold==f
        if tr.sum()==0 or te.sum()==0: continue
        Xtr=Xr[tr].astype(float).copy(); Xte=Xr[te].astype(float).copy()
        mu=np.nanmean(Xtr,0); sd=np.nanstd(Xtr,0); sd[sd==0]=1.0
        Xtr=np.where(np.isfinite(Xtr),Xtr,mu); Xte=np.where(np.isfinite(Xte),Xte,mu)
        Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
        Xtr=np.column_stack([np.ones(len(Xtr)),Xtr]); Xte=np.column_stack([np.ones(len(Xte)),Xte])
        beta=fit_ridge(Xtr,yv[tr],RIDGE); pred[te]=Xte@beta
    return yv,pred,np.array([gene[i] for i in rows])
def p1_credit_pooled(yv,pred,genes):
    by=defaultdict(list)
    for k in range(len(yv)): by[genes[k]].append(k)
    cred={}; chance={}
    for g,ix in by.items():
        ix=np.array(ix)
        if (yv[ix]==1).sum()!=1 or len(ix)<2: continue
        sc=pred[ix]; mx=np.nanmax(sc)
        if not np.isfinite(mx): continue
        am=sc==mx; cred[g]=float((yv[ix][am]==1).sum())/am.sum(); chance[g]=1.0/len(ix)
    return cred,chance

hdr("FITTED BLOCKS: 5-fold gene-split CV, canonical vs subclass (AUROC + P@1 vs own chance)")
W("block","subclass","n_canon","n_comp","auroc","auroc_lo","auroc_hi","p_at1","p1_lo","p1_hi","chance")
FB={}
for snm,m in SUBS:
    rows=np.where((isc==1)|m)[0]; rows=rows[have[rows]]
    ncan=int((isc[rows]==1).sum()); ncmp=int((isc[rows]==0).sum())
    for bnm,Xf in BLOCKS.items():
        yv,pred,genes=cv_pred(Xf,rows)
        ar=auroc(yv,pred)
        cred,chance=p1_credit_pooled(yv,pred,genes)
        gl=list(cred); credv=np.array([cred[g] for g in gl]); chv=np.array([chance[g] for g in gl])
        p1=credv.mean(); chc=chv.mean()
        gpool=defaultdict(list)
        for k in range(len(yv)): gpool[genes[k]].append(k)
        gukeys=list(gpool)
        arb=[]; p1b=[]
        for _ in range(400):
            samp=[gukeys[j] for j in rng.integers(0,len(gukeys),len(gukeys))]
            idx=np.concatenate([gpool[g] for g in samp]); arb.append(auroc(yv[idx],pred[idx]))
        for _ in range(400):
            s=credv[rng.integers(0,len(credv),len(credv))]; p1b.append(s.mean())
        alo,ahi=np.percentile(arb,[2.5,97.5]); plo,phi=np.percentile(p1b,[2.5,97.5])
        FB[(bnm,snm)]=dict(ar=ar,alo=alo,ahi=ahi,p1=p1,plo=plo,phi=phi,ch=chc)
        print(f"  {bnm:18s} {snm:16s} nC{ncan} nK{ncmp} AUROC {ar:.4f}[{alo:.4f},{ahi:.4f}] P@1 {p1:.4f}[{plo:.4f},{phi:.4f}] chance {chc:.4f}")
        W(bnm,snm,ncan,ncmp,f"{ar:.4f}",f"{alo:.4f}",f"{ahi:.4f}",f"{p1:.4f}",f"{plo:.4f}",f"{phi:.4f}",f"{chc:.4f}")
OUT.close()

order=["CLEAN_FP","CLEAN_FP_nodown","REAL_MINOR","ALL_comp"]
blocks=["COMP","SPLICE_arch","COMP+SPLICE_arch"]; bcol=[COMP_C,CAN,GRN]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.6))
x=np.arange(len(order)); wbar=0.26
for j,b in enumerate(blocks):
    vals=[FB[(b,s)]["p1"] for s in order]
    lo=[FB[(b,s)]["p1"]-FB[(b,s)]["plo"] for s in order]; hi=[FB[(b,s)]["phi"]-FB[(b,s)]["p1"] for s in order]
    ax1.bar(x+(j-1)*wbar,vals,wbar,yerr=[lo,hi],color=bcol[j],capsize=3,label=b,edgecolor="white",linewidth=.5)
for k,s in enumerate(order):
    ch=FB[("COMP",s)]["ch"]
    ax1.plot([x[k]-1.6*wbar,x[k]+1.6*wbar],[ch,ch],color=INK,lw=1.4,ls="--")
ax1.set_xticks(x); ax1.set_xticklabels(order,fontsize=8,rotation=12)
ax1.set_ylabel("P@1 (held-out)"); ax1.set_title("Fitted-block P@1 by subclass (dashed = per-class chance)",fontsize=10,color=INK)
ax1.legend(frameon=False,fontsize=8,loc="upper right"); ax1.set_ylim(0,1)
for sp in ("top","right"): ax1.spines[sp].set_visible(False)
for j,b in enumerate(blocks):
    vals=[FB[(b,s)]["ar"] for s in order]
    lo=[FB[(b,s)]["ar"]-FB[(b,s)]["alo"] for s in order]; hi=[FB[(b,s)]["ahi"]-FB[(b,s)]["ar"] for s in order]
    ax2.bar(x+(j-1)*wbar,vals,wbar,yerr=[lo,hi],color=bcol[j],capsize=3,label=b,edgecolor="white",linewidth=.5)
ax2.axhline(0.5,color=INK,lw=1.4,ls="--")
ax2.set_xticks(x); ax2.set_xticklabels(order,fontsize=8,rotation=12)
ax2.set_ylabel("AUROC (held-out)"); ax2.set_title("Fitted-block AUROC by subclass (dashed = 0.5)",fontsize=10,color=INK)
ax2.legend(frameon=False,fontsize=8,loc="lower right"); ax2.set_ylim(0.4,1)
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
fig.suptitle("What sequence separates a real 3' end from a model false positive (EXPLORATORY, post-hoc)",fontsize=11,color=INK,y=1.02)
fig.tight_layout()
fig.savefig("competitor_subclass.pdf",bbox_inches="tight"); fig.savefig("competitor_subclass.png",dpi=150,bbox_inches="tight")
print("\n[out] competitor_subclass_analysis.tsv  competitor_subclass.pdf/png")
