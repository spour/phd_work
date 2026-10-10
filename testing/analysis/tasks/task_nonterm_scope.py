#!/usr/bin/env python3
# scoped test of exon definition: restrict competitors to non-terminal compartments
import csv, numpy as np
from collections import defaultdict
NFOLD=5; RIDGE=1.0; THR=8.46
rng=np.random.default_rng(42)

R=list(csv.DictReader(open("fig3_new_positional_features.tsv"),delimiter="\t")); NP=len(R)
gene=np.array([r["gene"] for r in R]); isc=np.array([int(float(r["is_canonical"])) for r in R])
cen=np.array([int(r["center"]) for r in R]); strd=np.array([r["strand"] for r in R])
acc=np.array([float(r["acc"]) if r["acc"] not in("","nan") else np.nan for r in R])
comp=np.load("peak_compartment.npy"); F1=np.load("hypothesis_test_features.npy")
Zg=np.load("geom_cache.npz"); max_up=Zg["max_up"]; max_dn=Zg["max_dn"]; max5=Zg["max5"]
At=np.load("atlas_peak_cache.npz"); a_dist=At["dist"]
matched=a_dist<=25

gmap=defaultdict(list)
for i in range(NP): gmap[gene[i]].append(i)
presUP=np.where(np.isfinite(max_up),(max_up>=THR).astype(float),np.nan)
presDN=np.where(np.isfinite(max_dn),(max_dn>=THR).astype(float),np.nan)

C=isc==0
SUBS=[("ALL_comp",C),
      ("NONTERM",C&np.isin(comp,[1,2])),
      ("NONTERM+OTHER",C&np.isin(comp,[1,2,3])),
      ("NONTERM_CLEAN",C&np.isin(comp,[1,2])&(~matched)),
      ("NT+OTH_CLEAN",C&np.isin(comp,[1,2,3])&(~matched))]

# within-gene win-rate
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

# p@1 rules
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

# fitted blocks
have=np.isfinite(F1[:,0:60]).all(1)
allg=sorted(set(gene)); permf=np.random.default_rng(7).permutation(len(allg))
fold_of={g:permf[i]%NFOLD for i,g in enumerate(allg)}
BLOCKS={"SPLICE_arch":F1[:,66:70],"COMP":F1[:,0:60],"COMP+SPLICE_arch":np.column_stack([F1[:,0:60],F1[:,66:70]])}
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
def gene_groups(genes):
    by=defaultdict(list)
    for k in range(len(genes)): by[genes[k]].append(k)
    return {g:np.array(v) for g,v in by.items()}
def p1_cred(yv,pred,gg):
    cred={}; chance={}
    for g,ix in gg.items():
        if (yv[ix]==1).sum()!=1 or len(ix)<2: continue
        sc=pred[ix]; mx=np.nanmax(sc)
        if not np.isfinite(mx): continue
        am=sc==mx; cred[g]=float((yv[ix][am]==1).sum())/am.sum(); chance[g]=1.0/len(ix)
    return cred,chance
def mrr_by_gene(yv,pred,gg):
    rr={}
    for g,ix in gg.items():
        if (yv[ix]==1).sum()!=1 or len(ix)<2: continue
        sc=pred[ix]; sc_can=sc[yv[ix]==1][0]
        if not np.isfinite(sc_can): continue
        greater=int((sc>sc_can).sum()); tied=int((sc==sc_can).sum())
        rank=greater+(tied+1)/2.0; rr[g]=1.0/rank
    return rr

# run
DEF={}
for snm,m in SUBS:
    ng=len({gene[i] for i in np.where(m)[0]})
    rows=np.where((isc==1)|m)[0]; rows=rows[have[rows]]
    ncan=int((isc[rows]==1).sum()); ncmp=int((isc[rows]==0).sum())
    genes0=np.array([gene[i] for i in rows]); gg=gene_groups(genes0)
    preds={}; blk={}
    for bnm,Xf in BLOCKS.items():
        yv,pred,genes=cv_pred(Xf,rows); preds[bnm]=pred
        ar=auroc(yv,pred)
        cred,chance=p1_cred(yv,pred,gg); gl=list(cred)
        credv=np.array([cred[g] for g in gl]); chc=float(np.mean([chance[g] for g in gl]))
        mrrd=mrr_by_gene(yv,pred,gg); mrrv=np.array([mrrd[g] for g in mrrd])
        gk=list(gg); arb=[]
        for _ in range(400):
            samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]
            idx=np.concatenate([gg[g] for g in samp]); arb.append(auroc(yv[idx],pred[idx]))
        p1b=[credv[rng.integers(0,len(credv),len(credv))].mean() for _ in range(400)]
        mrb=[mrrv[rng.integers(0,len(mrrv),len(mrrv))].mean() for _ in range(400)]
        blk[bnm]=dict(ar=ar,alo=np.percentile(arb,2.5),ahi=np.percentile(arb,97.5),
                      p1=credv.mean(),plo=np.percentile(p1b,2.5),phi=np.percentile(p1b,97.5),ch=chc,
                      mrr=mrrv.mean(),mlo=np.percentile(mrb,2.5),mhi=np.percentile(mrb,97.5),cred=cred,mrrd=mrrd)
    yv=isc[rows].astype(float); pc=preds["COMP"]; pb=preds["COMP+SPLICE_arch"]
    d_au=auroc(yv,pb)-auroc(yv,pc)
    gk=list(gg); dau_b=[]
    for _ in range(400):
        samp=[gk[j] for j in rng.integers(0,len(gk),len(gk))]; idx=np.concatenate([gg[g] for g in samp])
        dau_b.append(auroc(yv[idx],pb[idx])-auroc(yv[idx],pc[idx]))
    crC=blk["COMP"]["cred"]; crB=blk["COMP+SPLICE_arch"]["cred"]
    gpe=[g for g in crC if g in crB]; dcred=np.array([crB[g]-crC[g] for g in gpe]); d_p1=dcred.mean()
    dp1_b=[dcred[rng.integers(0,len(dcred),len(dcred))].mean() for _ in range(400)]
    inc=dict(d_au=d_au,daulo=np.percentile(dau_b,2.5),dauhi=np.percentile(dau_b,97.5),
             d_p1=d_p1,dp1lo=np.percentile(dp1_b,2.5),dp1hi=np.percentile(dp1_b,97.5),
             frac_au=float(np.mean(np.array(dau_b)>0)),frac_p1=float(np.mean(np.array(dp1_b)>0)))
    wm={mnm:wg(val,build_pairs(m),hi) for mnm,val,hi in
        [("acc",acc,"canon"),("presUP",presUP,"comp"),("presDN",presDN,"comp")]}
    rules={rnm:p1_rule(sc,m) for rnm,sc in [("acc_only",strong),("term8",term8)]}
    DEF[snm]=dict(np=int(m.sum()),ng=ng,ncan=ncan,ncmp=ncmp,blk=blk,inc=inc,wm=wm,rules=rules)
    print(f"[done] {snm}")

COLS=[s for s,_ in SUBS]
def fmtci(v,lo,hi): return f"{v:.4f}[{lo:.4f},{hi:.4f}]"
rowspec=[]
rowspec.append(("n_peaks",lambda d:d["np"]))
rowspec.append(("n_genes(peaks)",lambda d:d["ng"]))
rowspec.append(("n_canon(fit)",lambda d:d["ncan"]))
rowspec.append(("n_comp(fit)",lambda d:d["ncmp"]))
for b in ["SPLICE_arch","COMP","COMP+SPLICE_arch"]:
    rowspec.append((f"{b} AUROC",lambda d,b=b:fmtci(d["blk"][b]["ar"],d["blk"][b]["alo"],d["blk"][b]["ahi"])))
    rowspec.append((f"{b} P@1",lambda d,b=b:fmtci(d["blk"][b]["p1"],d["blk"][b]["plo"],d["blk"][b]["phi"])))
    rowspec.append((f"{b} chance",lambda d,b=b:f"{d['blk'][b]['ch']:.4f}"))
    rowspec.append((f"{b} MRR",lambda d,b=b:fmtci(d["blk"][b]["mrr"],d["blk"][b]["mlo"],d["blk"][b]["mhi"])))
rowspec.append(("acc winrate(canon>comp)",lambda d:fmtci(d["wm"]["acc"]["wr"],d["wm"]["acc"]["lo"],d["wm"]["acc"]["hi"])))
rowspec.append(("acc excess",lambda d:f"{d['wm']['acc']['exc']:+.4f}"))
rowspec.append(("acc perm_p",lambda d:f"{d['wm']['acc']['p']:.4f}"))
rowspec.append(("acc n_genes",lambda d:d["wm"]["acc"]["ng"]))
rowspec.append(("presUP(anchor,pA) winrate",lambda d:fmtci(d["wm"]["presUP"]["wr"],d["wm"]["presUP"]["lo"],d["wm"]["presUP"]["hi"])))
rowspec.append(("presUP excess/p",lambda d:f"{d['wm']['presUP']['exc']:+.4f} p{d['wm']['presUP']['p']:.4f}"))
rowspec.append(("presDN(pA,pA+d) winrate",lambda d:fmtci(d["wm"]["presDN"]["wr"],d["wm"]["presDN"]["lo"],d["wm"]["presDN"]["hi"])))
rowspec.append(("presDN excess/p",lambda d:f"{d['wm']['presDN']['exc']:+.4f} p{d['wm']['presDN']['p']:.4f}"))
rowspec.append(("P@1 acceptor_only",lambda d:fmtci(d["rules"]["acc_only"][0],d["rules"]["acc_only"][1],d["rules"]["acc_only"][2])))
rowspec.append(("P@1 acc_only chance",lambda d:f"{d['rules']['acc_only'][3]:.4f}"))
rowspec.append(("P@1 terminal_like@8",lambda d:fmtci(d["rules"]["term8"][0],d["rules"]["term8"][1],d["rules"]["term8"][2])))
rowspec.append(("P@1 term8 chance",lambda d:f"{d['rules']['term8'][3]:.4f}"))
rowspec.append(("incr AUROC(+SPL over COMP)",lambda d:fmtci(d["inc"]["d_au"],d["inc"]["daulo"],d["inc"]["dauhi"])))
rowspec.append(("incr AUROC frac_boot>0",lambda d:f"{d['inc']['frac_au']:.3f}"))
rowspec.append(("incr P@1(+SPL over COMP)",lambda d:fmtci(d["inc"]["d_p1"],d["inc"]["dp1lo"],d["inc"]["dp1hi"])))
rowspec.append(("incr P@1 frac_boot>0",lambda d:f"{d['inc']['frac_p1']:.3f}"))

def render():
    w0=max(len(r[0]) for r in rowspec)
    colw=max(28,*(len(c) for c in COLS))
    hdr="  "+ "metric".ljust(w0)+" | "+" | ".join(c.center(colw) for c in COLS)
    print("\n"+hdr); print("  "+"-"*(len(hdr)-2))
    for lab,fn in rowspec:
        cells=[str(fn(DEF[c])).center(colw) for c in COLS]
        print("  "+lab.ljust(w0)+" | "+" | ".join(cells))
render()

with open("competitor_subclass_analysis.tsv","a") as fh:
    W=lambda *a: fh.write("\t".join(str(x) for x in a)+"\n")
    W("#","SCOPED TEST: canonical vs NON-TERMINAL competitors (exon definition applies only to terminal-vs-not).")
    W("#","OTHER = no compartment >=80% of the 500nt upstream window (boundary-straddling OR >=80% intergenic).")
    W("#","Columns compare each restriction to ALL_comp. EXPLORATORY post-hoc scoping.")
    W("metric",*COLS)
    for lab,fn in rowspec:
        W(lab,*[fn(DEF[c]) for c in COLS])
print("\nappended to competitor_subclass_analysis.tsv")
